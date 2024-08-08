# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import pickle
import threading
from collections import OrderedDict

import torch

from monai.apps.utils import get_logger

from .export_utils import onnx_export
from .module import optional_import

P, P_imported = optional_import("polygraphy")
if P_imported:
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.onnxrt import OnnxrtRunner, session_from_onnx
    from polygraphy.backend.trt import (
        CreateConfig,
        ModifyNetworkOutputs,
        Profile,
        engine_from_bytes,
        engine_from_network,
        network_from_onnx_path,
        save_engine,
    )

trt, trt_imported = optional_import("tensorrt")
cudart, _ = optional_import("cuda.cudart")

LOGGER = get_logger("run_cmd")

lock_sm = threading.Lock()


# Map of TRT dtype -> Torch dtype
def trt_to_torch_dtype_dict():
    return {
        trt.int32: torch.int32,
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.float16,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }


def get_dynamic_axes(profiles):
    """
    Given [[min,opt,max],...] list of profile dimensions,
    this method calculates dynamic_axes to use in onnx.export()
    """
    dynamic_axes: dict[str, list[int]] = {}
    if not profiles:
        return dynamic_axes
    for profile in profiles:
        for key in profile:
            axes = []
            vals = profile[key]
            for i in range(len(vals[0])):
                if vals[0][i] != vals[2][i]:
                    axes.append(i)
            if len(axes) > 0:
                dynamic_axes[key] = axes
    return dynamic_axes


def cuassert(cuda_ret):
    """
    Error reporting method for CUDA calls
    """
    err = cuda_ret[0]
    if err != 0:
        raise RuntimeError(f"CUDA ERROR: {err}")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class ShapeError(Exception):
    """
    Exception class to report errors from setting TRT plan input shapes
    """

    pass


class Engine:
    """
    An auxiliary class to implement running of TRT optimized engines

    """

    def __init__(self, engine_path):
        """
        Loads serialized engine, creates execution context and activates it
        """
        self.engine_path = engine_path
        LOGGER.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        self.dtypes = []
        self.cur_profile = 0
        dtype_dict = trt_to_torch_dtype_dict()
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_names.append(binding)
            elif self.engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(binding)
                dtype = dtype_dict[self.engine.get_tensor_dtype(binding)]
                self.dtypes.append(dtype)

    def allocate_buffers(self, device):
        """
        Allocates outputs to run TRT engine
        """
        ctx = self.context

        for i, binding in enumerate(self.output_names):
            shape = ctx.get_tensor_shape(binding)
            t = torch.empty(list(shape), dtype=self.dtypes[i], device=device).contiguous()
            self.tensors[binding] = t
            ctx.set_tensor_address(binding, t.data_ptr())

    @staticmethod
    def check_shape(shape, profile):
        shape = list(shape)
        minlist = profile[0]
        maxlist = profile[2]
        good = True
        for i, s in enumerate(shape):
            if s < minlist[i] or s > maxlist[i]:
                good = False
        return good

    def set_inputs(self, feed_dict, stream):
        """
        Sets input bindings for TRT engine according to feed_dict
        """
        e = self.engine
        ctx = self.context

        last_profile = self.cur_profile

        def try_set_inputs():
            for binding, t in feed_dict.items():
                if t is not None:
                    t = t.contiguous()
                    shape = t.shape
                    # TODO: port to new TRT10 API
                    # mincurmax = list(e.get_profile_shape(self.cur_profile, binding))
                    # if not self.check_shape(shape, mincurmax):
                    #    raise ShapeError(f"Input shape to be set is outside the bounds: {binding} -> {shape}")
                    ctx.set_input_shape(binding, shape)
                    ctx.set_tensor_address(binding, t.data_ptr())

        while True:
            try:
                try_set_inputs()
                break
            except ShapeError:
                next_profile = (self.cur_profile + 1) % e.num_optimization_profiles
                if next_profile == last_profile:
                    raise
                self.cur_profile = next_profile
                ctx.set_optimization_profile_async(self.cur_profile, stream)

        left = ctx.infer_shapes()
        assert len(left) == 0

    def infer(self, stream, use_cuda_graph=False):
        """
        Runs TRT engine.
        Note use_cuda_graph requires all inputs to be the same GPU memory between calls.
        """
        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                cuassert(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                cuassert(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                cuassert(
                    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal)
                )
                self.context.execute_async_v3(stream)
                graph = cuassert(cudart.cudaStreamEndCapture(stream))
                self.cuda_graph_instance = cuassert(cudart.cudaGraphInstantiate(graph, 0))
                LOGGER.info("CUDA Graph captured!")
        else:
            noerror = self.context.execute_async_v3(stream)
            cuassert(cudart.cudaStreamSynchronize(stream))
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors


class TRTWrapper(torch.nn.Module):
    """
    This wrapper implements TRT, ONNX and Torchscript persistent export
    and running with optional fallback to Torch (for TRT modules with limited profiles)
    """

    def __init__(
        self,
        path,
        model=None,
        input_names=None,
        output_names=None,
        use_cuda_graph=False,
        timestamp=None,
        fallback=False,
    ):
        super().__init__()
        self.input_names = input_names
        self.output_names = output_names
        self.model = model
        self.profiles = None
        self.engine: Engine | None = None
        self.jit_model = None
        self.onnx_runner = None
        self.path = path
        self.use_cuda_graph = use_cuda_graph
        self.fallback = fallback

        if os.path.exists(self.onnx_path):
            ftime = os.path.getmtime(self.onnx_path)
            if timestamp is not None and ftime < timestamp:
                os.remove(self.onnx_path)
            else:
                timestamp = ftime
        if (
            timestamp is not None
            and os.path.exists(self.engine_path)
            and os.path.getmtime(self.engine_path) < timestamp
        ):
            os.remove(self.engine_path)

    """
    Auxiliary getters/setters
    """

    @property
    def engine_path(self):
        return self.path + ".plan"

    @property
    def jit_path(self):
        return self.path + ".ts"

    @property
    def onnx_path(self):
        return self.path + ".onnx"

    @property
    def profiles_path(self):
        return self.path + ".profiles.pkl"

    def has_engine(self):
        return self.engine is not None

    def has_onnx(self):
        return os.path.exists(self.onnx_path)

    def has_jit(self):
        return os.path.exists(self.jit_path)

    def has_profiles(self):
        return os.path.exists(self.profiles_path)

    def delete_model(self):
        if self.fallback and self.model is not None:
            del self.model
            self.model = None

    def load_engine(self):
        """
        Loads TRT plan from disk and activates its execution context.
        """
        try:
            self.engine = Engine(self.engine_path)
            self.delete_model()
        except Exception as e:
            LOGGER.debug(f"Exception while loading the engine:\n{e}")

    def load_jit(self):
        """
        Loads Torchscript from disk
        """
        try:
            self.jit_model = torch.jit.load(self.jit_path)
            self.delete_model()
        except Exception:
            pass

    def load_onnx(self, providers=None):
        """
        Loads ONNX from disk and creates/activates OnnxrtRunner runner for it.
        """
        if providers is None:
            providers = ["CUDAExecutionProvider"]
        try:
            onnx_runner = OnnxrtRunner(session_from_onnx(self.onnx_path, providers=providers))
            onnx_runner.activate()
            self.onnx_runner = onnx_runner
            self.delete_model()
        except Exception:
            pass

    def load_profiles(self):
        """
        Loads saved optimization profiles from disk
        """
        with open(self.profiles_path, "rb") as fp:
            profiles = pickle.load(fp)
        self.profiles = profiles
        return profiles

    def save_profiles(self):
        """
        Saves optimization profiles to disk using pickle
        """
        with open(self.profiles_path, "wb") as fp:
            pickle.dump(self.profiles, fp)

    def forward(self, **args):
        """
        Main forward method: depending on TRT/Torchscript/ONNX representation available,
        runs appropriate accelerated method. If exception thrown, falls back to original Pytorch
        """
        try:
            if self.engine is not None:
                # forward_trt is not thread safe as we do not use per-thread execution contexts
                with lock_sm:
                    device = torch.cuda.current_device()
                    stream = torch.cuda.Stream(device=device)
                    self.engine.set_inputs(args, stream.cuda_stream)
                    self.engine.allocate_buffers(device=device)
                    # Need this to synchronize with Torch stream
                    stream.wait_stream(torch.cuda.current_stream())
                    ret = self.engine.infer(stream.cuda_stream, use_cuda_graph=self.use_cuda_graph)
                    ret = list(ret.values())

                    if len(ret) == 1:
                        ret = ret[0]
                    return ret
            elif self.jit_model is not None:
                return self.jit_model.forward(**args)
            elif self.onnx_runner is not None:
                ret = self.onnx_runner.infer(args)
                ret = list(ret.values())
                ret = [r.cuda() for r in ret]
                if len(ret) == 1:
                    ret = ret[0]
                return ret
        except Exception as e:
            if self.model:
                LOGGER.info(f"Exception: {e}\nFalling back to Pytorch ...")
            else:
                raise e
        return self.model.forward(**args)

    def onnx_to_trt(self, input_profiles=None, **build_args):
        """
        Builds TRT engine from ONNX file at self.onnx_path and saves to self.trt_path
        Args:
             input_profiles, build_args - passed to engine.build()
        """

        profiles = []
        if input_profiles:
            for input_profile in input_profiles:
                if isinstance(input_profile, Profile):
                    profiles.append(input_profile)
                else:
                    p = Profile()
                    for name, dims in input_profile.items():
                        assert len(dims) == 3
                        p.add(name, min=dims[0], opt=dims[1], max=dims[2])
                    profiles.append(p)
            self.profiles = profiles
            self.save_profiles()

        LOGGER.info(f"Building TensorRT engine for {self.onnx_path}: {self.engine_path}")

        network = network_from_onnx_path(self.onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        if self.output_names and False:
            LOGGER.info(f"Updating network outputs to {self.output_names}")
            network = ModifyNetworkOutputs(network, self.output_names)

        LOGGER.info("Calling engine_from_network...")

        engine = engine_from_network(network, config=CreateConfig(profiles=profiles, **build_args))
        save_engine(engine, path=self.engine_path)

    def build_and_save(self, input_example, export_args=None, input_profiles=None, **build_args):
        """
        If serialized engine is not found, exports self.model to ONNX,
        builds TRT engine and saves serialized TRT engine to the disk.
        Args:
             input_example, export_args:  passed to self.onnx_export()
             input_profiles: used to get dynamic axes for onnx_export(),
                             passed to self.build_engine()
             build_args : passed to onnx_to_trt()
        enable_all_tactics=True,
        """
        if not export_args:
            export_args: dict = {}
        if input_profiles:
            export_args.update({"dynamic_axes": get_dynamic_axes(input_profiles)})

        if not self.has_engine():
            try:
                if not self.has_onnx():
                    LOGGER.info(f"Exporting to {self.onnx_path}, export args: {export_args}")
                    onnx_export(
                        self.model,
                        input_example,
                        self.onnx_path,
                        input_names=self.input_names,
                        output_names=self.output_names,
                        **export_args,
                    )
                    LOGGER.info("Export to ONNX successful.")
                self.onnx_to_trt(input_profiles=input_profiles, **build_args)
                self.load_engine()
                os.remove(self.onnx_path)
            except Exception as e:
                LOGGER.info(f"Failed to build engine: {e}")
