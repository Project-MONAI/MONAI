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

import inspect
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence

import torch

from monai.apps.utils import get_logger
from monai.networks.utils import add_casts_around_norms, convert_to_onnx
from monai.utils.module import optional_import

P, P_imported = optional_import("polygraphy")
if P_imported:
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import (
        CreateConfig,
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


class TRTEngine:
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
    This wrapper implements:
      - TRT lazy persistent export
      - Running TRT with optional fallback to Torch
        (for TRT engines with limited profiles)
    """

    def __init__(
        self,
        path: str,
        model: torch.nn.Module | None = None,
        precision: str = "tf32",
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        export_args: Mapping | None = None,
        build_args: Mapping | None = None,
        input_profiles: Mapping | None = None,
        dynamic_batchsize: Sequence[int] | None = None,
        use_cuda_graph: bool = False,
        timestamp: int | None = None,
        fallback: bool = False,
    ):
        """
        Initialization method:
         Tries to load persistent serialized TRT engine
         Saves arguments for lazy TRT build on first forward() call
        Args:
            path : Path where to save persistent serialized TRT engine,
            model: Model to "wrap". Can be None if TRT engine is supposed to exist.
            input_names: Optional list of output names to pass to onnx.export()
            output_names: Optional list of output names to pass to onnx.export()
            export_args: Optional args to pass to onnx.export(). See onnx.export() for details.
            build_args: Optional args to pass to TRT builder. See polygraphy.Config for details
            input_profiles: Optional list of profiles for TRT builder and ONNX export.
                            Each profile is a map of "input name" -> [min,opt,max] values.
            dynamic_batchsize: A sequence with three elements to define the batch size range of the input for the model to be
                               converted. Should be a sequence like [MIN_BATCH, OPT_BATCH, MAX_BATCH].
            If both input_profiles and dynamic_batchsize are omitted, static shapes will be used to build TRT engine.
            use_cuda_graph: Use CUDA Graph for inference(all inputs have to be the same GPU memory between calls!)
            timestamp: Optional timestamp to rebuild TRT engine (e.g. if config file changes)
            fallback: Allow to fall back to Pytorch when TRT inference fails (e.g, shapes exceed max profile)
        """

        super().__init__()
        self.path = path
        self.model = model
        self.precision = precision
        self.output_names = output_names or []
        self.profiles = input_profiles or []
        self.dynamic_batchsize = dynamic_batchsize
        self.export_args = export_args or {}
        self.build_args = build_args or {}
        self.engine: TRTEngine | None = None
        self.use_cuda_graph = use_cuda_graph
        self.fallback = fallback
        self.disabled = False

        # Force engine rebuild if older than the timestamp passed
        if (
            timestamp is not None
            and os.path.exists(self.engine_path)
            and os.path.getmtime(self.engine_path) < timestamp
        ):
            os.remove(self.engine_path)

        # Normally we read input_names from forward() but can be overridden
        if input_names is None and self.model is not None:
            argspec = inspect.getfullargspec(self.model.forward)
            input_names = argspec.args[1:]
        self.input_names = input_names

        self._load_engine()

    """
    Auxiliary getters/setters
    """

    @property
    def engine_path(self):
        return self.path + ".plan"

    @property
    def onnx_path(self):
        return self.path + ".onnx"

    def _inputs_to_dict(self, input_example):
        trt_inputs = {}
        for i, inp in enumerate(input_example):
            input_name = self.input_names[i]
            trt_inputs[input_name] = inp
        return trt_inputs

    def _load_engine(self):
        """
        Loads TRT plan from disk and activates its execution context.
        """
        try:
            self.engine = TRTEngine(self.engine_path)
            self.input_names = self.engine.input_names
            if self.fallback and self.model is not None:
                del self.model
                self.model = None
        except Exception as e:
            LOGGER.debug(f"Exception while loading the engine:\n{e}")

    def forward(self, *argv, **kwargs):
        """
        Main forward method:
         Builds TRT engine if not available yet.
         Tries to run TRT engine
         If exception thrown and self.callback==True: falls back to original Pytorch

        Args: Passing through whatever args wrapped module's forward() has
        Returns: Passing through wrapped module's forward() return value(s)

        """
        try:
            if len(argv) > 0:
                kwargs.update(self._inputs_to_dict(argv))
                argv = []

            if self.engine is None and not self.disabled:
                self._build_and_save(kwargs)
            if self.engine is not None:
                # forward_trt is not thread safe as we do not use per-thread execution contexts
                with lock_sm:
                    device = torch.cuda.current_device()
                    stream = torch.cuda.Stream(device=device)
                    self.engine.set_inputs(kwargs, stream.cuda_stream)
                    self.engine.allocate_buffers(device=device)
                    # Need this to synchronize with Torch stream
                    stream.wait_stream(torch.cuda.current_stream())
                    ret = self.engine.infer(stream.cuda_stream, use_cuda_graph=self.use_cuda_graph)
                    ret = list(ret.values())

                    if len(ret) == 1:
                        ret = ret[0]
                    return ret
        except Exception as e:
            if self.model:
                LOGGER.info(f"Exception: {e}\nFalling back to Pytorch ...")
            else:
                raise e
        return self.model.forward(*argv, **kwargs)

    def _onnx_to_trt(self):
        """
        Builds TRT engine from ONNX file at self.onnx_path and saves to self.trt_path
        """

        profiles = []
        if self.profiles:
            for input_profile in self.profiles:
                if isinstance(input_profile, Profile):
                    profiles.append(input_profile)
                else:
                    p = Profile()
                    for name, dims in input_profile.items():
                        assert len(dims) == 3
                        p.add(name, min=dims[0], opt=dims[1], max=dims[2])
                    profiles.append(p)

        build_args = self.build_args.copy()
        build_args["tf32"] = self.precision != "fp32"
        build_args["fp16"] = self.precision == "fp16"
        build_args["bf16"] = self.precision == "bf16"

        LOGGER.info(f"Building TensorRT engine for {self.onnx_path}: {self.engine_path}")
        network = network_from_onnx_path(self.onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        engine = engine_from_network(network, config=CreateConfig(profiles=profiles, **build_args))
        save_engine(engine, path=self.engine_path)

    def _build_and_save(self, input_example):
        """
        If TRT engine is not ready, exports self.model to ONNX,
        builds TRT engine and saves serialized TRT engine to the disk.
        Args:
             input_example: passed to onnx.export()
        """
        if self.engine is not None:
            return

        if self.model is None:
            raise ValueError("ERROR: self.model is None!")

        try:
            export_args = self.export_args
            dbs = self.dynamic_batchsize
            if dbs:
                if len(self.profiles) > 0:
                    raise ValueError("ERROR: Both dynamic_batchsize and input_profiles set for TRTWrapper!")
                if len(dbs) != 3:
                    raise ValueError("dynamic_batchsize has to have len ==3 ")
                profiles = {}
                for id, val in input_example.items():
                    sh = val.shape[1:]
                    profiles[id] = [[dbs[0], *sh], [dbs[1], *sh], [dbs[2], *sh]]
                self.profiles = [profiles]

            if len(self.profiles) > 0:
                export_args.update({"dynamic_axes": get_dynamic_axes(self.profiles)})

            LOGGER.info(f"Exporting to {self.onnx_path}, export args: {export_args}")
            add_casts_around_norms(self.model)
            convert_to_onnx(
                self.model,
                (input_example,),
                filename=self.onnx_path,
                input_names=self.input_names,
                output_names=self.output_names,
                **export_args,
            )
            LOGGER.info("Export to ONNX successful.")
            self._onnx_to_trt()
            self._load_engine()
            os.remove(self.onnx_path)
        except Exception as e:
            if self.fallback:
                LOGGER.info(f"Failed to build engine: {e}")
                self.disabled = True
            else:
                raise e
