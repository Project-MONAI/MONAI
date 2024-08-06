# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import os
import pickle
import threading
from collections import OrderedDict

import torch

from monai.apps.utils import get_logger

from .module import optional_import

P, P_imported = optional_import("polygraphy")
if P_imported:
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.onnx import fold_constants, onnx_from_path, save_onnx
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
    dynamic_axes = {}
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
        raise RuntimeError(
            f"CUDA ERROR: {err}"
        )
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
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph

    def build(self, onnx_path, profiles=None, update_output_names=False, **config_kwargs):
        """
        Builds TRT engine from ONNX file at onnx_path and sets self.engine
        Args:
             update_output_names: if set, use update_output_names as output names
             profiles, config_kwargs: passed to TRT's engine_from_network()
        """

        LOGGER.info(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")

        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        if update_output_names:
            LOGGER.info(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)

        LOGGER.info("Calling engine_from_network...")

        engine = engine_from_network(network, config=CreateConfig(profiles=profiles, **config_kwargs))
        self.engine = engine

    def save(self):
        save_engine(self.engine, path=self.engine_path)

    def load(self):
        LOGGER.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, profile_num=0, reuse_device_memory=None):
        """
        Creates execution context for self.engine and activates it
        """
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        self.dtypes = []
        dtype_dict = trt_to_torch_dtype_dict()
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_names.append(binding)
            elif self.engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(binding)
                dtype = dtype_dict[self.engine.get_tensor_dtype(binding)]
                self.dtypes.append(dtype)
        self.cur_profile = profile_num

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
    and running with fallback to Torch (for TRT modules with limited profiles)

    """

    def __init__(self, path, model=None, input_names=None, output_names=None, use_cuda_graph=False, timestamp=None):
        super().__init__()
        self.input_names = input_names
        self.output_names = output_names
        self.model = model
        self.profiles = None
        self.engine = None
        self.jit_model = None
        self.onnx_runner = None
        self.path = path
        self.use_cuda_graph = use_cuda_graph

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

    def load_engine(self):
        """
        Loads TRT plan from disk and activates its execution context.
        """
        try:
            engine = Engine(self.engine_path)
            engine.load()
            engine.activate()
            self.engine = engine
        except Exception as e:
            LOGGER.debug(f"Exception while loading the engine:\n{e}")

    def load_jit(self):
        """
        Loads Torchscript from disk
        """
        try:
            self.jit_model = torch.jit.load(self.jit_path)
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

    def inputs_to_dict(self, input_example):
        """
        Converts list of inputs indo a dict usable with TRT engine
        """
        trt_inputs = {}
        for i, inp in enumerate(input_example):
            input_name = self.engine.input_names[i]
            trt_inputs[input_name] = inp
        return trt_inputs

    def forward(self, **args):
        """
        Main forward method: depending on TRT/Torchscript/ONNX representation available,
        runs appropriate accelerated method. If exception thrown, falls back to original Pytorch
        """
        try:
            if self.engine is not None:
                # forward_trt is not thread safe as we do not use per-thread execution contexts
                with lock_sm:
                    return self.forward_trt(args)
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
            LOGGER.info(f"Exception: {e}\nFalling back to Pytorch ...")

        return self.model.forward(**args)

    def forward_trt(self, trt_inputs):
        """
        Auxiliary method to run TRT engine.
        Sets input bindings from trt_inputs, allocates memory and runs activated TRT engine
        """
        stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self.engine.set_inputs(trt_inputs, stream.cuda_stream)
        self.engine.allocate_buffers(torch.device("cuda"))
        # Need this to synchronize with Torch stream
        stream.wait_stream(torch.cuda.current_stream())
        ret = self.engine.infer(stream.cuda_stream, use_cuda_graph=self.use_cuda_graph)
        ret = list(ret.values())

        if len(ret) == 1:
            ret = ret[0]
        return ret

    def build_engine(self, input_profiles=None, **build_args):
        """
        Builds TRT engine from ONNX file at self.onnx_path and sets self.engine
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

        engine = Engine(self.path + ".plan")
        engine.build(self.onnx_path, profiles, **build_args)
        engine.activate()
        self.engine = engine

    def jit_export(self, input_example, verbose=False):
        """
        Exports self.model to Torchscript at self.jit_path and sets self.jit_model
        """
        self.jit_model = torch.jit.trace(self.model, input_example).eval()
        self.jit_model = torch.jit.freeze(self.jit_model)
        torch.jit.save(self.jit_model, self.jit_path)

    def onnx_export(
        self, input_example, dynamo=False, onnx_registry=None, dynamic_shapes=None, verbose=False, opset_version=18
    ):
        """
        Exports self.model to ONNX file at self.onnx_path
        Args: passed to onnx.export()
        """

        LOGGER.info(f"Exporting to ONNX, dynamic shapes: {dynamic_shapes}")
        model = self.model
        from .export_utils import replace_for_export

        replace_for_export(model, do_cast=True)

        torch.onnx.export(
            model,
            input_example,
            self.onnx_path,
            dynamo=dynamo,
            verbose=verbose,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=dynamic_shapes,
        )
        LOGGER.info("Folding constants...")
        model_onnx = onnx_from_path(self.onnx_path)
        fold_constants(model_onnx, allow_onnxruntime_shape_inference=False)
        LOGGER.info("Done folding constants.")

        LOGGER.info("Saving model...")
        save_onnx(model_onnx, self.onnx_path)
        LOGGER.info("Done saving model.")

    def build_and_save(self, input_example, dynamo=False, verbose=False, input_profiles=None, **build_args):
        """
        If serialized engine is not found, exports self.model to ONNX,
        builds TRT engine and saves serialized TRT engine to the disk.
        Args:
             input_example, dynamo, verbose:  passed to self.onnx_export()
             input_profiles: used to get dynamic axes for onnx_export(),
                             passed to self.build_engine()
             build_args : passed to self.build_engine()
        enable_all_tactics=True,
        """
        if not self.has_engine():
            try:
                if not self.has_onnx():
                    self.onnx_export(
                        input_example, dynamo=dynamo, dynamic_shapes=get_dynamic_axes(input_profiles), verbose=verbose
                    )
                self.build_engine(input_profiles=input_profiles, **build_args)
                self.engine.save()
                os.remove(self.onnx_path)
            except Exception as e:
                raise e
