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

from collections import OrderedDict
import os
import pickle
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.onnx import onnx_from_path, fold_constants, save_onnx
from polygraphy.backend.onnxrt import OnnxrtRunner, session_from_onnx
from polygraphy.backend.trt import TrtRunner, CreateConfig, ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.logger import G_LOGGER as L_

import tensorrt as trt
import torch

from cuda import cudart

import threading

from monai.apps.utils import get_logger
LOGGER=get_logger("run_cmd")

lock_sm = threading.Lock()

# Map of torch dtype -> numpy dtype
trt_to_torch_dtype_dict = {
    trt.int32: torch.int32,
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.bfloat16: torch.float16,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}

def get_dynamic_axes(profiles, extra_axes={}):
    dynamic_axes=extra_axes
    for profile in profiles:
        for key in profile:
            axes=[]
            vals=profile[key]
            for i in range(len(vals[0])):
                if vals[0][i] != vals[2][i]:
                    axes.append(i)
            if len(axes) > 0:
                dynamic_axes[key] = axes
        return dynamic_axes

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != 0:
         raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None

class ShapeException(Exception):
    pass


class Engine:
    """
    An auxiliary class to implement running of TRT optimized engines

    """
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None # cuda graph

    def build(
        self,
        onnx_path,
        profiles=[],
        fp16=False,
        bf16=False,
        tf32=True,
        builder_optimization_level=3,
        enable_all_tactics=True,
        direct_io=False,
        timing_cache=None,
        update_output_names=None,
    ):
        L_.info(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        config_kwargs = {
            "builder_optimization_level": builder_optimization_level,
            "direct_io": direct_io,
        }
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        network = network_from_onnx_path(
            onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
        )
        if update_output_names:
            L_.info(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        # with L.verbosity(0):
        L_.info("Calling engine_from_network...")

        engine = engine_from_network(
            network,
            config=CreateConfig(
                fp16=fp16,
                bf16=bf16,
                tf32=tf32,
                profiles=profiles,
                load_timing_cache=timing_cache,
                **config_kwargs,
            ),
            save_timing_cache=timing_cache,
        )
        self.engine = engine

    def save(self):
        save_engine(self.engine, path=self.engine_path)

    def load(self):
        L_.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, profile_num=0, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        self.dtypes = []
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_names.append(binding)
            elif self.engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(binding)
                dtype = trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(binding)]
                self.dtypes.append(dtype)
        self.cur_profile = profile_num
        # L_.info(self.input_names)
        # L_.info(self.output_names)

    def allocate_buffers(self, device):
        # allocate outputs
        ctx = self.context

        for i, binding in enumerate(self.output_names):
            shape = ctx.get_tensor_shape(binding)
            t = torch.empty(
                list(shape), dtype=self.dtypes[i], device=device
            ).contiguous()
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
        e = self.engine
        ctx = self.context
        last_profile = self.cur_profile

        def try_set_inputs():
            for binding, t in feed_dict.items():
                if t is not None:
                    t = t.contiguous()
                    shape = t.shape
                    # mincurmax = list(e.get_profile_shape(self.cur_profile, binding))
                    # if not self.check_shape(shape, mincurmax):
                    #    raise ShapeException(f"Input shape to be set is outside the bounds: {binding} -> {shape}, profile is {mincurmax}, trying another profile: {self.cur_profile}")
                    ctx.set_input_shape(binding, shape)
                    ctx.set_tensor_address(binding, t.data_ptr())

        while True:
            try:
                try_set_inputs()
                break
            except ShapeException:
                next_profile = (self.cur_profile + 1) % e.num_optimization_profiles
                if next_profile == last_profile:
                    raise
                self.cur_profile = next_profile
                ctx.set_optimization_profile_async(self.cur_profile, stream)
                # torch.cuda.synchronize()

        left = ctx.infer_shapes()
        assert len(left) == 0

    def infer(self, stream, use_cuda_graph=False):
        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                CUASSERT(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(
                    cudart.cudaStreamBeginCapture(
                        stream,
                        cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
                    )
                )
                self.context.execute_async_v3(stream)
                graph = CUASSERT(cudart.cudaStreamEndCapture(stream))
                self.cuda_graph_instance = CUASSERT(
                    cudart.cudaGraphInstantiate(graph, 0)
                )
                LOGGER.info("CUDA Graph captured!")
        else:
            noerror = self.context.execute_async_v3(stream)
            CUASSERT(cudart.cudaStreamSynchronize(stream))
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors


class TRTWrapper(torch.nn.Module):
    """
    This wrapper implements TRT, ONNX and Torchscript persistent export
    and running with fallback to Torch (for TRT modules with limited profiles)

    """

    def __init__(self,
                 path,
                 model=None,
                 input_names=None,
                 output_names=None,
                 use_cuda_graph=False,
                 timestamp=None):
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
            ftime=os.path.getmtime(self.onnx_path)
            if timestamp is not None and ftime < timestamp:
                os.remove(self.onnx_path)
            else:
                timestamp = ftime
        if timestamp is not None and os.path.exists(self.engine_path) and os.path.getmtime(self.engine_path) < timestamp:
            os.remove(self.engine_path)


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
        try:
            engine = Engine(self.engine_path)
            engine.load()
            engine.activate()
            self.engine = engine
        except Exception as e:
            LOGGER.debug(f"Exception while loading the engine:\n{e}")

    def load_jit(self):
        try:
            self.jit_model = torch.jit.load(self.jit_path)
        except Exception:
            pass

    def load_onnx(self, providers=["CUDAExecutionProvider"]):
        try:
            onnx_runner = OnnxrtRunner(
                session_from_onnx(self.onnx_path, providers=providers)
            )
            onnx_runner.activate()
            self.onnx_runner = onnx_runner
        except Exception:
            pass

    def load_profiles(self):
        with open(self.profiles_path, "rb") as fp:
            profiles = pickle.load(fp)
        self.profiles = profiles
        return profiles

    def save_profiles(self):
        with open(self.profiles_path, "wb") as fp:
            pickle.dump(self.profiles, fp)

    def inputs_to_dict(self, input_example):
        trt_inputs = {}
        for i, inp in enumerate(input_example):
            input_name = self.engine.input_names[i]
            trt_inputs[input_name] = inp
        return trt_inputs

    def forward(self, **args):
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

    def forward_trt_runner(self, trt_inputs):
        with TrtRunner(self.engine) as runner:
            ret = runner.infer(trt_inputs)
        ret = list(ret.values())
        ret = [r.cuda() for r in ret]
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def build_engine(
        self,
        input_profiles=[],
        fp16=False,
        bf16=False,
        tf32=False,
        builder_optimization_level=3,
        direct_io=False,
        enable_all_tactics=True,
    ):
        profiles = []
        if len(input_profiles) > 0:
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
        engine.build(
            self.onnx_path,
            profiles,
            fp16=fp16,
            bf16=bf16,
            tf32=tf32,
            direct_io=direct_io,
            builder_optimization_level=builder_optimization_level,
            enable_all_tactics=enable_all_tactics,
        )
        engine.activate()
        self.engine = engine

    def jit_export(
        self,
        input_example,
        verbose=False,
    ):
        self.jit_model = torch.jit.trace(
            self.model,
            input_example,
        ).eval()
        self.jit_model = torch.jit.freeze(self.jit_model)
        torch.jit.save(self.jit_model, self.jit_path)

    def onnx_export(
        self,
        input_example,
        dynamo=False,
        onnx_registry=None,
        dynamic_shapes=None,
        verbose=False,
        opset_version=18,
    ):
        L_.info(f"Exporting to ONNX, dynamic shapes: {dynamic_shapes}")
        model = self.model
        from .export_utils import replace_for_export

        replace_for_export(model, do_cast=True)

        if dynamo:
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
                dynamic_shapes=dynamic_shapes,
            )
        else:
            torch.onnx.export(
                model,
                input_example,
                self.onnx_path,
                verbose=verbose,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=self.input_names,
                output_names=self.output_names,
                dynamic_axes=dynamic_shapes,
            )
        L_.info("Folding constants...")
        model_onnx = onnx_from_path(self.onnx_path)
        fold_constants(model_onnx, allow_onnxruntime_shape_inference=False)
        L_.info("Done folding constants.")

        L_.info("Saving model...")
        save_onnx(
            model_onnx,
            self.onnx_path,
        )
        L_.info("Done saving model.")

    def build_and_save(
        self,
        input_example,
        dynamo=False,
        verbose=False,
        input_profiles=[],
        fp16=False,
        bf16=False,
        tf32=True,
        builder_optimization_level=3,
        direct_io=False,
        enable_all_tactics=True,
    ):
        if not self.has_engine():
            try:
                if not self.has_onnx():
                    self.onnx_export(
                        input_example,
                        dynamo=dynamo,
                        dynamic_shapes=get_dynamic_axes(input_profiles),
                        verbose=verbose,
                    )
                self.build_engine(
                    input_profiles=input_profiles,
                    fp16=fp16, tf32=tf32,
                    direct_io=direct_io,
                    builder_optimization_level=5,
                    enable_all_tactics=enable_all_tactics)
                self.engine.save()
                os.remove(self.onnx_path)
            except Exception as e:
                raise e
