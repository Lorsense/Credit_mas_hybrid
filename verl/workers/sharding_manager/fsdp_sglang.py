# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import socket

import torch
import torch.distributed as dist
from sglang.srt.entrypoints.engine import Engine
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available
from verl.utils.sglang_weight_sync import (
    DEFAULT_WEIGHT_BUCKET_BYTES,
    STRICT_QWEN35_LOADER,
    flush_weight_caches,
    is_qwen35_config,
    iter_weight_buckets,
    update_weight_bucket,
    validate_local_tp_topology,
)

from .base import BaseShardingManager

# from vllm.distributed import parallel_state as sglang_ps
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


class FSDPSGLangShardingManager(BaseShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.qwen35 = is_qwen35_config(model_config)
        self.weight_update_bucket_bytes = DEFAULT_WEIGHT_BUCKET_BYTES

        # Full params
        self.full_params = full_params
        # Engine receives full HF tensors; DTensor state dicts from the device
        # mesh are gathered lazily below, leaving native inference TP slicing to
        # SGLang. Keep the existing FSDP1 full/sharded state-dict choice.
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()
        # Engine's public tensor update shares CUDA tensors through local IPC.
        # Same-node TP is supported; spanning hosts needs a distributed weight
        # update group instead of trying to deserialize another host's handle.
        topology = [None] * dist.get_world_size()
        dist.all_gather_object(topology, (socket.gethostname(), self.device_mesh["infer_tp"].mesh.tolist()))
        validate_local_tp_topology(topology)

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        torch.cuda.empty_cache()
        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)
        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        device = torch.cuda.current_device()  # used when fsdp2 set cpu_offload_policy
        params = {k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()}
        # Copy, not share memory
        self.update_weights(params)
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

        del params
        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        self.release_memory()
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def update_weights(self, params):
        if not params:
            raise ValueError("Cannot synchronize an empty Actor state dict")
        self._run_engine_operation(lambda: self.inference_engine.resume_memory_occupation())
        load_format = STRICT_QWEN35_LOADER if self.qwen35 else None
        for bucket in iter_weight_buckets(
            params.items(),
            max_bytes=self.weight_update_bucket_bytes,
            materialize=_preprocess_tensor_for_update_weights,
        ):
            self._run_engine_operation(lambda bucket=bucket: update_weight_bucket(self.inference_engine, bucket, load_format=load_format))
            del bucket
        # A separate checked flush prevents a rejected/busy cache reset from
        # silently reusing either KV or recurrent states from the old policy.
        self._run_engine_operation(lambda: flush_weight_caches(self.inference_engine))

    def _run_engine_operation(self, operation):
        """Propagate any engine failure before peers enter the next all-gather."""
        error = None
        if self.tp_rank == 0:
            try:
                operation()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        # All FSDP ranks participate, including separate rollout DP engines.
        # Otherwise one failed engine leaves peers hanging in DTensor.full_tensor.
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        errors = [message for message in errors if message is not None]
        if errors:
            raise RuntimeError("SGLang weight synchronization aborted; rollout weights may be partial: " + "; ".join(errors))

    def release_memory(self):
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            self.inference_engine.release_memory_occupation()

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = self.device_mesh["infer_tp"].get_group()

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
