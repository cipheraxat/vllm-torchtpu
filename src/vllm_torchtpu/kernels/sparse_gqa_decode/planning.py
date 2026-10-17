# Copyright 2026 Google LLC
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
"""Derive launch bounds from the unchanged RPA buffers and schedule."""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, schedule
from vllm_torchtpu.kernels.sparse_gqa_decode.tuned_block_sizes import DECODE_BLOCK


def make_config(
    batch: int,
    heads: int,
    dim: int,
    kv_heads: int,
    ratio: int,
    groups: int,
    cache_dtype: jnp.dtype,
) -> configs.RpaConfigs:
    return configs.RpaConfigs(
        block=DECODE_BLOCK,
        model=configs.ModelConfigs(
            num_q_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=dim,
            sm_scale=dim**-0.5,
            mask_value=float(jnp.finfo(jnp.float32).min),
        ),
        serve=configs.ServingConfigs(
            num_seqs=batch,
            page_size=ratio,
            total_q_tokens=batch,
            num_page_indices=batch * (groups + 1),
            dtype_q=jnp.dtype(jnp.float32),
            dtype_out=jnp.dtype(jnp.float32),
            dtype_kv=cache_dtype,
            skip_kv_update=True,
        ),
        vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        mode=configs.RpaCase.DECODE,
    )


def fits_launch(cfgs: configs.RpaConfigs) -> bool:
    """Check a complete schedule window and conservatively budget VMEM."""
    info = pltpu.get_tpu_info()
    n = cfgs.serve.num_seqs
    tiles = pl.cdiv(cfgs.serve.pages_per_seq * cfgs.serve.page_size, cfgs.bkv_sz)
    if pl.cdiv(n * tiles, cfgs.batch_size) > cfgs.max_steps_ub:
        return False
    metadata = schedule.BaseMetadataComputer.get_rpa_schedule(cfgs)
    schedule_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(metadata))
    # Actual scalar prefetch: cu_q_lens, offsets, q_lens and virtual pages.
    scalar_bytes = (3 * n + 1 + cfgs.serve.num_page_indices) * 4
    # Match RpaConfigs.max_steps_ub's compiler reserve. Its minimum one vector
    # of steps can exceed SMEM for small virtual pages, so check actual bytes.
    if scalar_bytes + schedule_bytes > info.smem_capacity_bytes - 32 * 1024:
        return False

    q_bytes = math.prod(cfgs.q_vmem_shape) * cfgs.serve.dtype_q.itemsize
    kv_bytes = math.prod(cfgs.kv_vmem_shape) * cfgs.serve.dtype_kv.itemsize
    buffers = (
        configs.NUM_INPUT_SLOTS * (q_bytes + kv_bytes)
        + configs.NUM_OUTPUT_SLOTS * q_bytes
    )
    scratches = (
        2 * math.prod(cfgs.lm_scratch_shape) + math.prod(cfgs.acc_scratch_shape)
    ) * cfgs.serve.dtype_out.itemsize
    # Same live Q/KV/QK working-set estimate as RPA.calculate_block_sizes,
    # plus the fused entry's resident BF16 gate and FP32 gate intermediates.
    heads, dim = cfgs.model.num_q_heads, cfgs.aligned_q_head_dim
    working = cfgs.batch_size * (
        heads * dim * 4
        + cfgs.bkv_sz * cfgs.model.num_kv_heads * dim * cfgs.serve.dtype_kv.itemsize
        + cfgs.bkv_sz * heads * 4
        + heads * dim * 4
    )
    gate = n * heads * dim * jnp.dtype(jnp.bfloat16).itemsize
    # Keep the existing RPA planner's 20% headroom for compiler layout/temps.
    return buffers + scratches + working + gate <= cfgs.vmem_limit_bytes * 0.8


def query_batch_size(
    batch: int,
    heads: int,
    dim: int,
    kv_heads: int,
    ratio: int,
    groups: int,
    cache_dtype: jnp.dtype,
) -> int:
    """Largest power-of-two launch fitting resources; zero means use RPA."""
    size = 1 << (batch - 1).bit_length()
    while size:
        cfgs = make_config(size, heads, dim, kv_heads, ratio, groups, cache_dtype)
        if fits_launch(cfgs):
            return size
        size //= 2
    return 0
