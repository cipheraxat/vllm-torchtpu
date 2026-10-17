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

"""Compact decode task assignment and fused vector DMA-schedule expansion."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, schedule


def _kernel(meta_ref, *out_refs, cfgs, names):
    steps, lanes = cfgs.max_steps_ub, cfgs.batch_size

    def lane(row):
        return meta_ref[row, :, :lanes]

    def gather(row, indices):
        return jnp.take_along_axis(
            meta_ref[row, :, :],
            jnp.broadcast_to(indices[None, :], (steps, indices.size)),
            axis=1,
        )

    col = jnp.arange(lanes * cfgs.bkv_p_cache * 3)
    idx, page, field = (
        col // (cfgs.bkv_p_cache * 3),
        (col // 3) % cfgs.bkv_p_cache,
        col % 3,
    )
    seq, k, length, live = (gather(row, idx) for row in range(4))
    src = jnp.minimum(
        seq * cfgs.serve.pages_per_seq + k * cfgs.bkv_p + page[None, :],
        cfgs.serve.num_page_indices - 1,
    )
    dst = jnp.broadcast_to(page[None, :] * cfgs.serve.page_size, src.shape)
    size = jnp.clip(length - k * cfgs.bkv_sz - dst, 0, cfgs.serve.page_size)
    dma = jnp.where(
        live != 0,
        jnp.where(field[None, :] == 0, src, jnp.where(field[None, :] == 1, dst, size)),
        0,
    )
    qcol = jnp.arange(lanes * 2)
    dq = jnp.where(qcol[None, :] % 2 == 0, gather(4, qcol // 2), gather(3, qcol // 2))
    live = lane(3)
    count = jnp.sum(live, axis=1, keepdims=True)
    last = lane(5)
    outs = jnp.sum(last, axis=1, keepdims=True)
    kv_count = jnp.sum(
        jnp.where(
            live != 0, jnp.clip(lane(2) - lane(1) * cfgs.bkv_sz, 0, cfgs.bkv_sz), 0
        ),
        axis=1,
        keepdims=True,
    )
    values = dict(
        s_idx=jnp.where(live != 0, lane(0), -1),
        k_idx=jnp.where(live != 0, lane(1), 0),
        is_last_k=last,
        dma_q=dq,
        dma_kv_cache=dma,
        total_wait_kv_in=kv_count * cfgs.kv_bytes_per_token // 512,
        total_wait_q_in=count * cfgs.q_bytes_per_token // 512,
        total_wait_o_out=outs * cfgs.o_bytes_per_token // 512,
        total_wait_lse_out=outs
        * cfgs.model.num_kv_heads
        * cfgs.aligned_num_q_heads_per_kv_head,
    )
    for name, ref in zip(names, out_refs):
        ref[...] = values[name] if name in values else jnp.zeros(ref.shape, jnp.int32)


def generate(
    cu_q_lens: jax.Array, lengths: jax.Array, cfgs: configs.RpaConfigs
) -> schedule.RpaSchedule:
    shape = schedule.BaseMetadataComputer.get_rpa_schedule(cfgs)
    steps, lanes = cfgs.max_steps_ub, cfgs.batch_size
    assert steps * lanes >= len(lengths) * pl.cdiv(
        cfgs.serve.pages_per_seq * cfgs.serve.page_size, cfgs.bkv_sz
    )
    tiles = (lengths + cfgs.bkv_sz - 1) // cfgs.bkv_sz
    ends = jnp.cumsum(tiles)
    slots = jnp.arange(steps * lanes).reshape(steps, lanes)
    seq = jnp.minimum(jnp.sum(slots[..., None] >= ends, axis=-1), len(lengths) - 1)
    begins = jnp.concatenate((jnp.zeros(1, jnp.int32), ends[:-1]))
    k = slots - begins[seq]
    live = slots < ends[-1]
    last = live & (k == tiles[seq] - 1)
    meta = jnp.stack(
        (
            seq,
            k,
            lengths[seq],
            live.astype(jnp.int32),
            jnp.where(live, cu_q_lens[seq], 0),
            last.astype(jnp.int32),
        )
    )
    meta = jnp.pad(meta, ((0, 0), (0, 0), (0, pltpu.get_tpu_info().num_lanes - lanes)))
    names = [
        f.name
        for f in dataclasses.fields(shape)
        if f.name not in ("actual_steps", "cfgs")
    ]
    out_shape = [
        jax.ShapeDtypeStruct(
            (steps, getattr(shape, name).data.size // steps), jnp.int32
        )
        for name in names
    ]
    outputs = pl.pallas_call(
        functools.partial(_kernel, cfgs=cfgs, names=names),
        out_shape=out_shape,
        name="qsa_decode_schedule",
    )(meta)
    result = {
        name: dataclasses.replace(getattr(shape, name), data=value.reshape(-1))
        for name, value in zip(names, outputs)
    }
    return dataclasses.replace(
        shape,
        **result,
        actual_steps=((ends[-1:] + lanes - 1) // lanes).astype(jnp.int32),
    )
