# SPDX-License-Identifier: Apache-2.0
"""Union-aware bounded prefill, measured TPU7x Q24/KV2/D256 only.

Uses the low-memory output schedule from standalone qsa_prefill v0.4.
Dispatch uses work bounds and calibrated per-stage costs; query buckets
are aligned to Q256 and subdivided into bounded outer chunks.

Keeps the validated TC paired-page kernel for reusable/small KV unions. SC
bounds selected-KV compute for large unions. Dispatch is outside both kernels;
all metadata and branch costs belong to the end-to-end call.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as candidate
from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import (
    select_sc,
    tc_always_wins,
)
from vllm_torchtpu.kernels.sparse_gqa_prefill.grouped import sc_grouped_attention
from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import SC_CHUNK_SIZE


def adaptive_prefill(
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    starts: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Attend within the measured envelope; cache is returned unchanged.

    The single request starts at zero. Inactive padded rows have length zero
    and group IDs -1; each active row has a unique valid prefix of selected
    complete groups. KV scales are one. Callers gate all other configurations
    with is_supported_by_adaptive_prefill before entering this function.
    """
    if not is_supported_by_adaptive_prefill(
        q,
        gate,
        blocks,
        lengths,
        table,
        cache,
        starts,
        kv_heads=2,
        ratio=4,
        k_scale=1.0,
        v_scale=1.0,
    ):
        raise NotImplementedError(
            "Adaptive prefill requires the measured TPU7x single-request "
            "Q24/KV2/D256, page1664, T multiple of 256 envelope; "
            "use block_sparse_gqa_attention for other configurations"
        )
    outer_chunk = math.gcd(len(q), 8192)
    sc_chunk_size = SC_CHUNK_SIZE[jnp.dtype(cache.dtype)]
    tc = functools.partial(
        candidate.block_sparse_gqa_attention,
        kv_heads=2,
        ratio=4,
        query_block=256,
        kv_tile=256,
    )
    sc = functools.partial(_sc_prefill, chunk_size=sc_chunk_size)
    num_tiles = (table.shape[1] * cache.shape[1] + 255) // 256
    static_tc = tc_always_wins(
        queries=outer_chunk, num_tiles=num_tiles, dtype=cache.dtype
    )

    def step(kv, xs):
        query, gate, ids, ls, pages = xs
        choose = select_sc(ids, ls, num_tiles=num_tiles, dtype=kv.dtype)
        result, kv = lax.cond(
            choose,
            lambda args: sc(*args),
            lambda args: tc(*args),
            (query, gate, ids, ls, pages, kv, jnp.array([0, outer_chunk], jnp.int32)),
        )
        return kv, result

    xs = tuple(
        x.reshape(-1, outer_chunk, *x.shape[1:])
        for x in (q, gate, blocks, lengths, table)
    )

    # Reuse one bounded metadata pass for both the whole-call TC fast path
    # and individual chunks. Only one boolean per chunk stays live.
    choices = None
    if len(q) > outer_chunk and not static_tc:
        choices = lax.map(
            lambda data: select_sc(
                data[0], data[1], num_tiles=num_tiles, dtype=cache.dtype
            ),
            (xs[2], xs[3]),
        )

    def run_adaptive(inputs):
        kv, inputs = inputs
        if len(q) == outer_chunk:
            return lax.scan(step, kv, inputs, unroll=1)
        full = tuple(x.reshape(-1, *x.shape[2:]) for x in inputs)
        output = jnp.empty_like(full[0])

        def block(i, state):
            kv, output = state
            offset = i * outer_chunk
            assert choices is not None
            choose = choices[i]

            def tc_block(state):
                kv, output = state
                ys = tuple(
                    lax.dynamic_slice_in_dim(x, offset, outer_chunk) for x in full
                )
                result, kv = tc(
                    ys[0],
                    ys[1],
                    ys[2],
                    ys[3],
                    ys[4],
                    kv,
                    jnp.array([0, outer_chunk], jnp.int32),
                )
                output = lax.dynamic_update_slice_in_dim(output, result, offset, axis=0)
                return kv, output

            def sc_block(state):
                def micro(j, state):
                    kv, output = state
                    start = offset + j * sc_chunk_size
                    ys = tuple(
                        lax.dynamic_slice_in_dim(x, start, sc_chunk_size) for x in full
                    )
                    result, kv = sc_grouped_attention(
                        ys[0],
                        ys[1],
                        ys[2],
                        ys[3],
                        ys[4],
                        kv,
                        kv_heads=2,
                        ratio=4,
                        query_group=8,
                        tile_k=256,
                    )
                    output = lax.dynamic_update_slice_in_dim(
                        output, result, start, axis=0
                    )
                    return kv, output

                return lax.fori_loop(0, outer_chunk // sc_chunk_size, micro, state)

            return lax.cond(choose, sc_block, tc_block, (kv, output))

        kv, output = lax.fori_loop(0, len(q) // outer_chunk, block, (kv, output))
        return kv, output.reshape(-1, outer_chunk, *q.shape[1:])

    def run_tc(inputs):
        kv, inputs = inputs

        def tc_step(kv, ys):
            result, kv = tc(
                ys[0],
                ys[1],
                ys[2],
                ys[3],
                ys[4],
                kv,
                jnp.array([0, outer_chunk], jnp.int32),
            )
            return kv, result

        return lax.scan(tc_step, kv, inputs, unroll=1)

    if static_tc:
        cache, out = run_tc((cache, xs))
    elif len(q) == outer_chunk:
        cache, out = run_adaptive((cache, xs))
    else:
        assert choices is not None
        possible = jnp.any(choices)
        cache, out = lax.cond(possible, run_adaptive, run_tc, (cache, xs))
    return out.reshape(q.shape), cache


def _sc_prefill(q, gate, blocks, lengths, table, cache, starts, *, chunk_size):
    def step(kv, inputs):
        out, kv = sc_grouped_attention(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            kv,
            kv_heads=2,
            ratio=4,
            query_group=8,
            tile_k=256,
        )
        return kv, out

    cache, output = lax.scan(
        step,
        cache,
        tuple(
            x.reshape(-1, chunk_size, *x.shape[1:])
            for x in (q, gate, blocks, lengths, table)
        ),
        unroll=1,
    )
    return output.reshape(q.shape), cache


def is_supported_by_adaptive_prefill(
    query: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    starts: jax.Array,
    *,
    kv_heads: int,
    ratio: int,
    k_scale: float,
    v_scale: float,
) -> bool:
    """Static eligibility only; request metadata obeys qsa_attention's contract."""
    if (
        query.ndim != 3
        or query.shape[1:] != (24, 256)
        or not 256 <= query.shape[0] <= 262144
        or query.shape[0] % 256
        or gate.shape != query.shape
        or query.dtype != jnp.bfloat16
        or gate.dtype != jnp.bfloat16
        or kv_heads != 2
        or ratio != 4
        or starts.shape != (2,)
        or cache.ndim != 5
        or cache.shape[0] == 0
        or cache.shape[1] != 1664
        or cache.dtype not in (jnp.bfloat16, jnp.float8_e4m3fn)
        or cache.size * jax.dtypes.itemsize_bits(cache.dtype) // 8 > 16 * 2**30
        or blocks.shape != (len(query), 512)
        or lengths.shape != (len(query),)
        or table.ndim != 2
        or table.shape[0] != len(query)
        or not 1 <= table.shape[1] <= (262144 + 1663) // 1664
        or any(x.dtype != jnp.int32 for x in (blocks, lengths, table, starts))
        or not isinstance(k_scale, (int, float))
        or not isinstance(v_scale, (int, float))
        or k_scale != 1.0
        or v_scale != 1.0
    ):
        return False
    packing = 32 // jax.dtypes.itemsize_bits(cache.dtype)
    if cache.shape[2:] != (4 // packing, packing, 256):
        return False
    info = pltpu.get_tpu_info()
    return (
        info.chip_version == pltpu.ChipVersion.TPU_7X and info.sparse_core is not None
    )
