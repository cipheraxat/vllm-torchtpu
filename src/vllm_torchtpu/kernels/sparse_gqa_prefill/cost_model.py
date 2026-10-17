# SPDX-License-Identifier: Apache-2.0
"""Work-based TC/SC dispatch for Q24/KV2/D256 sparse prefill.

Tile bounds concern execution work, not latency guarantees. Coefficients
are nonnegative so substituting a union bound preserves cost ordering.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.sparse_gqa_prefill.occupancy import (
    active_counts,
    bucket_upper_counts,
    exact_counts,
    sampled_metadata,
    span_counts,
)


class PrefillCostModel(NamedTuple):
    tc_fixed_ms: float
    tc_query_ms: float
    tc_mask_word_ms: float
    tc_tile_ms: float
    sc_fixed_ms: float
    sc_query_ms: float
    sc_attention_tile_ms: float


# Filled once from the frozen calibration set, before held-out validation.
COST_MODELS: dict[jnp.dtype, PrefillCostModel] = {
    jnp.dtype(jnp.bfloat16): PrefillCostModel(
        0.25000650408450037,
        8.991181022104898e-05,
        4.814218452898145e-08,
        0.004999265026206821,
        0.19743081215678976,
        0.005833960374709633,
        0.002476169547627867,
    ),
    jnp.dtype(jnp.float8_e4m3fn): PrefillCostModel(
        0.2491017824685177,
        8.889156791776907e-05,
        4.724934780077619e-08,
        0.005145153550878911,
        0.19896896616666457,
        0.0032255933216442456,
        0.0038026017010685054,
    ),
}


def tc_always_wins(*, queries: int, num_tiles: int, dtype: jnp.dtype) -> bool:
    """Prove TC wins even against the minimum scheduled SparseCore work."""
    model = COST_MODELS[jnp.dtype(dtype)]
    words = ((2 * num_tiles + 127) // 128) * 128
    upper_tc = (
        model.tc_fixed_ms
        + model.tc_query_ms * queries
        + model.tc_mask_word_ms * queries * words
        + model.tc_tile_ms * (queries // 256) * num_tiles
    )
    lower_sc = (
        model.sc_fixed_ms
        + model.sc_query_ms * queries
        + model.sc_attention_tile_ms * (queries // 8)
    )
    return upper_tc <= lower_sc * 1.02


def tc_cost(
    model: PrefillCostModel, *, queries: int, num_tiles: int, tile_visits: jax.Array
) -> jax.Array:
    words = ((2 * num_tiles + 127) // 128) * 128
    return (
        model.tc_fixed_ms
        + model.tc_query_ms * queries
        + model.tc_mask_word_ms * queries * words
        + model.tc_tile_ms * tile_visits.astype(jnp.float32)
    )


def sc_cost(model: PrefillCostModel, lengths: jax.Array) -> jax.Array:
    # Gather always transports the padded capacity, including inactive rows.
    # Attention executes one tile even for an all-inactive group of 8 queries.
    selected = jnp.minimum(lengths // 4, 512) * 4 + lengths % 4
    tiles = jnp.maximum(1, (selected.reshape(-1, 8).max(axis=1) + 255) // 256)
    return (
        model.sc_fixed_ms
        + model.sc_query_ms * lengths.size
        + model.sc_attention_tile_ms * tiles.astype(jnp.float32).sum()
    )


def tc_upper_bound_wins(
    blocks: jax.Array, lengths: jax.Array, *, num_tiles: int, dtype: jnp.dtype
) -> jax.Array:
    """Prove the TC work estimate wins using all rows, never sampled rows."""
    if tc_always_wins(queries=lengths.size, num_tiles=num_tiles, dtype=dtype):
        return jnp.array(True)
    model = COST_MODELS[jnp.dtype(dtype)]
    target = sc_cost(model, lengths) * 1.02

    def estimate(visits):
        return tc_cost(
            model,
            queries=lengths.size,
            num_tiles=num_tiles,
            tile_visits=jnp.asarray(visits),
        )

    return lax.cond(
        estimate((lengths.size // 256) * num_tiles) <= target,
        lambda _: jnp.array(True),
        lambda _: estimate(span_counts(blocks, lengths).sum()) <= target,
        None,
    )


def select_sc(
    blocks: jax.Array, lengths: jax.Array, *, num_tiles: int, dtype: jnp.dtype
) -> jax.Array:
    if tc_always_wins(queries=lengths.size, num_tiles=num_tiles, dtype=dtype):
        return jnp.array(False)
    model = COST_MODELS[jnp.dtype(dtype)]
    # A 2% reserve favors TC near the modeled crossover; no claim of a
    # universal latency bound is made by this fitted model.
    target = sc_cost(model, lengths) * 1.02

    def estimate(visits):
        return tc_cost(
            model, queries=lengths.size, num_tiles=num_tiles, tile_visits=visits
        )

    def refine_upper(_):
        upper = bucket_upper_counts(blocks, lengths, num_tiles=num_tiles).sum()
        return lax.cond(
            estimate(upper) <= target,
            lambda _: jnp.array(False),
            lambda _: estimate(exact_counts(blocks, lengths, num_tiles=num_tiles).sum())
            > target,
            None,
        )

    def unresolved(_):
        ids, ls = sampled_metadata(blocks, lengths)
        lower = active_counts(ids, ls, num_tiles=num_tiles, query_block=8).sum()
        return lax.cond(
            estimate(lower) > target,
            lambda _: jnp.array(True),
            refine_upper,
            None,
        )

    return lax.cond(
        tc_upper_bound_wins(blocks, lengths, num_tiles=num_tiles, dtype=dtype),
        lambda _: jnp.array(False),
        unresolved,
        None,
    )
