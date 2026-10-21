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

# Fixed geometry of the calibrated schedule, not hardware capacity limits.
_TC_QUERY_BLOCK = 256
_KV_TILE = 256
_SC_QUERY_GROUP = 8
_SAMPLED_QUERIES_PER_BLOCK = 8
_COMPRESSION_RATIO = 4
_MAX_SELECTED_GROUPS = 512
_MASK_WORD_ALIGNMENT = 128
_MASK_WORDS_PER_KV_TILE = _KV_TILE // _COMPRESSION_RATIO // jnp.iinfo(jnp.uint32).bits

# Policy margin, not a prediction-error bound. Keep the original float literal.
_TC_PREFERENCE_FACTOR = 1.02


class PrefillCostModel(NamedTuple):
    tc_fixed_ms: float
    tc_query_ms: float
    tc_mask_word_ms: float
    tc_tile_ms: float
    sc_fixed_ms: float
    sc_query_ms: float
    sc_attention_tile_ms: float


# Frozen fit and feature units: docs/developers_guide/sparse_gqa_prefill_cost_model.md
COST_MODELS: dict[jnp.dtype, PrefillCostModel] = {
    jnp.dtype(jnp.bfloat16): PrefillCostModel(
        tc_fixed_ms=0.25000650408450037,
        tc_query_ms=8.991181022104898e-05,
        tc_mask_word_ms=4.814218452898145e-08,
        tc_tile_ms=0.004999265026206821,
        sc_fixed_ms=0.19743081215678976,
        sc_query_ms=0.005833960374709633,
        sc_attention_tile_ms=0.002476169547627867,
    ),
    jnp.dtype(jnp.float8_e4m3fn): PrefillCostModel(
        tc_fixed_ms=0.2491017824685177,
        tc_query_ms=8.889156791776907e-05,
        tc_mask_word_ms=4.724934780077619e-08,
        tc_tile_ms=0.005145153550878911,
        sc_fixed_ms=0.19896896616666457,
        sc_query_ms=0.0032255933216442456,
        sc_attention_tile_ms=0.0038026017010685054,
    ),
}


def tc_always_wins(
    *, queries: int, num_tiles: int, dtype: jnp.dtype, sc_queries: int | None = None
) -> bool:
    """Prove TC wins even against the minimum scheduled SparseCore work."""
    model = COST_MODELS[jnp.dtype(dtype)]
    words = (
        (_MASK_WORDS_PER_KV_TILE * num_tiles + (_MASK_WORD_ALIGNMENT - 1))
        // _MASK_WORD_ALIGNMENT
    ) * _MASK_WORD_ALIGNMENT
    upper_tc = (
        model.tc_fixed_ms
        + model.tc_query_ms * queries
        + model.tc_mask_word_ms * queries * words
        + model.tc_tile_ms * (queries // _TC_QUERY_BLOCK) * num_tiles
    )
    scheduled_sc = queries if sc_queries is None else sc_queries
    lower_sc = (
        model.sc_fixed_ms
        + model.sc_query_ms * scheduled_sc
        + model.sc_attention_tile_ms * (scheduled_sc // _SC_QUERY_GROUP)
    )
    return upper_tc <= lower_sc * _TC_PREFERENCE_FACTOR


def tc_cost(
    model: PrefillCostModel, *, queries: int, num_tiles: int, tile_visits: jax.Array
) -> jax.Array:
    words = (
        (_MASK_WORDS_PER_KV_TILE * num_tiles + (_MASK_WORD_ALIGNMENT - 1))
        // _MASK_WORD_ALIGNMENT
    ) * _MASK_WORD_ALIGNMENT
    return (
        model.tc_fixed_ms
        + model.tc_query_ms * queries
        + model.tc_mask_word_ms * queries * words
        + model.tc_tile_ms * tile_visits.astype(jnp.float32)
    )


def sc_cost(model: PrefillCostModel, lengths: jax.Array) -> jax.Array:
    # Gather always transports the padded capacity, including inactive rows.
    # Attention executes one tile even for an all-inactive group of 8 queries.
    selected = (
        jnp.minimum(lengths // _COMPRESSION_RATIO, _MAX_SELECTED_GROUPS)
        * _COMPRESSION_RATIO
        + lengths % _COMPRESSION_RATIO
    )
    tiles = jnp.maximum(
        1,
        (selected.reshape(-1, _SC_QUERY_GROUP).max(axis=1) + (_KV_TILE - 1))
        // _KV_TILE,
    )
    return (
        model.sc_fixed_ms
        + model.sc_query_ms * lengths.size
        + model.sc_attention_tile_ms * tiles.astype(jnp.float32).sum()
    )


def tc_upper_bound_wins(
    blocks: jax.Array,
    lengths: jax.Array,
    *,
    num_tiles: int,
    dtype: jnp.dtype,
    sc_queries: int | None = None,
) -> jax.Array:
    """Prove the TC work estimate wins using all rows, never sampled rows."""
    if tc_always_wins(
        queries=lengths.size, num_tiles=num_tiles, dtype=dtype, sc_queries=sc_queries
    ):
        return jnp.array(True)
    model = COST_MODELS[jnp.dtype(dtype)]
    # TC pads to Q256, whereas SC pads only to its gather microchunk.
    target = sc_cost(model, lengths[:sc_queries]) * _TC_PREFERENCE_FACTOR

    def estimate(visits):
        return tc_cost(
            model,
            queries=lengths.size,
            num_tiles=num_tiles,
            tile_visits=jnp.asarray(visits),
        )

    return lax.cond(
        estimate((lengths.size // _TC_QUERY_BLOCK) * num_tiles) <= target,
        lambda _: jnp.array(True),
        lambda _: estimate(span_counts(blocks, lengths).sum()) <= target,
        None,
    )


def select_sc(
    blocks: jax.Array,
    lengths: jax.Array,
    *,
    num_tiles: int,
    dtype: jnp.dtype,
    sc_queries: int | None = None,
) -> jax.Array:
    if tc_always_wins(
        queries=lengths.size, num_tiles=num_tiles, dtype=dtype, sc_queries=sc_queries
    ):
        return jnp.array(False)
    model = COST_MODELS[jnp.dtype(dtype)]
    # A 2% reserve favors TC near the modeled crossover; no claim of a
    # universal latency bound is made by this fitted model.
    # TC pads to Q256, whereas SC pads only to its gather microchunk.
    target = sc_cost(model, lengths[:sc_queries]) * _TC_PREFERENCE_FACTOR

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
        lower = active_counts(
            ids, ls, num_tiles=num_tiles, query_block=_SAMPLED_QUERIES_PER_BLOCK
        ).sum()
        return lax.cond(
            estimate(lower) > target,
            lambda _: jnp.array(True),
            refine_upper,
            None,
        )

    return lax.cond(
        tc_upper_bound_wins(
            blocks, lengths, num_tiles=num_tiles, dtype=dtype, sc_queries=sc_queries
        ),
        lambda _: jnp.array(False),
        unresolved,
        None,
    )
