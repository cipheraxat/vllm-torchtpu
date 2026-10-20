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
"""KV256 union counts and rigorous bounds for adaptive prefill dispatch."""

import jax
import jax.numpy as jnp
from jax import lax


def active_counts(
    blocks: jax.Array, lengths: jax.Array, *, num_tiles: int, query_block: int = 256
) -> jax.Array:
    def one(xs):
        ids, ls = xs
        groups = jnp.concatenate((ids, (ls // 4)[:, None]), axis=1)
        valid = jnp.concatenate(
            ((ids >= 0) & (ls[:, None] > 0), ((ls > 0) & (ls % 4 != 0))[:, None]),
            axis=1,
        )
        indices = jnp.where(valid, groups // 64, num_tiles).reshape(-1)
        touched = (
            jnp.zeros((num_tiles,), jnp.int32)
            .at[indices]
            .max(valid.reshape(-1).astype(jnp.int32), mode="drop")
        )
        return touched.sum()

    return lax.map(
        one,
        (
            blocks.reshape(-1, query_block, blocks.shape[1]),
            lengths.reshape(-1, query_block),
        ),
        batch_size=4,
    )


def span_counts(
    blocks: jax.Array, lengths: jax.Array, *, query_block: int = 256
) -> jax.Array:
    live = (blocks >= 0) & (lengths[:, None] > 0)
    lo = jnp.min(jnp.where(live, blocks // 64, jnp.iinfo(jnp.int32).max), axis=1)
    hi = jnp.max(jnp.where(live, blocks // 64, -1), axis=1)
    tail_live = (lengths > 0) & (lengths % 4 != 0)
    tail = lengths // 256
    lo = jnp.where(tail_live, jnp.minimum(lo, tail), lo)
    hi = jnp.where(tail_live, jnp.maximum(hi, tail), hi)
    lo = lo.reshape(-1, query_block).min(axis=1)
    hi = hi.reshape(-1, query_block).max(axis=1)
    return jnp.where(hi >= 0, hi - lo + 1, 0)


def sampled_metadata(
    blocks: jax.Array, lengths: jax.Array
) -> tuple[jax.Array, jax.Array]:
    # Eight evenly spaced members of each Q256 block. Their tile union is a
    # mathematical lower bound on the full union, never an extrapolation.
    ids = blocks.reshape(-1, 256, blocks.shape[1])[:, ::32]
    ls = lengths.reshape(-1, 256)[:, ::32]
    return ids.reshape(-1, blocks.shape[1]), ls.reshape(-1)


def bucket_upper_counts(
    blocks: jax.Array,
    lengths: jax.Array,
    *,
    num_tiles: int,
    query_block: int = 256,
) -> jax.Array:
    """Bound unions with a 32-bit bitmap of coarse logical tile buckets."""
    tiles = jnp.concatenate((blocks // 64, (lengths // 256)[:, None]), axis=1)
    live = jnp.concatenate(
        (
            (blocks >= 0) & (lengths[:, None] > 0),
            ((lengths > 0) & (lengths % 4 != 0))[:, None],
        ),
        axis=1,
    )
    bits = jnp.iinfo(jnp.uint32).bits
    width = (num_tiles + bits - 1) // bits
    buckets = jnp.where(live, tiles // width, 0).astype(jnp.uint32)
    masks = jnp.where(live, jnp.uint32(1) << buckets, jnp.uint32(0))
    masks = masks.reshape(-1, query_block, masks.shape[-1])
    touched = lax.reduce(masks, jnp.uint32(0), lax.bitwise_or, (1, 2))
    return jnp.minimum(lax.population_count(touched) * width, num_tiles).astype(
        jnp.int32
    )


def exact_counts(blocks: jax.Array, lengths: jax.Array, *, num_tiles: int) -> jax.Array:
    """Count full KV256 unions using sequential bitwise reductions."""
    from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
        UNION_WORD_BATCH_SIZE,
    )

    tiles = jnp.concatenate((blocks // 64, (lengths // 256)[:, None]), axis=1)
    live = jnp.concatenate(
        (
            (blocks >= 0) & (lengths[:, None] > 0),
            ((lengths > 0) & (lengths % 4 != 0))[:, None],
        ),
        axis=1,
    )
    word_bits = jnp.iinfo(jnp.uint32).bits
    words = (tiles // word_bits).reshape(-1, 256, tiles.shape[-1])
    bits = jnp.where(
        live,
        jnp.uint32(1) << (tiles % word_bits).astype(jnp.uint32),
        jnp.uint32(0),
    ).reshape(words.shape)

    def one(word):
        masks = jnp.where(words == word, bits, jnp.uint32(0))
        union = lax.reduce(masks, jnp.uint32(0), lax.bitwise_or, (1, 2))
        return lax.population_count(union).astype(jnp.int32)

    return lax.map(
        one,
        jnp.arange((num_tiles + word_bits - 1) // word_bits),
        batch_size=UNION_WORD_BATCH_SIZE,
    ).sum(axis=0)
