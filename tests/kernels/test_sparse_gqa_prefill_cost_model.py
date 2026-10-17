# SPDX-License-Identifier: Apache-2.0
"""Work bounds must not hide dispersed queries or partial groups."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import select_sc


@pytest.mark.parametrize("fp8", [False, True])
def test_unsampled_disjoint_queries_cannot_be_mistaken_for_local_work(fp8):
    """Identical sampled queries must not conceal the work of other rows."""
    rng = np.random.default_rng(2917)
    tokens = 512
    lengths = np.arange(262144 - tokens + 1, 262145, dtype=np.int32)
    recent = np.stack([np.arange(n // 4 - 512, n // 4) for n in lengths])
    hidden = np.stack([rng.choice(n // 4, 512, replace=False) for n in lengths])
    hidden[::32] = recent[::32]
    np.testing.assert_array_equal(hidden[::32], recent[::32])
    route = jax.jit(
        functools.partial(
            select_sc, num_tiles=1027, dtype=jnp.float8_e4m3fn if fp8 else jnp.bfloat16
        )
    )
    assert not bool(route(jnp.asarray(recent), jnp.asarray(lengths)))
    assert bool(route(jnp.asarray(hidden), jnp.asarray(lengths)))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("num_tiles", [33, 257, 1027])
def test_union_bounds_include_partial_tails_and_inactive_queries(seed, num_tiles):
    from vllm_torchtpu.kernels.sparse_gqa_prefill.occupancy import (
        active_counts,
        bucket_upper_counts,
        exact_counts,
        sampled_metadata,
        span_counts,
    )

    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, num_tiles * 256 + 1, 512, dtype=np.int32)
    lengths[::7] = 0
    lengths[1::31] = 1
    ids = np.full((512, 512), -1, np.int32)
    for row, length in enumerate(lengths):
        n = min(length // 4, 512)
        ids[row, :n] = rng.choice(length // 4, n, replace=False)
    expected = []
    for start in range(0, 512, 256):
        tiles = set()
        for row in range(start, start + 256):
            tiles.update(int(g) // 64 for g in ids[row] if g >= 0)
            if lengths[row] % 4:
                tiles.add(int(lengths[row]) // 256)
        expected.append(len(tiles))
    blocks, ls = jnp.asarray(ids), jnp.asarray(lengths)
    exact = np.asarray(active_counts(blocks, ls, num_tiles=num_tiles))
    lower = np.asarray(
        active_counts(*sampled_metadata(blocks, ls), num_tiles=num_tiles, query_block=8)
    )
    upper = np.asarray(span_counts(blocks, ls))
    np.testing.assert_array_equal(exact, expected)
    assert np.all(lower <= exact)
    assert np.all(exact <= upper)
    refined = np.asarray(bucket_upper_counts(blocks, ls, num_tiles=num_tiles))
    assert np.all(exact <= refined)
    assert np.all(refined <= num_tiles)
    coverage = np.asarray(exact_counts(blocks, ls, num_tiles=num_tiles))
    np.testing.assert_array_equal(exact, coverage)
    assert np.all(coverage <= num_tiles)


def test_exact_bitmap_counts_a_distant_partial_tail():
    from vllm_torchtpu.kernels.sparse_gqa_prefill.occupancy import (
        bucket_upper_counts,
        exact_counts,
    )

    ids = jnp.broadcast_to(jnp.arange(512, dtype=jnp.int32), (256, 512))
    lengths = jnp.full((256,), 262141, jnp.int32)
    # Eight complete KV256 tiles plus a partial group in tile 1023.
    np.testing.assert_array_equal(exact_counts(ids, lengths, num_tiles=1027), [9])
    assert int(bucket_upper_counts(ids, lengths, num_tiles=1027)[0]) >= 9
    np.testing.assert_array_equal(
        exact_counts(ids, jnp.zeros_like(lengths), num_tiles=1027),
        [0],
    )
