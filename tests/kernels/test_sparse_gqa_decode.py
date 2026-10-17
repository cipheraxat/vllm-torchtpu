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

"""Decode dispatch, arithmetic, tail masking and cache ownership."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import rpa_attention
from vllm_torchtpu.kernels.sparse_gqa_decode import (
    is_supported_by_fused_decode,
    qsa_decode_attention,
)


def _case(batch, context, page, fp8):
    rng = np.random.default_rng(6317 + batch + page)
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    packing = 4 if fp8 else 2
    columns = max(1, math.ceil(max(context, 1) / page))
    table = rng.permutation(batch * columns).reshape(batch, columns).astype(np.int32)
    lengths = np.maximum(context - np.arange(batch) % 4, 0).astype(np.int32)
    if batch > 1:
        lengths[batch // 2] = 0
    blocks = np.full((batch, 512), -1, np.int32)
    for row, length in enumerate(lengths):
        n = min(length // 4, 512)
        blocks[row, :n] = rng.choice(length // 4, n, replace=False)
    q = jnp.asarray(rng.normal(0, 0.5, (batch, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 3, q.shape), jnp.bfloat16)
    q = jnp.where(jnp.asarray(lengths)[:, None, None] > 0, q, jnp.nan)
    gate = jnp.where(jnp.asarray(lengths)[:, None, None] > 0, gate, jnp.nan)
    cache = jnp.asarray(
        rng.normal(0, 0.4, (batch * columns, page, 4 // packing, packing, 256)), dtype
    )
    for row, length in enumerate(lengths):
        # All cache values beyond the actual context are poisoned, including
        # tokens in the final partially selected compression group.
        for logical in range(columns):
            start = max(0, int(length) - logical * page)
            if start < page:
                cache = cache.at[int(table[row, logical]), start:].set(jnp.nan)
    return q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), cache


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "batch,context,page",
    [
        (1, 1, 128),
        (3, 4, 128),
        (3, 8193, 1664),
        (3, 2051, 128),
        (65, 2051, 128),
        (3, 2051, 1664),
        (65, 2051, 1664),
        (3, 0, 1664),
    ],
)
def test_qsa_decode_entry_matches_rpa_and_reference_without_cache_writes(
    batch, context, page, fp8
):
    if pltpu.get_tpu_info().chip_version != pltpu.ChipVersion.TPU_7X:
        pytest.skip("Measured fused decode schedule requires TPU7x")
    args = _case(batch, context, page, fp8)
    ks, vs = (0.375, 2.5) if fp8 else (1.0, 1.0)
    opts = dict(kv_heads=2, ratio=4, k_scale=ks, v_scale=vs)
    host = tuple(np.asarray(x).copy() for x in args)
    reference, _ = jax.jit(functools.partial(rpa_attention, **opts))(*args)
    compiled = (
        jax.jit(
            functools.partial(qsa_decode_attention, **opts),
            donate_argnums=(5,),
        )
        .lower(*args)
        .compile()
    )
    assert "fused_qsa_decode" in compiled.as_text()
    out, cache = compiled(*args)
    actual, expected = np.asarray(out, np.float32), np.asarray(reference, np.float32)
    assert np.isfinite(actual).all()
    # Keep the original numerical contract. Bitwise identity is not required
    # between distinct compiled programs at very small output magnitudes.
    print(
        "NUMERIC",
        batch,
        context,
        page,
        fp8,
        "max_abs",
        float(np.max(np.abs(actual - expected))),
        "relative_l2",
        float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)),
        flush=True,
    )
    np.testing.assert_array_equal(actual[host[3] == 0], 0)
    assert (
        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12) < 0.01
    )
    np.testing.assert_array_equal(
        np.asarray(cache).view(np.uint8), host[5].view(np.uint8)
    )
    live = np.flatnonzero(host[3] > 0)
    for row in live[
        np.linspace(0, max(0, len(live) - 1), min(3, len(live)), dtype=int)
    ]:
        length = int(host[3][row])
        selected = (
            host[2][row, : min(length // 4, 512), None] * 4 + np.arange(4)
        ).reshape(-1)
        selected = np.r_[selected, np.arange(length // 4 * 4, length)]
        values = (
            host[5][host[4][row, selected // page], selected % page]
            .reshape(-1, 4, 256)
            .astype(np.float64)
        )
        wanted = np.empty((24, 256), np.float64)
        for head in range(2):
            logits = (
                host[0][row, head * 12 : (head + 1) * 12].astype(np.float64)
                @ values[:, head * 2].T
                * (ks / 16)
            )
            mass = np.exp(logits - logits.max(axis=1, keepdims=True))
            wanted[head * 12 : (head + 1) * 12] = (
                (mass / mass.sum(axis=1, keepdims=True)) @ values[:, head * 2 + 1] * vs
            )
        wanted *= 1 / (1 + np.exp(-host[1][row].astype(np.float64)))
        assert (
            np.linalg.norm(actual[row] - wanted) / max(np.linalg.norm(wanted), 1e-12)
            < 0.01
        )


def _specs(fp8=False):
    packing = 4 if fp8 else 2
    return [
        jax.ShapeDtypeStruct((8, 24, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((8, 24, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((8, 512), jnp.int32),
        jax.ShapeDtypeStruct((8,), jnp.int32),
        jax.ShapeDtypeStruct((8, 158), jnp.int32),
        jax.ShapeDtypeStruct(
            (158, 1664, 4 // packing, packing, 256),
            jnp.float8_e4m3fn if fp8 else jnp.bfloat16,
        ),
    ]


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "supported",
        "heads",
        "page",
        "budget",
        "batch",
        "dtype",
        "table",
        "scale",
        "hardware",
    ],
)
def test_decode_eligibility_keeps_unvalidated_shapes_on_rpa(monkeypatch, fp8, case):
    chip = pltpu.ChipVersion.TPU_V5P if case == "hardware" else pltpu.ChipVersion.TPU_7X
    monkeypatch.setattr(
        pltpu, "get_tpu_info", lambda: pltpu.get_tpu_info_for_chip(chip, 1)
    )
    args = _specs(fp8)
    changes = {
        "heads": (0, (8, 12, 256)),
        "page": (5, (158, 256, *args[5].shape[2:])),
        "budget": (2, (8, 256)),
        "batch": (0, (257, 24, 256)),
        "table": (4, (8, 159)),
    }
    if case in changes:
        i, shape = changes[case]
        args[i] = jax.ShapeDtypeStruct(shape, args[i].dtype)
    if case == "dtype":
        args[0] = jax.ShapeDtypeStruct(args[0].shape, jnp.float32)
    scale = jax.ShapeDtypeStruct((), jnp.float32) if case == "scale" else 1.0
    assert is_supported_by_fused_decode(
        *args, kv_heads=2, ratio=4, k_scale=scale, v_scale=1.0
    ) == (case == "supported")
    if case != "supported":
        with pytest.raises(NotImplementedError, match="measured TPU7x"):
            qsa_decode_attention(*args, kv_heads=2, k_scale=scale)
