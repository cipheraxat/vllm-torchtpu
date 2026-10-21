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
"""Mocked static resource boundaries; these are not cross-chip hardware tests."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_resources import (
    HcHardware,
    full_mix_memory,
    supports_full_mix,
    supports_residual,
)
from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import build_full_mix_op


def hardware(capacity=64 << 20):
    return HcHardware("mock-layout", capacity, 128, 16, 8, 256)


@pytest.mark.parametrize("capacity,expected", [(16, False), (64, True), (128, True)])
def test_full_residency_and_capacity(capacity, expected):
    info = hardware(capacity << 20)
    # 320 lowrank rows, but up-weight minor320 pads to384 vector columns.
    estimate = full_mix_memory(4, 2560, 320, 256, info)
    assert estimate.resident_bytes == 10240 * 2 + (320 + 384) * 10240 * 2
    assert estimate.buffer_bytes == (4 * 256 * 10240 + 2 * 256 * 2560) * 2
    assert supports_full_mix(4, 2560, 320, 256, info) is expected
    # 512-row rings plus the same working policy exceed64MiB, without an OOM.
    assert not supports_full_mix(4, 2560, 320, 512, hardware())


def test_exact_budget_boundary_and_cache_identity():
    info = hardware()
    needed = full_mix_memory(4, 2560, 320, 256, info).total_bytes
    assert not supports_full_mix(
        4, 2560, 320, 256, replace(info, vmem_bytes=needed - 1)
    )
    assert supports_full_mix(4, 2560, 320, 256, replace(info, vmem_bytes=needed))
    a = build_full_mix_op(4, 128, 1e-6, 128, info)
    assert build_full_mix_op(4, 128, 1e-6, 128, info) is a
    assert (
        build_full_mix_op(4, 128, 1e-6, 128, replace(info, vmem_bytes=128 << 20))
        is not a
    )
    assert build_full_mix_op(4, 128, 1e-6, 128, replace(info, chip="other")) is not a


@pytest.mark.parametrize(
    "branches,hidden,lowrank,tile",
    [
        (0, 128, 128, 128),
        (4, 0, 128, 128),
        (4, 129, 128, 128),
        (4, 128, 0, 128),
        (4, 128, 128, 0),
        (4, 128, 128, 64),
        (2**31, 128, 128, 128),
    ],
)
def test_static_invalid_or_overflow_geometry(branches, hidden, lowrank, tile):
    assert not supports_full_mix(branches, hidden, lowrank, tile, hardware())


def test_unknown_or_unqualified_layout():
    assert not supports_full_mix(4, 128, 128, 128, None)
    for info in [
        replace(hardware(), vmem_bytes=0),
        replace(hardware(), lanes=64),
        replace(hardware(), bf16_rows=0),
        replace(hardware(), mxu_columns=0),
    ]:
        assert not supports_full_mix(4, 128, 128, 128, info)


@pytest.mark.parametrize("tile", [16, 32, 64, 128])
def test_residual_alignment_is_dtype_derived(tile):
    assert supports_residual(3, 128, tile, hardware())
    assert supports_residual(4, 2560, tile, hardware())
    different_layout = replace(hardware(), bf16_rows=32, fp32_rows=16)
    assert supports_residual(3, 128, tile, different_layout) is (tile % 32 == 0)
    assert not supports_residual(3, 128, tile + 1, hardware())
    assert not supports_residual(3, 128, tile, None)


def test_direct_resource_and_alignment_failure_before_lowering():
    info = hardware()
    shape = jax.ShapeDtypeStruct
    args = [
        shape(s, jnp.bfloat16)
        for s in [(512, 10240), (10240,), (320, 10240), (10240, 320)]
    ]
    with pytest.raises(ValueError, match="VMEM budget"):
        jax.eval_shape(
            lambda *a: _full_mix(
                *a, branches=4, hidden=2560, eps=1e-6, token_tile=512, hardware=info
            ),
            *args,
        )
    fn = gated_residual_fn(3, 128, 8, info)
    with pytest.raises(ValueError, match="packing"):
        jax.eval_shape(
            fn,
            shape((8, 384), jnp.bfloat16),
            shape((8, 128), jnp.bfloat16),
            shape((8, 3), jnp.float32),
        )


def test_fake_registration_never_queries_hardware(monkeypatch):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as adapter

    def forbidden():
        raise AssertionError("Fake registration must not query/initialize a TPU")

    monkeypatch.setattr(adapter.pltpu, "get_tpu_info", forbidden)
    from types import SimpleNamespace

    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(is_initialized=lambda: False), raising=False
    )
    assert adapter.current_hc_hardware() is None
    full = adapter.build_full_mix_op(3, 128, 1e-6, 128)
    residual = adapter.build_gated_residual_op(3, 128, 16)
    with FakeTensorMode():
        x = torch.empty(128, 384, dtype=torch.bfloat16)
        norm = torch.empty(384, dtype=torch.bfloat16)
        down = torch.empty(128, 384, dtype=torch.bfloat16)
        up = torch.empty(384, 128, dtype=torch.bfloat16)
        assert full(x, norm, down, up)[0].shape == (128, 128)
        assert (
            residual(
                x,
                torch.empty(128, 128, dtype=torch.bfloat16),
                torch.empty(128, 3, dtype=torch.float32),
            ).shape
            == x.shape
        )
