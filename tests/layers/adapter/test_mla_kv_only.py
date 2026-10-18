# SPDX-License-Identifier: Apache-2.0
"""The KV-only pass (spec_decode/mtp_kv_only.py) mirrors the hot-path code it
does not share: every latent it writes must be bit-identical to what the full
MTP forward hands the attention / indexer writers.

* `mla_kv_latents` vs `VllmTPUMultiHeadLatentAttentionWrapper.forward`'s
  `(kv_c_normed, k_pe)` (both rope layouts, bf16 and fp8 projections, both
  rope classes);
* `indexer_k_packed` vs `VllmTPUIndexer.forward` + the packing in
  `VllmTPUSparseAttnIndexer._forward_impl`;
* `mtp_layer_input` vs the upstream `DeepSeekMultiTokenPredictor` ->
  `DeepSeekMultiTokenPredictorLayer` -> `DeepseekV2DecoderLayer` prefix;
* the non-DCP XLA indexer-K append, read back slot by slot.

Runs on CPU; the projections are stand-ins in the processed runtime layouts.
"""

import contextlib
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import vllm.model_executor.custom_op as custom_op_mod
import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.models.deepseek_v2 as deepseek_v2
import vllm.model_executor.parameter as parameter_mod
from vllm.config import CacheConfig
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache, Indexer

from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.spec_decode import mtp_kv_only

ROPE = 64
NOPE = 192
KV_LORA = 512
Q_LORA = 128
V_DIM = 256
HIDDEN = 256
QK = NOPE + ROPE
MAX_POS = 4096
FP8_BLOCK = 128
IDX_HEAD_DIM = 128
IDX_N_HEAD = 8


def _rope(is_neox_style=False, scaling=False):
    cfg = CompilationConfig(mode=CompilationMode.NONE, custom_ops=["none"])
    with patch.object(custom_op_mod, "get_cached_compilation_config", lambda: cfg):
        if scaling:
            return mla_op.VllmDeepseekScalingRotaryEmbedding(
                head_size=ROPE,
                rotary_dim=ROPE,
                max_position_embeddings=MAX_POS,
                base=10000.0,
                is_neox_style=is_neox_style,
                scaling_factor=4.0,
                dtype=torch.bfloat16,
            )
        return RotaryEmbedding(
            head_size=ROPE,
            rotary_dim=ROPE,
            max_position_embeddings=MAX_POS,
            base=10000.0,
            is_neox_style=is_neox_style,
            dtype=torch.bfloat16,
        )


class _Linear(torch.nn.Module):
    """A dense linear in a processed runtime layout (bf16 flipped, or FP8
    flipped with a blockwise scale), as in the rope-permute tests."""

    def __init__(self, n_in, n_out, layout, gen):
        super().__init__()
        w = torch.randn(n_in, n_out, generator=gen) / math.sqrt(n_in)
        self.layout = layout
        if layout == "fp8":
            nb = n_in // FP8_BLOCK
            blocks = w.view(nb, FP8_BLOCK, n_out)
            scale = blocks.abs().amax(1) / 448.0
            q = (blocks / scale[:, None, :]).to(torch.float8_e4m3fn)
            self.weight = torch.nn.Parameter(q.view(n_in, n_out), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(
                scale[None, :, None, :].contiguous(), requires_grad=False
            )
        else:
            self.weight = torch.nn.Parameter(w.bfloat16(), requires_grad=False)
        setattr(self, WEIGHT_FLIPPED_ATTR, True)

    def dense(self):
        if self.layout == "fp8":
            s = self.weight_scale[0, :, 0, :].repeat_interleave(FP8_BLOCK, 0)
            return self.weight.float() * s
        return self.weight.float()

    def forward(self, x):
        return (x.float() @ self.dense()).to(torch.bfloat16), None


class _RmsNorm(torch.nn.Module):
    def __init__(self, n, gen):
        super().__init__()
        self.weight = torch.nn.Parameter(
            (1.0 + 0.2 * torch.randn(n, generator=gen)).bfloat16(), requires_grad=False
        )

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (xf * self.weight.float()).to(x.dtype)


def _wrapper(*, layout="flipped", seed=0, rope=None, gate_is_fused=False):
    """A real wrapper with stand-in projections, built as the rope-permute
    tests build it (bypassing the attention constructor)."""
    gen = torch.Generator().manual_seed(seed)
    heads = 4
    cls = mla_op.VllmTPUMultiHeadLatentAttentionWrapper
    w = cls.__new__(cls)
    torch.nn.Module.__init__(w)
    w.kimi_preprocess = None
    w.q_lora_rank = Q_LORA
    w.kv_lora_rank = KV_LORA
    w.qk_nope_head_dim = NOPE
    w.qk_rope_head_dim = ROPE
    w.qk_head_dim = QK
    w.num_heads = heads
    w.v_head_dim = V_DIM
    w.dcp_q_replicate = False
    w.dcp_group_size = 1
    gate = heads * V_DIM if gate_is_fused else 0
    w.fused_qkv_a_proj = _Linear(HIDDEN, Q_LORA + KV_LORA + ROPE + gate, layout, gen)
    w.q_b_proj = _Linear(Q_LORA, heads * QK, layout, gen)
    w.kv_a_proj_with_mqa = None
    w.q_proj = None
    w.q_a_layernorm = _RmsNorm(Q_LORA, gen)
    w.kv_a_layernorm = _RmsNorm(KV_LORA, gen)
    w.rotary_emb = rope if rope is not None else _rope()
    w.is_sparse = True
    w.indexer = None
    w.skip_topk = True
    w.topk_indices_buffer = torch.zeros((4096, 8), dtype=torch.int32)
    w.g_proj = None
    w.gate_is_fused = gate_is_fused
    w._rope_neox_permuted = False
    w.o_proj = lambda value: (value, None)
    return w


def _wrapper_latents(w, hidden, positions):
    """What `forward` hands the MLA attention as (kv_c_normed, k_pe)."""
    w.mla_attn = MagicMock(
        return_value=torch.zeros((hidden.shape[0], w.num_heads * V_DIM))
    )
    mla_op.VllmTPUMultiHeadLatentAttentionWrapper.forward(
        w, positions=positions, hidden_states=hidden
    )
    _, kv_c, k_pe = w.mla_attn.call_args.args[:3]
    return kv_c, k_pe


def _inputs(num_tokens, seed=1):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(num_tokens, HIDDEN, generator=gen).bfloat16()
    positions = torch.randperm(MAX_POS, generator=gen)[:num_tokens]
    return hidden, positions


@pytest.mark.parametrize("layout", ["flipped", "fp8"])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("scaling", [False, True], ids=["rope", "deepseek_scaling"])
@pytest.mark.parametrize("num_tokens", [16, 96])
def test_mla_kv_latents_match_the_wrapper(layout, permuted, scaling, num_tokens):
    w = _wrapper(layout=layout, rope=_rope(scaling=scaling))
    if permuted:
        w.permute_rope_to_half_split()
        assert w._rope_neox_permuted
    hidden, positions = _inputs(num_tokens)
    want_kv, want_pe = _wrapper_latents(w, hidden, positions)
    got_kv, got_pe = mtp_kv_only.mla_kv_latents(w, positions, hidden)
    assert torch.equal(got_kv, want_kv)
    assert got_pe.shape == want_pe.shape
    assert torch.equal(got_pe, want_pe)


def test_mla_kv_latents_with_fused_gate():
    w = _wrapper(gate_is_fused=True)
    hidden, positions = _inputs(16)
    want_kv, want_pe = _wrapper_latents(w, hidden, positions)
    got_kv, got_pe = mtp_kv_only.mla_kv_latents(w, positions, hidden)
    assert torch.equal(got_kv, want_kv)
    assert torch.equal(got_pe, want_pe)


def test_mla_kv_latents_positive_control():
    """Different positions must change k_pe: the comparison above can fail."""
    w = _wrapper()
    hidden, positions = _inputs(16)
    want_kv, want_pe = _wrapper_latents(w, hidden, positions)
    _, got_pe = mtp_kv_only.mla_kv_latents(w, positions + 1, hidden)
    assert not torch.equal(got_pe, want_pe)


# ---------------------------------------------------------------------------
# Indexer K
# ---------------------------------------------------------------------------


class _StubIndexerCache(DeepseekV32IndexerCache):
    def __init__(self, head_dim, dtype, prefix, cache_config):
        torch.nn.Module.__init__(self)
        self.head_dim = head_dim
        self.dtype = dtype
        self.prefix = prefix
        self.cache_config = cache_config


@contextlib.contextmanager
def _default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _indexer(seed=0):
    config = SimpleNamespace(
        index_topk=8,
        index_n_heads=IDX_N_HEAD,
        index_head_dim=IDX_HEAD_DIM,
        qk_rope_head_dim=ROPE,
    )
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=MAX_POS))
    with (
        patch.object(deepseek_v2, "DeepseekV32IndexerCache", _StubIndexerCache),
        patch.object(deepseek_v2, "SparseAttnIndexer", MagicMock()),
        patch.object(linear_mod, "get_tensor_model_parallel_rank", lambda: 0),
        patch.object(linear_mod, "get_tensor_model_parallel_world_size", lambda: 1),
        patch.object(parameter_mod, "get_tensor_model_parallel_rank", lambda: 0),
        patch.object(parameter_mod, "get_tensor_model_parallel_world_size", lambda: 1),
        _default_dtype(torch.bfloat16),
    ):
        ind = Indexer(
            vllm_config=vllm_config,
            config=config,
            hidden_size=HIDDEN,
            q_lora_rank=Q_LORA,
            quant_config=None,
            cache_config=CacheConfig(),
            topk_indices_buffer=torch.full((16, 8), -1, dtype=torch.int32),
            prefix="model.layers.78.self_attn.indexer",
        )
        ind = mla_op.VllmTPUIndexer.rebind(ind)
    gen = torch.Generator().manual_seed(seed)
    ind.wq_b = _Linear(Q_LORA, IDX_N_HEAD * IDX_HEAD_DIM, "flipped", gen)
    ind.wk_weights_proj = _Linear(HIDDEN, IDX_HEAD_DIM + IDX_N_HEAD, "flipped", gen)
    with torch.no_grad():
        ind.k_norm.weight.copy_(1.0 + 0.3 * torch.randn(IDX_HEAD_DIM, generator=gen))
        ind.k_norm.bias.copy_(0.3 * torch.randn(IDX_HEAD_DIM, generator=gen))
    ind.indexer_op = MagicMock(side_effect=lambda *a: a)
    return ind


class _Captured(Exception):
    def __init__(self, args):
        super().__init__()
        self.args_ = args


def _packed_by_the_indexer_op(hidden, q_fp8, k, weights):
    """The packed rows `_forward_impl` hands the top-k op (and its writer)."""

    def capture(*args):
        raise _Captured(args)

    op = SimpleNamespace(
        k_cache=SimpleNamespace(prefix="idx", kv_cache=torch.zeros(1)),
        topk_op_dcp=None,
        topk_op=capture,
    )
    md = AttentionMetadata(
        input_positions=torch.zeros(1),
        block_tables=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.zeros(1, dtype=torch.int32),
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        request_distribution=torch.zeros(3, dtype=torch.int32),
    )
    with (
        patch.object(
            mla_op,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"idx": md}),
        ),
        pytest.raises(_Captured) as exc,
    ):
        mla_op.VllmTPUSparseAttnIndexer._forward_impl(
            op, hidden, q_fp8, k, weights, None
        )
    return exc.value.args_[3]


@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("num_tokens", [16, 96])
def test_indexer_k_packed_matches_the_indexer(permuted, num_tokens):
    ind = _indexer()
    rope = _rope()
    if permuted:
        ind.permute_rope_to_half_split()
    hidden, positions = _inputs(num_tokens)
    gen = torch.Generator().manual_seed(5)
    qr = torch.randn(num_tokens, Q_LORA, generator=gen).bfloat16()
    with torch.no_grad():
        h, q_fp8, k, weights = ind(hidden, qr, positions, rope)
        want = _packed_by_the_indexer_op(h, q_fp8, k, weights)
        got = mtp_kv_only.indexer_k_packed(ind, hidden, positions, rope)
    assert got.dtype == want.dtype == torch.uint8
    assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# The MTP layer prefix
# ---------------------------------------------------------------------------


class _Stop(Exception):
    pass


def _upstream_mtp(vocab=64, hidden=32, seed=0):
    from vllm.model_executor.models.deepseek_mtp import (
        DeepSeekMultiTokenPredictor,
        DeepSeekMultiTokenPredictorLayer,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer

    gen = torch.Generator().manual_seed(seed)

    class _Capture(torch.nn.Module):
        def forward(self, positions, hidden_states, *_args):
            self.seen = hidden_states
            raise _Stop

    class _Block(torch.nn.Module):
        forward = DeepseekV2DecoderLayer.forward

    class _Layer(torch.nn.Module):
        forward = DeepSeekMultiTokenPredictorLayer.forward

    class _Predictor(torch.nn.Module):
        forward = DeepSeekMultiTokenPredictor.forward

    block = _Block()
    block.use_sequence_parallel_moe = False
    block.use_mha = False
    block.input_layernorm = _RmsNorm(hidden, gen)
    block.self_attn = _Capture()
    layer = _Layer()
    layer.enorm = _RmsNorm(hidden, gen)
    layer.hnorm = _RmsNorm(hidden, gen)
    layer.eh_proj = torch.nn.Linear(2 * hidden, hidden, bias=False).bfloat16()
    torch.nn.init.normal_(layer.eh_proj.weight, generator=gen)
    layer.mtp_block = block
    predictor = _Predictor()
    predictor.layers = torch.nn.ModuleDict({"78": layer})
    predictor.mtp_start_layer_idx = 78
    predictor.num_mtp_layers = 1
    predictor.embed_tokens = torch.nn.Embedding(vocab, hidden).bfloat16()
    return SimpleNamespace(model=predictor), block.self_attn


def test_mtp_layer_input_matches_the_upstream_prefix():
    mtp, capture = _upstream_mtp()
    gen = torch.Generator().manual_seed(3)
    input_ids = torch.randint(0, 64, (12,), generator=gen)
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 7, 8, 9, 10, 0, 5])
    hidden = torch.randn(12, 32, generator=gen).bfloat16()
    with torch.no_grad(), pytest.raises(_Stop):
        mtp.model(input_ids, positions, hidden, None, 0)
    with torch.no_grad():
        layer, got = mtp_kv_only.mtp_layer_input(mtp, input_ids, positions, hidden)
    assert layer is mtp.model.layers["78"]
    assert torch.equal(got, capture.seen)


# ---------------------------------------------------------------------------
# Support check
# ---------------------------------------------------------------------------


class _SelfAttn(torch.nn.Module):
    def __init__(self, mla_attn):
        super().__init__()
        self.mla_attn = mla_attn


def _supported_mtp():
    w = _wrapper()
    w.indexer = _indexer()
    w.indexer.indexer_op = MagicMock()
    mtp, _ = _upstream_mtp()
    mtp.model.layers["78"].mtp_block.self_attn = _SelfAttn(w)
    return mtp


def test_kv_only_supported_for_a_dsa_mtp():
    assert mtp_kv_only.kv_only_unsupported_reason(_supported_mtp()) is None


@pytest.mark.parametrize(
    "breakage", ["none", "two_layers", "plain_wrapper", "dense", "kimi"]
)
def test_kv_only_unsupported_reasons(breakage):
    mtp = None if breakage == "none" else _supported_mtp()
    if breakage == "two_layers":
        mtp.model.layers["79"] = mtp.model.layers["78"]
    elif breakage == "plain_wrapper":
        mtp.model.layers["78"].mtp_block.self_attn = _SelfAttn(torch.nn.Identity())
    elif breakage == "dense":
        mtp.model.layers["78"].mtp_block.self_attn.mla_attn.is_sparse = False
    elif breakage == "kimi":
        mtp.model.layers["78"].mtp_block.self_attn.mla_attn.kimi_preprocess = object()
    assert mtp_kv_only.kv_only_unsupported_reason(mtp) is not None


# ---------------------------------------------------------------------------
# Non-DCP indexer-K append (XLA writer)
# ---------------------------------------------------------------------------


def test_indexer_k_append_local_xla_writes_every_row_in_place():
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        KVLayout,
        get_kv_cache_shape,
    )

    page_size, num_pages, width = 16, 8, IDX_HEAD_DIM + 1
    shape = get_kv_cache_shape(num_pages, page_size, width, KVLayout.HEAD_ALONG_SUBLANE)
    cache = jnp.zeros(shape, dtype=jnp.uint8)
    # Two requests: 3 new tokens at positions 5..7, 4 new tokens at 14..17
    # (crossing a page); one padded row.
    seq_lens = jnp.array([8, 18], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 3, 7], dtype=jnp.int32)
    pages = jnp.array([[2, 6], [3, 5]], dtype=jnp.int32).reshape(-1)
    rng = np.random.default_rng(0)
    k_packed = jnp.asarray(rng.integers(1, 255, (8, width), dtype=np.uint8))
    out = np.asarray(
        mtp_kv_only._indexer_k_append_local(
            cache, k_packed, seq_lens, pages, cu_q_lens, "xla"
        )
    )
    kv_packing = shape[2]
    expected = np.zeros_like(out)
    rows = [(0, p) for p in range(5, 8)] + [(1, p) for p in range(14, 18)]
    page_table = np.asarray(pages).reshape(2, -1)
    for t, (r, pos) in enumerate(rows):
        page = page_table[r, pos // page_size]
        slot = pos % page_size
        expected[page, slot // kv_packing, slot % kv_packing, :width] = np.asarray(
            k_packed[t]
        )
    np.testing.assert_array_equal(out, expected)
