# SPDX-License-Identifier: Apache-2.0
"""Pass 1 of a two-pass MTP draft step 0: write the caches, nothing else.

Draft step 0 runs the MTP layer on every row of the target's chunk, but a draft
token is read from one row per request only (the end of its accepted prefix,
`last_token_indices`). Every other row still has to leave its MTP KV entry and
its indexer-K entry behind, because later rows and later steps attend to them,
yet nothing reads its attention output, MoE output or top-k. With
`TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS` the proposer splits a large step 0 in two:

* pass 1 (`MtpKvOnlyModel`, this module) runs only what produces the two cache
  entries -- embedding, enorm/hnorm, eh_proj, input norm, then kv_a + kv norm +
  RoPE into the MLA KV cache and wk + k norm + RoPE into the indexer-K cache --
  for every row, with the target's own step-0 metadata;
* pass 2 is the draft's ordinary step-0 program on the per-request rows alone,
  reading the caches pass 1 just wrote (`Eagle3Proposer._two_pass_step0`).

Pass 1 calls the same cache writers the attention and indexer ops call
(`update_sparse_mla_kv_cache[_dcp]`, `_cp_scatter_indexer_k` or the non-DCP
indexer append), with the same owner masks and slots, so every entry it writes
is the entry the full pass writes. The latents themselves are computed by
code that mirrors `DeepSeekMultiTokenPredictorLayer.forward` up to the decoder
layer's input norm, `VllmTPUMultiHeadLatentAttentionWrapper.forward` up to the
k_pe RoPE and `VllmTPUIndexer.forward` up to k. That glue is duplicated rather
than factored out of those hot files; `tests/spec_decode/test_mtp_kv_only.py`
pins it against them.

This lives outside eagle3.py because that module uses postponed annotations,
which ``support_torch_compile`` cannot read.
"""

from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context

# Same dims as the draft's own compile (`MTP_DYNAMIC_ARG_DIMS`): dim 0, the
# token dimension, of every tensor argument.
MTP_KV_ONLY_DYNAMIC_ARG_DIMS: dict[str, int | list[int] | dict[int, str]] = {
    "input_ids": 0,
    "positions": 0,
    "hidden_states": 0,
}


def _single_mtp_layer(mtp: nn.Module) -> nn.Module | None:
    predictor = getattr(mtp, "model", None)
    layers: Any = getattr(predictor, "layers", None)
    if not isinstance(layers, nn.ModuleDict):
        return None
    if len(layers) != 1:
        return None
    start = getattr(predictor, "mtp_start_layer_idx", None)
    if start is None or str(start) not in layers:
        return None
    return layers[str(start)]


def kv_only_unsupported_reason(mtp: nn.Module | None) -> str | None:
    """Why pass 1 cannot serve this draft, or None when it can.

    Pass 1 mirrors one specific forward: a single-layer `DeepSeekMTP` whose
    decoder layer runs the TPU sparse (DSA) MLA wrapper with its own indexer.
    Anything else keeps the single full step-0 pass.
    """
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmTPUIndexer,
        VllmTPUMultiHeadLatentAttentionWrapper,
    )

    if mtp is None:
        return "no draft model"
    layer = _single_mtp_layer(mtp)
    if layer is None:
        return "the draft is not a single-layer DeepSeek-style MTP"
    if getattr(mtp.model, "embed_tokens", None) is None:
        return "the MTP predictor has no embed_tokens"
    for name in ("enorm", "hnorm", "eh_proj", "mtp_block"):
        if getattr(layer, name, None) is None:
            return f"the MTP layer has no {name}"
    block = layer.mtp_block
    if getattr(block, "input_layernorm", None) is None:
        return "the MTP decoder layer has no input_layernorm"
    wrapper = getattr(getattr(block, "self_attn", None), "mla_attn", None)
    if not isinstance(wrapper, VllmTPUMultiHeadLatentAttentionWrapper):
        return "the MTP attention is not the TPU MLA wrapper"
    if not wrapper.is_sparse or not isinstance(wrapper.indexer, VllmTPUIndexer):
        return "the MTP attention is not sparse (DSA) MLA with an indexer"
    if wrapper.kimi_preprocess is not None:
        return "the Kimi fused MLA preprocess is not mirrored"
    if getattr(wrapper.indexer, "indexer_op", None) is None:
        return "the MTP indexer has no indexer op"
    return None


def mla_kv_latents(
    wrapper: nn.Module, positions: torch.Tensor, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(kv_c_normed, k_pe)` exactly as `VllmTPUMultiHeadLatentAttentionWrapper
    .forward` hands them to the MLA attention, without the q path.

    The q columns of `fused_qkv_a_proj` are computed and dropped: the same op
    as the full pass. The rope is applied to k_pe only (it doubles as the
    query argument through a copy, whose result is dropped); a rope's key
    output does not depend on the query.
    """
    if wrapper.q_lora_rank is not None:
        qkv_lora = wrapper.fused_qkv_a_proj(hidden_states)[0]
        if wrapper.gate_is_fused:
            _, kv_lora, _ = qkv_lora.split(
                [
                    wrapper.q_lora_rank,
                    wrapper.kv_lora_rank + wrapper.qk_rope_head_dim,
                    wrapper.num_heads * wrapper.v_head_dim,
                ],
                dim=-1,
            )
        else:
            _, kv_lora = qkv_lora.split(
                [wrapper.q_lora_rank, wrapper.kv_lora_rank + wrapper.qk_rope_head_dim],
                dim=-1,
            )
    else:
        kv_lora = wrapper.kv_a_proj_with_mqa(hidden_states)[0]
    kv_c, k_pe = kv_lora.split([wrapper.kv_lora_rank, wrapper.qk_rope_head_dim], dim=-1)
    kv_c_normed = wrapper.kv_a_layernorm(kv_c)
    k_pe = k_pe.unsqueeze(1)
    if wrapper.rotary_emb is not None:
        from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
            rope_half_split,
        )

        k_pe_dtype = k_pe.dtype
        if getattr(wrapper, "_rope_neox_permuted", False):
            _, k_pe = rope_half_split(wrapper.rotary_emb, positions, k_pe.clone(), k_pe)
        else:
            _, k_pe = wrapper.rotary_emb(positions, k_pe.clone(), k_pe)
        assert k_pe is not None
        k_pe = k_pe.to(dtype=k_pe_dtype)
    return kv_c_normed, k_pe


def indexer_k_packed(
    indexer: nn.Module,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: nn.Module,
) -> torch.Tensor:
    """The packed fp8 + e8m0 indexer K rows `VllmTPUIndexer.forward` and
    `VllmTPUSparseAttnIndexer._forward_impl` write into the indexer cache."""
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        rope_half_split,
        rope_half_split_leading,
        rope_half_split_leading_supported,
    )
    from vllm_torchtpu.layers.core.quantization import quantize_tensor

    kw, _ = indexer.wk_weights_proj(hidden_states)
    k = kw[:, : indexer.head_dim]
    k = indexer.k_norm(k)
    permuted = getattr(indexer, "_rope_neox_permuted", False)
    if permuted and rope_half_split_leading_supported(rotary_emb):
        k = rope_half_split_leading(rotary_emb, positions, k, indexer.rope_dim)
    else:
        k_pe, k_nope = torch.split(
            k, [indexer.rope_dim, indexer.head_dim - indexer.rope_dim], dim=-1
        )
        k_pe = k_pe.unsqueeze(1)
        if permuted:
            _, k_pe = rope_half_split(rotary_emb, positions, k_pe.clone(), k_pe)
        else:
            _, k_pe = rotary_emb(positions, k_pe.clone(), k_pe)
        assert k_pe is not None
        k_pe = k_pe.reshape(-1, 1, indexer.rope_dim)
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)
    k_quant, k_scale = quantize_tensor(k, torch.float8_e4m3fn, axis=-1, use_ue8m0=True)
    return torch.cat([k_quant.view(torch.uint8), k_scale], dim=-1)


def _indexer_k_append_local(
    cache_kv, k_packed, seq_lens, page_indices, cu_q_lens, mode
):
    """The non-DCP indexer K write of `VllmTPUSparseAttnIndexer.
    _build_streamidx_op`, without its top-k."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        coalesced_kv_append,
    )
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        _resolve_indexer_kv_cache_mode,
    )

    writer, kv_layout = _resolve_indexer_kv_cache_mode(cache_kv, k_packed, mode=mode)
    if writer != "xla":
        return coalesced_kv_append(
            cache_kv, k_packed, seq_lens, page_indices, cu_q_lens, kv_layout=kv_layout
        )
    _, block_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
    block_size = block_size_per_kv_packing * kv_packing
    k_packed = jnp.pad(k_packed, ((0, 0), (0, lkv_dim - k_packed.shape[-1])))
    tok = jnp.arange(k_packed.shape[0], dtype=jnp.int32)
    seq_id = jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(jnp.int32)
    q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
    pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
    page = jnp.where(
        tok < cu_q_lens[-1],
        page_indices.reshape(seq_lens.shape[0], -1)[seq_id, pos // block_size],
        jnp.int32(2**30),
    )
    slot = pos % block_size
    return cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k_packed)


class _KvOnlyWriteOp:
    """One op per MTP layer that writes its MLA KV and indexer-K rows.

    Built eagerly at load (Dynamo cannot trace an op's construction). Both
    halves are the writers the full pass uses, run inside the same partition
    specs: under DCP the caches are sharded over the DCP axis and the rows are
    replicated, and each rank keeps only the rows it owns.
    """

    def __init__(self, wrapper: nn.Module, mesh: Any) -> None:
        from torch_tpu._internal import pallas

        from vllm_torchtpu import envs
        from vllm_torchtpu.distributed.dcp import (
            dcp_rank_operand,
            get_or_create_dcp_mesh,
        )
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
            DCP_AXIS_NAME,
            cp_rank_as_data,
        )
        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
            pcp_streaming_jax_op,
        )
        from vllm_torchtpu.kernels.mla.kv_cache_utils import (
            update_sparse_mla_kv_cache,
            update_sparse_mla_kv_cache_dcp,
        )
        from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
            _cp_scatter_indexer_k,
            _resolve_indexer_kv_cache_mode,
        )

        self.wrapper = wrapper
        self.mla_layer = wrapper.mla_attn
        self.indexer = wrapper.indexer
        self.indexer_op = wrapper.indexer.indexer_op
        impl = self.mla_layer.impl
        self.impl = impl
        dcp_size = int(getattr(impl, "dcp_size", 1))
        if int(getattr(self.indexer_op, "dcp_size", 1)) != dcp_size:
            raise ValueError(
                "the MTP MLA and indexer caches disagree on the DCP size: "
                f"{dcp_size} vs {self.indexer_op.dcp_size}"
            )
        self.dcp_size = dcp_size
        mla_layer = self.mla_layer
        kv_cache_mode = (
            envs.VLLM_TPU_INDEXER_KV_CACHE_MODE
            if envs.is_indexer_kv_cache_mode_explicit()
            else None
        )
        name = mla_layer.layer_name.replace(".", "_")

        def _fake(nope, rope, idx_cache, *rest):
            del rest
            return (
                torch.empty_like(nope),
                torch.empty_like(rope),
                torch.empty_like(idx_cache),
            )

        self.rank_tensor = None
        if dcp_size > 1:
            interleave = int(impl.dcp_interleave_size)
            dcp_mesh = get_or_create_dcp_mesh(axis_name=DCP_AXIS_NAME)
            ax = DCP_AXIS_NAME

            def _write_dcp(
                nope,
                rope,
                idx_cache,
                kv_c,
                k_pe,
                k_packed,
                seq_lens,
                block_tables,
                qsl,
                idx_seq_lens,
                idx_pages,
                idx_cu,
                rank_ids,
            ):
                nope_spec, rope_spec = mla_layer.mla_kv_spec

                def _local(nope_, rope_, kv_c_, k_pe_, sl_, bt_, qsl_, *rank_):
                    rank = rank_[0][0] if rank_ else cp_rank_as_data(ax, dcp_size)
                    return update_sparse_mla_kv_cache_dcp(
                        nope_,
                        rope_,
                        kv_c_,
                        k_pe_,
                        sl_,
                        bt_,
                        qsl_,
                        rank,
                        nope_spec=nope_spec,
                        rope_spec=rope_spec,
                        dcp_size=dcp_size,
                        interleave_size=interleave,
                    )

                rank_args = () if rank_ids is None else (rank_ids,)
                in_specs = (
                    P(ax),
                    P(ax),
                    P(None, None),
                    P(None, None),
                    P(None),
                    P(None),
                    P(None),
                ) + (P(ax),) * len(rank_args)
                nope, rope = jax.jit(
                    jax.shard_map(
                        _local,
                        mesh=dcp_mesh,
                        in_specs=in_specs,
                        out_specs=(P(ax), P(ax)),
                        check_vma=False,
                    )
                )(nope, rope, kv_c, k_pe, seq_lens, block_tables, qsl, *rank_args)
                writer, kv_layout = _resolve_indexer_kv_cache_mode(
                    idx_cache, k_packed, mode=kv_cache_mode
                )
                idx_cache = _cp_scatter_indexer_k(
                    idx_cache,
                    k_packed,
                    idx_seq_lens,
                    idx_pages,
                    idx_cu,
                    mesh=dcp_mesh,
                    cp_size=dcp_size,
                    interleave_size=interleave,
                    axis_name=ax,
                    writer=writer,
                    kv_layout=kv_layout,
                    rank_ids=rank_ids,
                )
                return nope, rope, idx_cache

            # Fixed arity: torch custom-op schemas take no varargs.
            def _write_dcp_jax(
                nope: jax.Array,
                rope: jax.Array,
                idx_cache: jax.Array,
                kv_c: jax.Array,
                k_pe: jax.Array,
                k_packed: jax.Array,
                seq_lens: jax.Array,
                block_tables: jax.Array,
                qsl: jax.Array,
                idx_seq_lens: jax.Array,
                idx_pages: jax.Array,
                idx_cu: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _write_dcp(
                    nope,
                    rope,
                    idx_cache,
                    kv_c,
                    k_pe,
                    k_packed,
                    seq_lens,
                    block_tables,
                    qsl,
                    idx_seq_lens,
                    idx_pages,
                    idx_cu,
                    None,
                )

            def _write_dcp_rank_jax(
                nope: jax.Array,
                rope: jax.Array,
                idx_cache: jax.Array,
                kv_c: jax.Array,
                k_pe: jax.Array,
                k_packed: jax.Array,
                seq_lens: jax.Array,
                block_tables: jax.Array,
                qsl: jax.Array,
                idx_seq_lens: jax.Array,
                idx_pages: jax.Array,
                idx_cu: jax.Array,
                rank_ids: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _write_dcp(
                    nope,
                    rope,
                    idx_cache,
                    kv_c,
                    k_pe,
                    k_packed,
                    seq_lens,
                    block_tables,
                    qsl,
                    idx_seq_lens,
                    idx_pages,
                    idx_cu,
                    rank_ids,
                )

            if envs.TPU_DCP_RANK_OPERAND:
                self.rank_tensor = dcp_rank_operand()
            op_name = f"pallas::mtp_kv_only_write_dcp_{name}"
            fn = _write_dcp_jax
            if self.rank_tensor is not None:
                op_name += "_rank"
                fn = _write_dcp_rank_jax
            self.jax_op = pcp_streaming_jax_op(
                op_name,
                fn,
                donate_argnums=(0, 1, 2),
                mesh=dcp_mesh,
                input_partition_specs=(P(ax), P(ax), P(ax))
                + (P(),) * 9
                + ((P(ax),) if self.rank_tensor is not None else ()),
                output_partition_specs=(P(ax), P(ax), P(ax)),
            )
        else:

            def _write_jax(
                nope: jax.Array,
                rope: jax.Array,
                idx_cache: jax.Array,
                kv_c: jax.Array,
                k_pe: jax.Array,
                k_packed: jax.Array,
                seq_lens: jax.Array,
                block_tables: jax.Array,
                qsl: jax.Array,
                idx_seq_lens: jax.Array,
                idx_pages: jax.Array,
                idx_cu: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                nope_spec, rope_spec = mla_layer.mla_kv_spec

                def _local(nope_, rope_, kv_c_, k_pe_, sl_, bt_, qsl_):
                    return update_sparse_mla_kv_cache(
                        nope_,
                        rope_,
                        kv_c_,
                        k_pe_,
                        sl_,
                        bt_,
                        qsl_,
                        nope_spec=nope_spec,
                        rope_spec=rope_spec,
                    )

                nope, rope = jax.jit(
                    jax.shard_map(
                        _local,
                        mesh=mesh,
                        in_specs=(
                            P(None),
                            P(None),
                            P(None, None),
                            P(None, None),
                            P(None),
                            P(None),
                            P(None),
                        ),
                        out_specs=(P(None), P(None)),
                        check_vma=False,
                    )
                )(nope, rope, kv_c, k_pe, seq_lens, block_tables, qsl)
                idx_cache = _indexer_k_append_local(
                    idx_cache,
                    k_packed,
                    idx_seq_lens,
                    idx_pages,
                    idx_cu,
                    kv_cache_mode,
                )
                return nope, rope, idx_cache

            self.jax_op = pallas.jax_op(
                f"pallas::mtp_kv_only_write_{name}",
                _write_jax,
                donate_argnums=(0, 1, 2),
            )
        self.jax_op.register_fake(_fake)

    def __call__(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        """Write this step's rows; returns k_pe (the program needs an output)."""
        from vllm.model_executor.layers.attention.attention import (
            get_attention_context,
        )

        from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
        from vllm_torchtpu.layers.core.quantization import quantize_kv

        wrapper, layer = self.wrapper, self.mla_layer
        kv_c_normed, k_pe = mla_kv_latents(wrapper, positions, hidden_states)
        k_packed = indexer_k_packed(
            self.indexer, hidden_states, positions, wrapper.indexer_rope_emb
        )

        md, _, kv_cache, _ = get_attention_context(layer.layer_name)
        fwd_md = get_forward_context().attn_metadata
        idx_md = (
            fwd_md.get(self.indexer_op.k_cache.prefix)
            if isinstance(fwd_md, dict)
            else None
        )
        idx_cache = self.indexer_op.k_cache.kv_cache
        # The memory probe runs before the caches are bound; write nothing.
        if (
            not isinstance(kv_cache, (tuple, list))
            or not isinstance(md, AttentionMetadata)
            or not isinstance(idx_md, AttentionMetadata)
            or idx_cache.numel() == 0
        ):
            return k_pe

        _, k_scale, _ = self.impl._get_kv_scales(layer)
        if layer.kv_cache_quantized_dtype is not None:
            assert k_scale is not None
            kv_c_normed, _ = quantize_kv(
                layer.kv_cache_quantized_dtype, kv_c_normed, value=None, k_scale=k_scale
            )
            k_pe_q, _ = quantize_kv(
                layer.kv_cache_quantized_dtype, k_pe, value=None, k_scale=k_scale
            )
        else:
            k_pe_q = k_pe
        nope, rope = kv_cache
        rank = () if self.rank_tensor is None else (self.rank_tensor,)
        new_nope, new_rope, new_idx = self.jax_op(
            nope,
            rope,
            idx_cache,
            kv_c_normed.view(-1, layer.kv_lora_rank),
            k_pe_q.view(-1, layer.qk_rope_head_dim),
            k_packed,
            md.seq_lens,
            md.block_tables,
            md.query_start_loc,
            idx_md.seq_lens,
            idx_md.block_tables,
            idx_md.query_start_loc,
            *rank,
        )
        nope.copy_(new_nope)
        rope.copy_(new_rope)
        idx_cache.copy_(new_idx)
        return k_pe


def mtp_layer_input(
    mtp: nn.Module,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> tuple[nn.Module, torch.Tensor]:
    """The MTP decoder layer and what its attention receives: the prefix of
    `DeepSeekMultiTokenPredictor.forward` -> `DeepSeekMultiTokenPredictorLayer
    .forward` -> `DeepseekV2DecoderLayer.forward` (residual None) up to the
    `self_attn` call, at draft step 0."""
    layer = _single_mtp_layer(mtp)
    assert layer is not None
    inputs_embeds = mtp.model.embed_tokens(input_ids)
    # masking inputs at position 0, as not needed by MTP
    inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    inputs_embeds = layer.enorm(inputs_embeds)
    previous_hidden_states = layer.hnorm(hidden_states)
    hidden = layer.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))
    return layer, layer.mtp_block.input_layernorm(hidden)


@support_torch_compile(dynamic_arg_dims=MTP_KV_ONLY_DYNAMIC_ARG_DIMS)
class MtpKvOnlyModel(nn.Module):
    """Pass 1: writes the MTP layer's KV and indexer-K rows for every token."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mtp: nn.Module,
        mesh: Any = None,
    ) -> None:
        # vllm_config is read by the compile decorator (see MtpLoopStepModel).
        super().__init__()
        # Read by the plugin's compile_prefix patch: own compile-cache folder.
        self.prefix = prefix
        reason = kv_only_unsupported_reason(mtp)
        if reason is not None:
            raise ValueError(reason)
        self.mtp = mtp
        layer = _single_mtp_layer(mtp)
        assert layer is not None
        self._write = _KvOnlyWriteOp(layer.mtp_block.self_attn.mla_attn, mesh)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _, attn_input = mtp_layer_input(self.mtp, input_ids, positions, hidden_states)
        return self._write(positions, attn_input)
