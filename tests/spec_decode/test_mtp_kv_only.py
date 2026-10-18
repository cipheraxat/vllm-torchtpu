# SPDX-License-Identifier: Apache-2.0
"""Two-pass MTP draft step 0 (TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS).

CPU tests of the proposer side: when the two-pass path is taken, the narrowed
pass-2 metadata, and that a propose with the flag on leaves the same draft
tokens, KV / indexer-K caches and IndexShare rows as the single full pass.
The draft is a toy that keeps the property the argument rests on: rows couple
only through a causal read of caches every row writes first.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

import vllm_torchtpu.spec_decode.eagle3 as e3
from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.spec_decode.eagle3 import DraftChunkInputs, Eagle3Proposer

DRAFT_LAYER = "model.layers.78.mtp_block.self_attn.attn"
HIDDEN = 4
VOCAB = 32
TOPK = 2


def _make_proposer(method: str = "mtp") -> Eagle3Proposer:
    speculative_config = SimpleNamespace(
        draft_tensor_parallel_size=1, method=method, draft_model_config=None
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    return Eagle3Proposer(
        runner=mock.MagicMock(_pcp_enabled=False, _mtp_enabled=method == "mtp"),
        vllm_config=vllm_config,
    )


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def _gating_proposer(monkeypatch, *, min_tokens=8, lockstep=False, target_md=True):
    monkeypatch.setattr(envs, "TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS", min_tokens)
    proposer = _make_proposer()
    proposer._mtp_kv_only_model = object()
    proposer._dp_lockstep_sharded = lambda: lockstep
    proposer._target_draft_attn_metadata = lambda chunk: (
        {DRAFT_LAYER: object()} if target_md else None
    )
    return proposer


@pytest.mark.parametrize(
    "kwargs, num_tokens, num_rows, expected",
    [
        ({}, 16, 4, True),
        ({}, 8, 4, True),  # exactly the gate
        ({}, 7, 4, False),  # below the gate
        ({"min_tokens": 4}, 4, 4, False),  # nothing to drop: T == p
        ({"min_tokens": 0}, 4096, 16, False),  # flag off
        ({"lockstep": True}, 16, 4, False),  # EP-DP dummy replay
        ({"target_md": False}, 16, 4, False),  # dummy chunk / no target md
    ],
)
def test_use_kv_only_two_pass_gate(monkeypatch, kwargs, num_tokens, num_rows, expected):
    proposer = _gating_proposer(monkeypatch, **kwargs)
    assert proposer._use_kv_only_two_pass(object(), num_tokens, num_rows) is expected


def test_use_kv_only_two_pass_needs_the_program(monkeypatch):
    proposer = _gating_proposer(monkeypatch)
    proposer._mtp_kv_only_model = None
    assert proposer._use_kv_only_two_pass(object(), 4096, 16) is False


def _build_proposer(monkeypatch, *, min_tokens=1024, method="mtp"):
    monkeypatch.setattr(envs, "TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS", min_tokens)
    proposer = _make_proposer(method)
    proposer.draft_model = object()
    proposer.draft_vllm_config = object()
    proposer.runner = SimpleNamespace(_pcp_enabled=False, mesh=object())
    return proposer


def _patch_builder(monkeypatch, reason=None, raises=False):
    import vllm_torchtpu.spec_decode.mtp_kv_only as kv_only

    built = []

    class _FakeModel:
        def __init__(self, **kwargs):
            if raises:
                raise RuntimeError("boom")
            built.append(kwargs)

    monkeypatch.setattr(kv_only, "kv_only_unsupported_reason", lambda mtp: reason)
    monkeypatch.setattr(kv_only, "MtpKvOnlyModel", _FakeModel)
    for name in ("set_model_tag", "set_vllm_model_wrapper_context"):
        monkeypatch.setattr(e3, name, lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(
        e3, "set_current_vllm_config", lambda *a, **k: contextlib.nullcontext()
    )
    return built


def test_build_kv_only_model_off_by_default(monkeypatch):
    proposer = _build_proposer(monkeypatch, min_tokens=0)
    built = _patch_builder(monkeypatch)
    proposer._build_mtp_kv_only_model()
    assert proposer._mtp_kv_only_model is None
    assert built == []


def test_build_kv_only_model_builds_with_draft_config_and_own_prefix(monkeypatch):
    proposer = _build_proposer(monkeypatch)
    built = _patch_builder(monkeypatch)
    proposer._build_mtp_kv_only_model()
    assert proposer._mtp_kv_only_model is not None
    (kwargs,) = built
    assert kwargs["vllm_config"] is proposer.draft_vllm_config
    assert kwargs["mtp"] is proposer.draft_model
    assert kwargs["mesh"] is proposer.runner.mesh
    assert kwargs["prefix"] == "mtp_prefill_kv_only"


@pytest.mark.parametrize(
    "setup",
    ["eagle3", "kv_shared", "pcp", "unsupported", "build_fails"],
)
def test_build_kv_only_model_refusals_keep_the_single_pass(monkeypatch, setup):
    proposer = _build_proposer(
        monkeypatch, method="eagle3" if setup == "eagle3" else "mtp"
    )
    built = _patch_builder(
        monkeypatch,
        reason="not DSA" if setup == "unsupported" else None,
        raises=setup == "build_fails",
    )
    if setup == "kv_shared":
        proposer._draft_kv_shared = True
    if setup == "pcp":
        proposer.runner._pcp_enabled = True
    proposer._build_mtp_kv_only_model()
    assert proposer._mtp_kv_only_model is None
    assert built == []


# ---------------------------------------------------------------------------
# Pass-2 metadata
# ---------------------------------------------------------------------------


def _ctx(seq_lens, qsl, num_reqs, knr):
    seq = np.zeros(knr, dtype=np.int32)
    seq[: len(seq_lens)] = seq_lens
    qsl_full = np.full(knr + 1, qsl[-1], dtype=np.int32)
    qsl_full[: len(qsl)] = qsl
    return SimpleNamespace(
        use_max_model_len=True,
        seq_lens=torch.from_numpy(seq),
        query_start_loc=torch.from_numpy(qsl_full),
        request_distribution=torch.tensor([0, num_reqs, num_reqs], dtype=torch.int32),
        position_ids_override=None,
    )


@pytest.mark.parametrize("async_rejected", [False, True])
def test_pass2_attn_metadata_is_one_query_per_request(async_rejected):
    proposer = _make_proposer()
    knr = 6
    proposer.runner = SimpleNamespace(
        num_reqs_max_model_len=knr, num_reqs_most_model_len=None, device="cpu"
    )
    chunk = SimpleNamespace(
        num_reqs=3, attn_ctx=_ctx([10, 6, 8], [0, 4, 10, 15], 3, knr)
    )
    md = AttentionMetadata(
        input_positions=torch.zeros(16, dtype=torch.int32),
        block_tables=torch.arange(12, dtype=torch.int32),
        seq_lens=chunk.attn_ctx.seq_lens,
        query_start_loc=chunk.attn_ctx.query_start_loc,
        request_distribution=chunk.attn_ctx.request_distribution,
    )
    other = AttentionMetadata(input_positions=torch.zeros(16, dtype=torch.int32))
    target_md = {"a": md, "b": md, "c": other}
    rejected = np.array([2, 0, 0], dtype=np.int32)
    if async_rejected:
        rejected = torch.from_numpy(rejected)

    out = proposer._pass2_attn_metadata(chunk, target_md, rejected)

    assert out["a"] is out["b"]  # shared metadata stays shared
    new = out["a"]
    assert new.block_tables is md.block_tables
    assert new.seq_lens.tolist() == [8, 6, 8, 0, 0, 0]
    assert new.query_start_loc.tolist() == [0, 1, 2, 3, 3, 3, 3]
    assert new.request_distribution.tolist() == [3, 3, 3]
    # The target's own metadata is untouched.
    assert md.seq_lens is chunk.attn_ctx.seq_lens
    assert md.query_start_loc is chunk.attn_ctx.query_start_loc


def test_save_topk_snapshots_the_front_rows():
    proposer = _make_proposer()
    buf = torch.arange(24, dtype=torch.int32).reshape(6, 4)
    proposer._mtp_topk_buffers = [buf]
    (saved,) = proposer._mtp_save_topk(2)
    assert torch.equal(saved, buf[:2])
    buf.zero_()
    assert saved.sum() > 0  # a copy, not a view


# ---------------------------------------------------------------------------
# Flag on == flag off, on a toy draft
# ---------------------------------------------------------------------------


class _ToyWorld:
    """Per-request caches indexed by absolute position, plus the shared top-k
    table. `write` and `attend` are the only row couplings, as in the MTP
    layer (KV + indexer-K writes, causal sparse attention over them)."""

    def __init__(self, num_reqs: int, max_len: int, prefix_lens):
        g = torch.Generator().manual_seed(0)
        self.kv = torch.zeros(num_reqs, max_len, HIDDEN)
        self.idx = torch.zeros(num_reqs, max_len)
        for r, n in enumerate(prefix_lens):
            self.kv[r, :n] = torch.randn(n, HIDDEN, generator=g)
            self.idx[r, :n] = torch.randn(n, generator=g)
        self.topk_buf = torch.full((32, TOPK), -1, dtype=torch.int32)
        self.md = None
        self.skip_topk = False
        self.step0_rows_seen: list[int] = []
        self.kv_only_rows_seen: list[int] = []

    def rows(self, num_tokens: int, positions: torch.Tensor):
        """(row, request, position) of every row the metadata places."""
        (md,) = {id(m): m for m in self.md.values()}.values()
        qsl = md.query_start_loc.tolist()
        seq_lens = md.seq_lens.tolist()
        num_reqs = int(md.request_distribution[2])
        out = []
        for t in range(num_tokens):
            r = next((r for r in range(num_reqs) if qsl[r] <= t < qsl[r + 1]), None)
            if r is None:
                continue
            pos = seq_lens[r] - (qsl[r + 1] - qsl[r]) + (t - qsl[r])
            # The metadata and the row's own position agree.
            assert pos == int(positions[t]), (t, r, pos, int(positions[t]))
            out.append((t, r, pos))
        return out

    def write(self, input_ids, positions, hidden):
        placed = self.rows(input_ids.shape[0], positions)
        for t, r, pos in placed:
            self.kv[r, pos] = hidden[t] * 0.5 + input_ids[t] * 0.01 + pos * 0.001
            self.idx[r, pos] = hidden[t].sum() + input_ids[t] * 0.1
        return placed

    def forward(self, input_ids, positions, hidden_states, spec_step_idx=0):
        placed = self.write(input_ids, positions, hidden_states)
        out = hidden_states.clone()
        for t, r, pos in placed:
            if self.skip_topk:
                sel = self.topk_buf[t]
            else:
                scores = self.idx[r, : pos + 1]
                k = min(TOPK, pos + 1)
                sel = torch.full((TOPK,), -1, dtype=torch.int32)
                sel[:k] = torch.topk(scores, k).indices.to(torch.int32)
                self.topk_buf[t] = sel
            valid = sel[sel >= 0].long()
            out[t] = (
                hidden_states[t] * 0.9
                + self.kv[r, : pos + 1].mean(dim=0) * 0.3
                + self.kv[r, valid].sum(dim=0) * 0.2
            )
        if spec_step_idx == 0 and not self.skip_topk:
            self.step0_rows_seen.append(input_ids.shape[0])
        return out, out * 2.0


class _ToyMtpModel:
    def __init__(self, world):
        self.world = world

    def set_skip_topk(self, skip):
        self.world.skip_topk = skip

    def compact_topk_indices(self, slot_ids):
        buf = self.world.topk_buf
        buf[: slot_ids.numel()] = buf[slot_ids]


class _ToyDraft:
    def __init__(self, world):
        self.world = world
        self.model = _ToyMtpModel(world)
        g = torch.Generator().manual_seed(1)
        self.head = torch.randn(HIDDEN, VOCAB, generator=g)

    def __call__(self, *, input_ids, positions, hidden_states, spec_step_idx):
        return self.world.forward(input_ids, positions, hidden_states, spec_step_idx)

    def compute_logits(self, hidden):
        return hidden @ self.head


class _ToyLoopProgram:
    """IndexShare's loop-step program: the same forward, run with the
    indexer off (it reads the saved top-k rows)."""

    def __init__(self, world):
        self.world = world

    def __call__(self, *, input_ids, positions, hidden_states, spec_step_idx):
        assert self.world.skip_topk
        return self.world.forward(input_ids, positions, hidden_states, spec_step_idx)


class _ToyKvOnly:
    def __init__(self, world):
        self.world = world
        self.calls = 0

    def __call__(self, input_ids, positions, hidden_states):
        self.calls += 1
        placed = self.world.write(input_ids, positions, hidden_states)
        self.world.kv_only_rows_seen.append(len(placed))
        return hidden_states


class _Req:
    def __init__(self, tokens, num_computed):
        self.tokens = tokens
        self.num_computed_tokens = num_computed

    def get_token_id(self, i):
        return self.tokens[i]


# Three requests in one chunk, K = 3:
#   r0 verify: context 6, 1 + K = 4 tokens (positions 6..9), 2 drafts rejected;
#   r1 prefill whose prompt completes: 6 tokens (positions 0..5);
#   r2 intermediate prefill chunk: 5 tokens (positions 3..7), discarded.
_Q_LENS = [4, 6, 5]
_CONTEXT = [6, 0, 3]
_REJECTED = [2, 0, 0]


def _scenario(mode: str):
    num_reqs = 3
    knr = 4
    T_real = sum(_Q_LENS)
    T = 16
    qsl = np.concatenate([[0], np.cumsum(_Q_LENS)]).astype(np.int32)
    seq_lens = [c + q for c, q in zip(_CONTEXT, _Q_LENS)]
    positions = np.zeros(T, dtype=np.int32)
    for r in range(num_reqs):
        positions[qsl[r] : qsl[r + 1]] = np.arange(_CONTEXT[r], seq_lens[r])
    input_ids = np.zeros(T, dtype=np.int32)
    input_ids[:T_real] = np.arange(T_real) % 7 + 3
    g = torch.Generator().manual_seed(2)
    hidden = torch.randn(T, HIDDEN, generator=g)
    md = AttentionMetadata(
        input_positions=torch.from_numpy(positions),
        block_tables=torch.zeros(num_reqs * 4, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens + [0] * (knr - num_reqs), dtype=torch.int32),
        query_start_loc=torch.tensor(
            list(qsl) + [qsl[-1]] * (knr - num_reqs), dtype=torch.int32
        ),
        request_distribution=torch.tensor([0, num_reqs, num_reqs], dtype=torch.int32),
    )
    attn_ctx = SimpleNamespace(
        use_max_model_len=True,
        seq_lens=md.seq_lens,
        query_start_loc=md.query_start_loc,
        request_distribution=md.request_distribution,
        position_ids_override=None,
    )
    chunk = DraftChunkInputs(
        input_ids=torch.from_numpy(input_ids),
        position_ids=torch.from_numpy(positions),
        query_start_loc_np=qsl,
        attn_ctx=attn_ctx,
        start_index=0,
        num_reqs=num_reqs,
        aux_hidden_states=[],
        hidden_states=hidden,
        attn_metadata={DRAFT_LAYER: md, "target.attn": md},
        draft_lengths=torch.tensor([3, 0, 0], dtype=torch.int32),
    )
    requests = {
        "r2": _Req(list(range(100, 120)), _CONTEXT[2]),
    }
    kwargs = dict(
        discard_sampled_tokens_req_indices=[2],
        scheduler_output=SimpleNamespace(
            num_scheduled_tokens={"r0": 4, "r1": 6, "r2": 5},
            scheduled_spec_decode_tokens={"r0": [0, 0, 0]},
        ),
    )
    if mode == "sync":
        kwargs.update(
            sampled_token_ids=[[11, 12], [13], []],
            num_rejected_tokens_np=np.array(_REJECTED, dtype=np.int32),
        )
    else:
        # The rejection-sampler output that gives the same seeds / rejections:
        # r0 keeps 1 draft + bonus (2 of 4 valid -> 2 rejected).
        kwargs.update(
            sampled_token_ids=None,
            num_rejected_tokens_np=None,
            next_tokens_per_chunk=[
                torch.tensor(
                    [[11, 12, -1, -1], [13, -1, -1, -1], [0, -1, -1, -1]],
                    dtype=torch.int32,
                )
            ],
            return_device=True,
        )
    prefix_lens = _CONTEXT
    return chunk, requests, kwargs, prefix_lens, seq_lens


def _run(monkeypatch, *, two_pass: bool, mode: str):
    import torch._dynamo

    torch._dynamo.reset()
    chunk, requests, kwargs, prefix_lens, seq_lens = _scenario(mode)
    world = _ToyWorld(3, 24, prefix_lens)
    K = 3

    proposer = _make_proposer()
    proposer.speculative_config.num_speculative_tokens = K
    proposer.draft_vllm_config = object()
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=3, req_ids=["r0", "r1", "r2"]),
        requests=requests,
        device=torch.device("cpu"),
        num_tokens_paddings=[4, 8, 16, 32],
        max_num_reqs=4,
        num_reqs_max_model_len=4,
        num_reqs_most_model_len=None,
        uses_mrope=False,
        mesh=object(),
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda n: n,
        _has_mamba_state=False,
        _unified_kv_layout=False,
    )
    proposer._draft_attn_layer_names = {DRAFT_LAYER}
    proposer.draft_model = _ToyDraft(world)
    proposer.draft_chunks = [chunk]
    # IndexShare, on the toy's shared top-k table.
    proposer._share_mtp_indices = True
    proposer._mtp_topk_buffers = [world.topk_buf]
    proposer._mtp_loop_model = _ToyLoopProgram(world)
    proposer._mtp_loop_model_warm = True

    monkeypatch.setattr(envs, "TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS", 8)
    kv_only = _ToyKvOnly(world)
    proposer._mtp_kv_only_model = kv_only if two_pass else None

    @contextlib.contextmanager
    def fake_forward_context(attn_metadata, *_a, **_k):
        world.md = attn_metadata
        yield

    monkeypatch.setattr(e3, "set_forward_context", fake_forward_context)
    for name in ("set_model_tag", "set_vllm_model_wrapper_context"):
        monkeypatch.setattr(e3, name, lambda *a, **k: contextlib.nullcontext())

    drafts = proposer.propose(**kwargs)
    if isinstance(drafts, torch.Tensor):
        drafts = drafts.tolist()
    return SimpleNamespace(
        drafts=drafts,
        world=world,
        kv_only_calls=kv_only.calls,
        seq_lens=seq_lens,
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_two_pass_step0_matches_the_single_pass(monkeypatch, mode):
    ref = _run(monkeypatch, two_pass=False, mode=mode)
    new = _run(monkeypatch, two_pass=True, mode=mode)

    # The paths actually differ: one full step 0 on all 16 rows vs a KV-only
    # pass over the 15 real rows and a full pass on p = 4 rows.
    assert ref.kv_only_calls == 0
    assert ref.world.step0_rows_seen == [16]
    assert new.kv_only_calls == 1
    assert new.world.kv_only_rows_seen == [15]
    assert new.world.step0_rows_seen == [4]

    assert new.drafts == ref.drafts
    # Same cache entries everywhere: prefix, this step's rows (rejected tail
    # included), and the K - 1 loop steps' draft entries.
    assert torch.equal(new.world.kv, ref.world.kv)
    assert torch.equal(new.world.idx, ref.world.idx)
    # The loop steps' top-k rows (IndexShare) are the same per request.
    assert torch.equal(new.world.topk_buf[:3], ref.world.topk_buf[:3])


def test_two_pass_step0_is_not_taken_below_the_gate(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS", 32)
    proposer = _make_proposer()
    proposer._mtp_kv_only_model = object()
    proposer._dp_lockstep_sharded = lambda: False
    proposer._target_draft_attn_metadata = lambda chunk: {DRAFT_LAYER: object()}
    assert proposer._use_kv_only_two_pass(object(), 16, 4) is False
