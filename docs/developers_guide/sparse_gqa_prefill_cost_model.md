# Sparse prefill cost model

The selector compares TensorCore block-sparse attention (TC) with SparseCore
selected-KV gather plus grouped attention (SC). Both preserve the same QSA
semantics. The model estimates work, not measured latency bounds.

## Work and decision

Predicted times are in milliseconds:

```text
TC = a0 + a1 * Qtc + a2 * Qtc * W + a3 * V
SC = b0 + b1 * Qsc + b2 * A
Select SC if TC > 1.02 * SC; otherwise select TC.
```

- `Qtc` includes Q256 padding; `Qsc` includes SC microchunk padding and may be
  bounded by `sc_queries` for a tail.
- `W = ceil(2 * num_tiles / 128) * 128` is the aligned mask-word count.
- `V` sums distinct KV256 tiles per Q256 block over valid queries.
- `A` sums KV256 attention tiles per group of eight queries. Each group takes
  its largest selected length and executes at least one tile, even if inactive.
  Selected length is `min(L // 4, 512) * 4 + L % 4` for query length `L`.

The selector refines static bounds, full-query spans, a sampled lower bound,
a coarse upper bound, then exact bitmap counts. Nonnegative coefficients keep
these work bounds monotone. Eight sampled queries per Q256 block are distinct
from SC's eight-query attention groups. Sampling may establish SC, but cannot
establish TC by missing dispersed queries.

## Constants and scope

The fitted schedule uses TPU7x, Q24/KV2/D256, ratio four, top-k 512 groups,
page size 1664 and unit K/V scales. Tiles of 256, mask alignment of 128 and
SC grouping of eight describe that schedule; they are not hardware capacity
limits. BF16/FP8 gather microchunks of 32/64 bound selected-KV storage to
144 MiB; outer dispatch chunks are at most 8192 queries.

The `1.02` factor prefers TC near the predicted crossover. It is a policy
margin, not a confidence interval. The coefficients and this margin are not
universal across hardware or geometry. No per-term ablation or margin sweep
establishes that every term is necessary or that 2% is optimal. Prediction
error and selector overhead still require matched runtime validation.

## Calibration and reproduction

The 14 coefficients come from 60 calibration cases, 30 per dtype: 27
query/context/pattern combinations plus three partial-active cases. The
historical sampling used two warmups and 12 alternating synchronized durations
per forced arm, with cache donation; compilation and input construction were
outside timing. Input/cache seeds were `41001 + case_index` and `2117`.

The CPU fitter uses nonnegative least squares separately for TC/SC and each
dtype. Its features are `[1, Q/8192, Q*W/(8192*2048), V/32768]` for TC and
`[1, Q/8192, A/9216]` for SC, weighted by `1/max(measured_ms, 0.25)`. Conditioning
scales are divided back out; they are not runtime thresholds. See the
[calibration data][data], [fitter][fit], [expected fit][expected] and
[offline commands][reproduce]. The expected fit retains exact coefficient
precision and per-case predictions without duplicating them here.

These October 17, 2026 measurements reference
`6af38de82602deb158b18a31ef9268bacdc59d92`.
They are not current serving measurements. Historical validation and its
packaging records remain in the [fixed source snapshot][archive]. Validation
feedback informed dispatcher revisions, and later tail/padding handling was
outside that harness; it is not an untouched blind test of every current path.
Repeated samples do not establish cross-process or full-model generality.

## Implementation and current checks

- [cost_model.py][cost]: coefficients and feature units.
- [selector.py][selector]: ordered bounds and branch selection.
- [occupancy.py][occupancy]: exact counts and bounds.
- [prefill guide][guide]: current eligibility, tests and reusable benchmarks.

[data]: ../../scripts/sparse_gqa_prefill_cost_model/calibration/results.json
[fit]: ../../scripts/sparse_gqa_prefill_cost_model/fit.py
[expected]: ../../scripts/sparse_gqa_prefill_cost_model/expected-fit.json
[reproduce]: ../../scripts/sparse_gqa_prefill_cost_model/README.md
[archive]: https://github.com/vllm-project/vllm-torchtpu/tree/b56b6ec6928edef4972793d4651e84454938d927/scripts/sparse_gqa_prefill_cost_model
[cost]: ../../src/vllm_torchtpu/kernels/sparse_gqa_prefill/cost_model.py
[selector]: ../../src/vllm_torchtpu/kernels/sparse_gqa_prefill/selector.py
[occupancy]: ../../src/vllm_torchtpu/kernels/sparse_gqa_prefill/occupancy.py
[guide]: sparse_gqa_prefill.md
