# Cost model technical appendix

The 14 coefficients were fitted to measured runtimes using nonnegative least
squares. They match calibration reference revision `6af38de8` exactly. The
measurements support routing within the original calibration scope; they do not
establish that every term is necessary or that the 2% margin is optimal.

## Formula and work counts

See the [cost model source][ref-1], [selector source][selector], and
[occupancy source][ref-2]. Predicted times are in milliseconds.

```text
TC = a0 + a1 × Qtc + a2 × Qtc × W + a3 × V
SC = b0 + b1 × Qsc + b2 × A

Select SC if TC > 1.02 × SC; otherwise select TC.
```

- **Qtc:** `queries`, including Q256 padding.
- **Qsc:** `lengths.size`, optionally sliced by `sc_queries`, including SC
  microchunk padding.
- **W:** `ceil(2 × num_tiles / 128) × 128` aligned 32-bit mask words. The caller
  supplies the context tile count.
- **V:** `tile_visits`, the sum over Q256 blocks of distinct KV256 tiles touched
  by valid queries.
- **A:** `attention_tiles`, the sum of KV256 attention tile counts for groups of
  eight queries.

For a query of length L, SC uses `selected = min(L // 4, 512) × 4 + L % 4`. Each
group of eight queries takes the maximum selected count, divides by 256 with
ceiling, and executes at least one tile, including for an all-inactive group.
The transport term counts scheduled queries, including padding.

The intercepts a0 and b0 absorb fixed costs. The other terms model query, mask,
and tile work. These interpretations describe model features; the coefficients
are not isolated hardware instruction latencies.

## Constants and their roles

- **Kernel tiles — 256:** TC query blocks and KV tiles. SC also counts KV256
  attention tiles.
- **SC grouping — `_SC_QUERY_GROUP = 8`:** Eight queries per group, with
  independent softmax operations.
- **Occupancy sampling — `_SAMPLED_QUERIES_PER_BLOCK = 8`:** Queries returned by
  `sampled_metadata` per Q256 block. This is separate from SC grouping.
- **Calibration geometry — 4 and 512:** Compression ratio four, up to 512
  complete selected groups, plus the partial tail.
- **Derived mask layout — 64 and 2:** A KV256 tile has 64 compressed groups
  represented by two 32-bit words.
- **Mask alignment — 128:** Alignment of the mask word dimension.
- **Outer dispatch — `PREFILL_DISPATCH_CHUNK_QUERIES = 8192`:** Maximum query
  rows per outer dispatch chunk. Larger inputs are split into full chunks and a
  remainder.
- **SC microchunks — 32 and 64:** BF16 and FP8 transport scheduling,
  respectively.
- **Policy — `_TC_PREFERENCE_FACTOR = 1.02`:** Prefer TC near the predicted
  crossover. This is not a confidence interval.
- **Weight floor — 0.25:** The floor in `1 / max(measured_ms, 0.25)` limits the
  influence of short cases.
- **Fit scales — 8192, 2048, 32768, 9216:** Numerical conditioning only. They
  are divided back out after fitting and are not runtime thresholds.

Work bounds are not measured latency bounds. The selector uses static maximum
work, full-query spans, sampled lower bounds, and coarse upper bounds before
computing exact bitmap unions when necessary. Nonnegative coefficients preserve
monotonicity within the model. Prediction errors still affect real performance.
A sampled subset that misses dispersed queries cannot establish that TC wins.

## Coefficients

Values come from the [saved fit][ref-3]. BF16 and FP8 refer to KV cache dtype;
query and gate inputs are BF16 in both cases.

| Symbol | Field | Unit |
| --- | --- | --- |
| a0 | `tc_fixed_ms` | ms |
| a1 | `tc_query_ms` | ms/query |
| a2 | `tc_mask_word_ms` | ms/(query·word) |
| a3 | `tc_tile_ms` | ms/visit |
| b0 | `sc_fixed_ms` | ms |
| b1 | `sc_query_ms` | ms/query |
| b2 | `sc_attention_tile_ms` | ms/tile |

| Symbol | BF16 KV | FP8 E4M3FN KV |
| --- | --- | --- |
| a0 | 0.25000650408450037 | 0.2491017824685177 |
| a1 | 8.991181022104898e-05 | 8.889156791776907e-05 |
| a2 | 4.814218452898145e-08 | 4.724934780077619e-08 |
| a3 | 0.004999265026206821 | 0.005145153550878911 |
| b0 | 0.19743081215678976 | 0.19896896616666457 |
| b1 | 0.005833960374709633 | 0.0032255933216442456 |
| b2 | 0.002476169547627867 | 0.0038026017010685054 |

The decimal precision preserves the fit output for comparison. It does not imply
equivalent measurement precision. Rounding or merging coefficients could change
decisions near a crossover and is outside this documentation change.

## Calibration and fitting

The [calibration data][ref-4] contains 60 cases, 30 per KV dtype:

- 27 combinations of query count {256, 2048, 8192}, context {8192, 65536,
  262144}, and pattern {uniform, recent, shared}.
- Three short-input cases with 8192 scheduled queries and active counts of 64,
  1024, or 4096.

The fixed scope was TPU7x, Q24/KV2/D256, ratio four, top-k 512, page size 1664,
and unit K/V scales. The [runtime record][ref-5] lists Python 3.12.3, JAX/JAXLIB
0.11.2, libtpu 0.0.50, NumPy 2.5.4, and SciPy 1.18.1.

The original calibration used input seed `41001 + case_index` and cache seed
2117. Each arm receives one correctness call and two warmups before 12
alternating measurements. Calls donate the cache and synchronize both output and
cache. Compilation and input construction are outside timing. Saved samples are
durations, not paired start/end timestamps.

[fit.py][ref-6] reads only calibration data and fits TC and SC separately for
each dtype with SciPy `nnls`:

```text
TC features = [1, Q/8192, Q×W/(8192×2048), V/32768]
SC features = [1, Q/8192, A/9216]
w = 1 / max(y, 0.25)
Objective = minimize Σ [w × (predicted_ms − measured_median_ms)]²
```

Dividing the fitted coefficients by the feature scales produces the runtime
coefficients. Weighting emphasizes relative error so long-running cases do not
dominate. The fit uses measured times, not desired route labels, and does not
read validation data.

## Historical validation

The records contain [72 held-out cases][ref-7], [eight full-call or multichunk
cases][ref-8], and [eight context-edge cases][ref-9]. Eighty cases use a single
chunk. Each compares forced TC, forced SC, adaptive, and the previous
implementation, with three warmup rounds and 30 synchronized samples per arm.
Arm order rotates; 30 rounds do not balance four starting positions exactly.

Validation uses input seed `91001 + case_index`, distinct from calibration, but
retains cache seed 2117. Numerical checks include whole-output relative L2
against TC, independent FP64 references for up to five active rows, finiteness,
and inactive zeros. These are standalone kernel checks, not checkpoint or
model-serving tests.

The saved data contains 12000 durations and 246 chunk decisions. The table
summarizes their historical results; adding these files does not constitute a
new experiment.

| Metric | BF16 KV | FP8 KV |
| --- | --- | --- |
| Calibration TC error¹ | 2.78% / 9.19% | 2.59% / 8.97% |
| Calibration SC error¹ | 0.84% / 1.81% | 0.47% / 1.05% |
| Validation TC error¹ | 2.83% / 10.46% | 2.84% / 11.31% |
| Validation SC error¹ | 0.93% / 3.06% | 0.58% / 1.95% |
| Faster forced arm selected² | 40/40 | 40/40 |
| Maximum dispatch overhead³ | 8.27% | 8.86% |

¹ Mean / maximum relative prediction error.

² Single-chunk cases, using measured medians.

³ Adaptive time above the faster forced arm in validation.

Selecting the faster branch does not remove dispatch overhead. Multichunk calls
can mix branches, so whole-call forced-arm timings cannot establish the optimal
branch for each individual chunk.

## Limits and implications for simplification

- **Term necessity:** no recorded per-term removal, refit, and independent
  validation establishes which terms can be removed.
- **Margin choice:** no threshold sweep was found. The closest measured
  single-chunk crossovers had forced-arm median gaps of about 3.29% for BF16 and
  4.81% for FP8. The data does not cover measured gaps of 0–2% or justify
  treating 2% as an error bound.
- **Generalization:** repeated samples within a case are not an independent
  multi-process confidence study. Other chips, head layouts, dimensions, page
  geometries, and serving workloads require separate validation.
- **Validation independence:** coefficients were frozen before held-out
  validation, but dispatcher revisions v3/v4/v5 used validation feedback. These
  cases were excluded from coefficient fitting; they were not an untouched blind
  test of the final overall policy.
- **Revision coverage:** later `sc_queries` and tail/padding handling are absent
  from the historical harness. Matching coefficients does not make historical
  timings measurements of every path at the reviewed head.

Keep the formula, coefficients, margin, and dispatcher unchanged for this
documentation update. A later simplification should compare routing errors, full
dispatch overhead, and run-to-run variation, using near-crossover cases that
were not used for tuning. Training error alone is insufficient.

See the [offline reproduction guide][ref-10] for the saved inputs, fitter,
hashes, and commands. This change preserves coefficient-fit reproduction; it
does not package the historical TPU harness or claim a new hardware run.

[ref-1]: ../../src/vllm_torchtpu/kernels/sparse_gqa_prefill/cost_model.py
[ref-2]: ../../src/vllm_torchtpu/kernels/sparse_gqa_prefill/occupancy.py
[ref-3]: ../../scripts/sparse_gqa_prefill_cost_model/expected-fit.json
[ref-4]: ../../scripts/sparse_gqa_prefill_cost_model/calibration/results.json
[ref-5]: ../../scripts/sparse_gqa_prefill_cost_model/runtime.json
[ref-6]: ../../scripts/sparse_gqa_prefill_cost_model/fit.py
[ref-7]: ../../scripts/sparse_gqa_prefill_cost_model/validation/heldout-results.json
[ref-8]: ../../scripts/sparse_gqa_prefill_cost_model/validation/full-results.json
[ref-9]: ../../scripts/sparse_gqa_prefill_cost_model/validation/context-edges-results.json
[ref-10]: ../../scripts/sparse_gqa_prefill_cost_model/README.md

[selector]: ../../src/vllm_torchtpu/kernels/sparse_gqa_prefill/selector.py
