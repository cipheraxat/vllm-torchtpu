# Understanding the cost model

## What it does

Sparse GQA prefill has two attention paths. **TensorCore (TC)** reuses KV data
across a block of queries. **SparseCore (SC)** gathers the KV data selected by
each query before computing attention. Which path is faster depends on the
input.

The cost model estimates both runtimes and selects a path. It does not change
the mathematical definition of attention.

## How it chooses

| Path | Work included in the estimate |
| --- | --- |
| TC | Fixed overhead, query count, mask processing, and KV tile visits |
| SC | Fixed overhead, query count, and attention tiles |

A tile is a block of data processed by the kernel. Fitted coefficients convert
these work counts into milliseconds. BF16 and FP8 KV caches use separate
coefficients.

**Choose SC only when predicted TC time exceeds 1.02 times predicted SC time.
Otherwise, keep TC.** For example, predicted times of 10 ms for TC and 9.9 ms
for SC select TC; 10 ms and 9 ms select SC. These are illustrative numbers. The
2% margin is a policy choice, not an error guarantee.

## Reading the implementation

- `cost_model.py`: coefficients and runtime estimates in milliseconds.
- `selector.py`: ordered bounds and the TC/SC decision.
- `adaptive.py`: chunk scheduling and execution of the chosen branch.

The selector tries static bounds, full-row upper bounds, a sampled lower bound,
a coarse upper bound, and finally exact counts. Each stage can stop the search.
Sampling can select SC; it cannot establish that TC is preferable.

## Where the numbers come from

| Kind of number | Examples | Meaning |
| --- | --- | --- |
| Kernel layout | 256, 8, 128 | Tile sizes, query grouping, and alignment |
| Fitted coefficients | 14 values | Convert work counts into time |
| Dispatch policy | 1.02 | Prefer TC when predictions are close |

The coefficients have a traceable source: **timings from 60 calibration cases →
fitting script → saved fit → production coefficients**. All 14 values match the
saved fit at the calibration reference revision.

Calibration used TPU7x, 24 query heads, 2 KV heads, head dimension 256,
compression ratio 4, top-k 512, and page size 1664. These coefficients are not
universal parameters for other hardware or shapes.

## What the evidence supports

In historical validation, all **80 single-chunk cases** selected the branch with
the lower measured median time. Selection still has overhead: adaptive execution
was up to **8.9% slower** than directly running the faster forced branch.

Three questions remain open:

- Can a model term be removed? There is no recorded per-term ablation.
- Is 2% the best margin? There is no threshold sweep or sufficient
  near-crossover coverage.
- Do these results cover later code and other workloads? The historical harness
  does not validate every current path. Validation feedback also informed
  dispatcher implementation changes.

## Where to start

Keep the algorithm and coefficients unchanged while reviewing their provenance:

1. [Calibration data][ref-1]: work counts and raw timings.
2. [Fitting script][ref-2]: computes coefficients from saved data on CPU.
3. [Expected fit][ref-3]: coefficients and prediction errors.

See the [reproduction guide][ref-4] for commands. A new TPU run is not needed to
inspect these materials. Any later simplification should compare routing errors
and full dispatch overhead, then use validation cases that were not used for
tuning.

Measurements date from October 17, 2026. The reviewed revision is
`6af38de82602deb158b18a31ef9268bacdc59d92`. Details are in the [technical
appendix][ref-5].

[ref-1]: ../../scripts/sparse_gqa_prefill_cost_model/calibration/results.json
[ref-2]: ../../scripts/sparse_gqa_prefill_cost_model/fit.py
[ref-3]: ../../scripts/sparse_gqa_prefill_cost_model/expected-fit.json
[ref-4]: ../../scripts/sparse_gqa_prefill_cost_model/README.md
[ref-5]: sparse_gqa_prefill_cost_model_details.md
