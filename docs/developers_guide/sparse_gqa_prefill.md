# Adaptive sparse GQA prefill

`kernels/sparse_gqa_prefill` provides an opt-in attention kernel. It selects
between query-block KV reuse on TensorCore (TC) and per-query selected-KV
transport on SparseCore (SC). QSA integration is reviewed separately in #1964;
adding the standalone kernel does not change any model's default path.

## Supported envelope

The calibrated schedule is limited to TPU7x with SparseCore and the pinned
JAX/JAXLIB 0.11.2 and libtpu 0.0.50. Query and gate are BF16 `[T, 24, 256]`,
with two KV heads, compression ratio four, and 512 selected groups per query.
T is positive and at most 262144; a final partial tile is masked.

The native cache uses BF16 or FP8 E4M3FN packing with int32 metadata. The
wrapper passes unit K/V scales to both branches, so nonunit or dynamic scales
must retain the general TC path.

Page size 1664, context/query count 262144, and the 16 GiB pool guard describe
the validated calibration envelope. The table endpoint is
`ceil(262144 / 1664) = 158` columns, derived from that context endpoint, rather
than an addressing limit. The lookup supports padded 128-column chunks;
changing page/context geometry changes its work and has not been calibrated
here. Physical row addresses must fit int32, but the 16 GiB guard is stricter
than that implementation constraint and is not a hardware capacity claim.
Q24/KV2/D256 and ratio four likewise scope this fitted selector, not general
SC support across local head layouts. Broader SC calibration is separate work;
other configurations retain TC. The paired TC policy has no D256 upper bound:
its dimension alignment and working-set checks determine eligibility.

Request starts are nondecreasing, starting at zero, and include padded slots.
Active rows have `min(length // 4, 512)` unique valid complete groups in a
prefix, followed by -1 IDs, plus the implicit partial tail. Inactive rows
have zero length and -1 IDs. Their output is zero even if their query or
gate contains NaN. The kernel returns the cache unchanged.

`is_supported_by_adaptive_prefill` checks static eligibility. Callers must
provide valid runtime metadata. Padded slots have repeated starts and do not
prevent optimized routing.
Boundaries inside a Q256 tile retain request-aware TC alignment because the
calibrated union costs do not include that padding. Large shared pools may exceed
the cache bound. Kernel benchmarks do not establish model-serving coverage.

The paired TC tile policy is separate: it derives eligibility from local Q/KV
heads, GQA divisibility, dimension, native packing, 128-token page segments,
hardware and a working-set bound relative to the measured tile. It does not
assume a tensor-parallel degree or use token buckets as layout constraints.
Other valid geometry retains the general one-selection-word TC path. This
resource bound is not a latency prediction for unmeasured head layouts.

## Dispatch and memory

Outer chunks contain 8192 queries followed by a bounded remainder. Only the
last TC tile or SC microchunk is padded; full inputs and selected KV are never
padded to a context-sized buffer. The cost model estimates TC work
from Q256/KV256 tile unions and SC work from scheduled queries and Q8/KV256
attention tiles. Coefficients are specific to the supported device and
geometry; they are not universal hardware constants.

See [Understanding the cost model](sparse_gqa_prefill_cost_model.md) for the
coefficient sources, named constants, evidence limits, and offline reproduction.

SC is selected when estimated TC cost exceeds estimated SC cost by 2%.
Static maximum-work bounds, full-query spans, sampled lower bounds and
coarse occupancy upper bounds can resolve the decision early. Otherwise,
exact bitmap union counts resolve it. A sampled subset may establish SC,
but cannot establish TC by missing dispersed queries. CPU tests compare
these bounds with explicit tile sets, including partial tails and padding.

SC gathers native packed KV into bounded compact buffers. BF16 uses 32
queries per gather and FP8 uses 64; both use 144 MiB of compact storage at
the supported top-k. Grouped attention keeps each query's softmax independent.
FP32 accumulation and BF16 output/gate rounding retain the existing QSA
arithmetic boundaries. Multi-chunk calls write into the final output and
reuse scratch. Compiler temporary memory is tested separately from total HBM.
The 400 MiB mixed-output check is an empirical regression target, not a device
capacity or measured runtime peak. The full-context compile-only check compares
16384 and 262144 queries at fixed cache geometry: additional scratch must fit
within the increase in source-shaped selection metadata. This detects a new
whole-query/gate workspace without choosing another absolute compiler budget.
The earlier 640 MiB assertion is tracked in review measurements, including any
regression against its unchanged parent. Neither check establishes latency or
peak-HBM guarantees. Ragged TC tiles write directly into the final output,
avoiding a full-chunk alignment/scatter workspace. Selected-KV scratch is bounded
by the microchunk, independently of unused physical cache pages.

Returning an unchanged cache from a standalone, non-donating JIT can still
materialize an observable returned array. Output-only callers eliminate that
return. The pooled serving custom op already donates its mutable pool; this
kernel does not introduce donation or change cache ownership.

## Reproduction

Run the standalone CPU and TPU suites in the pinned environment:

```bash
python -m pytest -q -m 'not tpu_test' \
  tests/kernels/test_sparse_gqa_grouped_prefill.py \
  tests/kernels/test_sparse_gqa_prefill_cost_model.py \
  tests/kernels/test_sparse_gqa_adaptive_prefill.py
python -m pytest -q -m tpu_test \
  tests/kernels/test_sparse_gqa_grouped_prefill.py \
  tests/kernels/test_sparse_gqa_adaptive_prefill.py
```

Compare adaptive against fixed Q256/KV256 attention with identical inputs:

```bash
for dtype in bfloat16 float8_e4m3fn; do
  PYTHONPATH=src python scripts/benchmark_qsa_prefill.py \
    --candidate adaptive --baseline paired --cache-dtype "$dtype" \
    --page-size 1664 --tokens 8192 --contexts 0 122880 \
    --patterns recent uniform --iterations 10 \
    --output "/tmp/adaptive-prefill-$dtype"
done
```

The script records raw synchronized timings, versions, HLO and XProf traces.
It includes dispatch, selection metadata, gather, attention and gating;
excludes indexer scoring, cache writes, compilation and other model layers.
The 128K final KV case is an 8K query chunk plus 120K cached history.

The separate QSA integration tests exercise real runner-derived metadata,
the native metadata builder, Qwen adapter and pooled custom op/indexer in direct
and fullgraph TPU calls. Scheduler/storage/page ownership and projected inputs
are fixtures; these are not checkpoint, full-model or distributed-TP tests.
The captured FP8 long-context call executes SC in direct, compiled and repeated
compiled markers. Large setup copies occur before those markers. This does not
establish universal copy absence or runtime peak HBM. Full-model execution,
concurrent serving and larger pools still require separate validation.
Decode is outside this standalone kernel.

## Reproduce the review A/B

`scripts/benchmark_qsa_prefill_ab.py` preserves the measured seeds, data,
shuffled physical pages, selected groups and padded request starts. It uses
three warmups and ten synchronized, alternating measurements per arm.
The baseline is the clean pre-review commit
`d3b574ec1f560ec45bb71c76a9258b8af9084d2a`; it uses adaptive where originally
supported and that checkout's TC tiles otherwise. Both arms consume identical
arrays. Setup, compilation, indexer scoring and cache writes are excluded;
dispatch, selector metadata, attention and gating are included. Calls consume
only output, without donation. JSON records raw samples, summaries, source and
dependency hashes/versions, HLO fingerprints and device names.

```bash
git worktree add --detach /path/on/data-disk/qsa-baseline d3b574ec1
PYTHONPATH=src python scripts/benchmark_qsa_prefill_ab.py --list-cases
PYTHONPATH=src python scripts/benchmark_qsa_prefill_ab.py \
  --baseline-checkout /path/on/data-disk/qsa-baseline \
  --output /path/on/data-disk/qsa-ab-results \
  --cases common-fp8-tc common-fp8-sc common-bf16-tc common-bf16-sc \
    runner-padded-common tail8448-padded tail8448-gcd-control \
    long16640-recent long16640-uniform short512-bf16 local12-kv1 local8-kv4 \
    short17-bf16
```

`--list-cases` exposes exact token/context counts, local Q/KV heads, page sizes,
KV types and padding for all cases. These include page896, multiple GQA ratios
and repeated padded starts; no TP degree is an implementation input. The nine
adaptive cases were rerun on the final kernel. Four TC cases retain measurements
from an earlier snapshot whose invoked TC/tiling/rounding files are byte-identical.
The listed command reproduces those thirteen comparisons; it does not claim
that packaging the CLI is a new hardware run. Limited measurements are not a
universal latency guarantee.
