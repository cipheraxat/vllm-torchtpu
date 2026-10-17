# Adaptive sparse GQA prefill

`kernels/sparse_gqa_prefill` provides an opt-in attention kernel. It selects
between query-block KV reuse on TensorCore (TC) and per-query selected-KV
transport on SparseCore (SC). QSA integration is reviewed separately in #1964;
adding the standalone kernel does not change any model's default path.

## Supported envelope

The calibrated schedule is limited to TPU7x with SparseCore and the pinned
JAX/JAXLIB 0.11.2 and libtpu 0.0.50. Query and gate are BF16 `[T, 24, 256]`,
with two KV heads, compression ratio four, and 512 selected groups per query.
T is a positive multiple of 256, at most 262144.

The native cache has 1664-token pages and BF16 or FP8 E4M3FN packing. The
cache tensor must fit within 16 GiB; the page table has at most 158 columns.
K/V scales must be statically known Python scalars equal to one.

One request slot is supported (`starts.shape == (2,)`, starting at zero).
Active rows have `min(length // 4, 512)` unique valid complete groups in a
prefix, followed by -1 IDs, plus the implicit partial tail. Inactive rows
have zero length and -1 IDs. Their output is zero even if their query or
gate contains NaN. The kernel returns the cache unchanged.

`is_supported_by_adaptive_prefill` checks static eligibility. Callers must
provide valid runtime metadata. Padded multi-slot metadata falls back even
when only one request is active. Large shared serving pools may also exceed
the cache bound. Kernel benchmarks do not establish model-serving coverage.

## Dispatch and memory

Outer chunks contain `gcd(T, 8192)` queries. The cost model estimates TC work
from Q256/KV256 tile unions and SC work from scheduled queries and Q8/KV256
attention tiles. Coefficients are specific to the supported device and
geometry; they are not universal hardware constants.

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

Full model execution, Torch bridge behavior, concurrent requests and larger
shared cache pools require separate validation. Decode is outside this kernel.
