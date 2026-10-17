# Fused sparse GQA decode

QSA dispatches supported decode inputs to `kernels/sparse_gqa_decode` and keeps
batched RPA for other decode configurations. The prefill branch is unchanged.

## Supported inputs

- TPU7x; BF16 queries and gates; positive query/KV head counts with query
  heads divisible by KV heads, and head dimension aligned to hardware lanes.
- Positive selected-group budget; compression ratio divides the KV tile.
  The partial tail is included after the selected complete groups.
- Native `HEAD_ALONG_SUBLANE` BF16 or FP8 E4M3FN cache with a positive page
  size divisible by the compression ratio.
- Nonempty batches and page tables; physical token offsets, logical context
  lengths and flattened page-table indices must fit int32.
- Positive finite scalar K/V scales and the existing valid-prefix selection
  contract. Cache already contains the current decode token.

`is_supported_by_fused_decode` owns this envelope. The direct entry rejects
unsupported inputs; the QSA dispatcher retains the existing RPA fallback.
Batch size and context length are not capped at the benchmark maxima. Attention
launches are bounded by chunking; page-table lookup and tensor allocations still
consume resources in proportion to their shapes.

## Implementation and ownership

Selected groups become virtual pages of `ratio` tokens; no selected-KV tensor is
materialized in HBM. A compact vector schedule preserves request order and
skips inactive rows. Launch size is derived from the selected-group budget,
compression ratio, native RPA schedule size, query/KV buffers, gate and hardware
SMEM/VMEM capacity. The planner retains RPA's compiler headroom. A configuration
that cannot fit one query falls back to RPA; larger batches are chunked.
For Q24/KV2/D256 with ratio 4 and 512 groups this still selects 64 queries per
launch (at most 72 decode steps). Other geometries do not inherit these constants.

The new kernel calls the existing RPA `rpa_body`, `create_allocs`, and RPA
`pipeline.run_manual_triple_buffer`. It applies sigmoid, the existing explicit
BF16 rounding boundaries, and multiplication before the final output DMA.
It returns the aliased cache so callers retain donation and persistent cache ownership.

The upstream RPA files are unchanged. Page mapping
and the chunk-execution helper are shared with the existing QSA RPA adapter.
The RPA adapter retains its original batch-size policy. The QSA
cache update, indexer and prefill dispatch are unchanged.

The decode specialization originated in `tokamax/experiments/sparse_gqa_decode_v2`.
It composes the RPA implementation in the #1964 base; the eleven RPA files match
upstream commit `e79cc0cd17a92d1cc009e0e2c84568e314bd9afd` byte-for-byte.

## Validation

Run the numerical and lifecycle regression tests in the project environment:

```bash
pytest tests/kernels/test_sparse_gqa_decode.py tests/kernels/test_qwen4_exp_qsa_rpa.py
```

The tests cover dispatch and fallback, both cache dtypes, a float64 oracle,
non-unit FP8 scales, partial groups, empty requests, poisoned inactive storage,
the B65/257/513 chunk boundaries, page sizes 4/128/132/256/1664, contexts
through 524289 tokens, cache bytes, and prefill-to-decode state transitions.
The new hardware tests carry the `tpu_test` marker for the TPU CI queue.

## Performance

Measured on one TPU7x chip, 2026-10-17, JAX/JAXLIB 0.11.2 and libtpu 0.0.50.
Both arms use the repository's same startup patches, native cache layout and
inputs. Each result is the median of 50 alternating measurements after three
warmups. The boundary includes metadata, layouts, attention, gate and blocking
completion; compilation and initialization are excluded.

At context 8192 and page size 1664:

| Batch | BF16 RPA → fused (µs) | Speedup | FP8 RPA → fused (µs) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 288.2 → 254.4 | 1.133× | 285.3 → 254.6 | 1.120× |
| 2 | 305.4 → 273.1 | 1.118× | 301.3 → 267.9 | 1.125× |
| 4 | 318.1 → 285.8 | 1.113× | 316.9 → 288.6 | 1.098× |
| 8 | 339.8 → 312.5 | 1.087× | 338.1 → 309.8 | 1.091× |
| 16 | 384.7 → 368.2 | 1.045× | 388.7 → 366.6 | 1.060× |
| 32 | 486.6 → 396.2 | 1.228× | 486.2 → 393.2 | 1.237× |
| 64 | 686.0 → 511.9 | 1.340× | 679.6 → 499.4 | 1.361× |
| 128 | 1150.5 → 806.8 | 1.426× | 1136.4 → 783.2 | 1.451× |
| 256 | 2066.3 → 1382.6 | 1.495× | 2054.9 → 1339.2 | 1.534× |

Additional long-context and page128 measurements:

| Context / page size | Batch | BF16 speedup | FP8 speedup |
| --- | ---: | ---: | ---: |
| 262144 / 1664 | 1 | 1.115× | 1.105× |
| 262144 / 1664 | 16 | 1.043× | 1.050× |
| 8193 / 128 | 8 | 1.079× | 1.096× |
| 8193 / 128 | 64 | 1.322× | 1.364× |

All 26 combinations improved. Geometric mean speedup is **1.187× BF16** and
**1.198× FP8**; ranges are 1.043–1.495× and 1.050–1.534× respectively.
Paired bootstrap 95% intervals from the 50 samples have lower bounds above 1
in every case. The largest measured relative L2 difference against RPA is
1.53e-8. These measurements sample the supported envelope, not every possible
batch/context combination.

Additional measurements beyond the original benchmark range, with the same
hardware, versions, timing boundary and 50-sample protocol:

| Batch | Context | Page size | BF16 speedup | FP8 speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 2051 | 4 | 1.076× | 1.088× |
| 8 | 8193 | 132 | 1.089× | 1.094× |
| 64 | 8193 | 132 | 1.352× | 1.370× |
| 257 | 8193 | 256 | 1.475× | 1.511× |
| 512 | 8193 | 256 | 1.549× | 1.595× |
| 1 | 524289 | 256 | 1.141× | 1.140× |
| 4 | 524289 | 256 | 1.106× | 1.104× |

All 14 additional cases improve, with paired bootstrap 95% lower bounds above 1.
These are measured configurations, not dispatch limits or a guarantee of a
speedup at every larger shape. The attention math is unchanged; the resource
planner retains chunks of up to 64 queries for this Q24/KV2/D256, ratio4,
512-group geometry.

XProf captures three completed calls per arm. At BF16 B64/context8192,
device-active time (union of XLA-op intervals) falls from 459.7 to 284.2 µs,
and operations per call from 113 to 75. The principal attention kernel itself
rises from 202.0 to 227.5 µs: the overall gain comes from reducing surrounding
work, not a faster QK/softmax/PV implementation. At FP8 B64, device-active time
falls from 457.3 to 274.0 µs and operations from 115 to 75.
At B512/context8193/page256, the additional traces show device-active time
falling from 3687.8 to 2302.6 µs for BF16 and 3669.7 to 2218.5 µs for FP8;
operations per call fall from 942 to 732 for both.

Compiled temporary allocation at BF16 B64 falls from 17,021,952 to 3,517,536
bytes; cache alias bytes remain 1,090,519,040 for both. This is executable memory
analysis, not a full-model peak-HBM measurement.

### Geometry and resource generalization

The current planner passed 66 CPU and 52 TPU cases, including nondefault head
geometry, ratio 8/16, budgets 17/64/128/256/1024, and NaNs in unused FP8 MQA
packing channels. Existing configurations and cache lifecycle tests also pass.

A further 22 matched A/B cases use context 8195, page 128, the same environment,
and 50 alternating samples. Four additional geometries run at B8/B64:

| Q / KV / D | Ratio / groups | BF16 speedup B8 / B64 | FP8 speedup B8 / B64 |
| --- | --- | --- | --- |
| 12 / 2 / 128 | 4 / 128 | 1.041× / 1.165× | 1.046× / 1.183× |
| 16 / 4 / 256 | 8 / 256 | 1.041× / 1.245× | 1.030× / 1.237× |
| 8 / 1 / 128 | 16 / 64 | 0.999× / 1.090× | 1.004× / 1.096× |
| 24 / 2 / 512 | 4 / 1024 | 1.120× / 1.447× | 1.131× / 1.430× |

The original Q24/KV2/D256, ratio4, 512-group geometry was rechecked at B1/B64/B256:
BF16 speedups are 1.138× / 1.344× / 1.493×; FP8 is 1.131× / 1.346× / 1.526×.
Twenty cases have paired bootstrap 95% lower bounds above 1. MQA B8 is at parity:
BF16 0.999× (95% CI 0.989–1.010), FP8 1.004× (0.994–1.011).
The maximum relative L2 difference versus RPA is 1.11e-8.

XProf confirms the distinction: BF16 D512/B64 device-active time falls
918.6→566.0 µs and XLA operations 215→160, despite the principal kernel rising
212.2→226.4 µs. BF16 MQA/B8 device-active time is nearly unchanged,
59.0→58.7 µs. No universal speedup is claimed.

Reproduce the measured cases in the project TPU environment:

```bash
for dtype in bf16 fp8; do
  args=()
  if [ "$dtype" = fp8 ]; then args+=(--fp8); fi
  python scripts/benchmark_sparse_gqa_decode.py "${args[@]}" \
    --trace --output "results/$dtype-main"
  python scripts/benchmark_sparse_gqa_decode.py "${args[@]}" \
    --batches 1 16 --contexts 262144 --output "results/$dtype-long"
  python scripts/benchmark_sparse_gqa_decode.py "${args[@]}" \
    --batches 8 64 --contexts 8193 --page-size 128 \
    --trace --output "results/$dtype-page128"
done
```

Each output directory contains timing samples, numerical errors, executable
memory statistics, compiled HLO and, with `--trace`, XProf traces. Each request
owns separate native cache pages; context includes the current decode token.
Reproduce the additional configurations:

```bash
for dtype in bf16 fp8; do
  args=()
  if [ "$dtype" = fp8 ]; then args+=(--fp8); fi
  python scripts/benchmark_sparse_gqa_decode.py "${args[@]}" \
    --batches 257 512 --contexts 8193 --page-size 256 \
    --trace --output "results/$dtype-large-batch"
  python scripts/benchmark_sparse_gqa_decode.py "${args[@]}" \
    --batches 1 4 --contexts 524289 --page-size 256 \
    --output "results/$dtype-wide-table"
  python scripts/benchmark_sparse_gqa_decode.py "${args[@]}" \
    --batches 8 64 --contexts 8193 --page-size 132 \
    --output "results/$dtype-page132"
  python scripts/benchmark_sparse_gqa_decode.py "${args[@]}" \
    --batches 8 --contexts 2051 --page-size 4 \
    --output "results/$dtype-page4"
done
```

Reproduce the geometry sweep (add `--fp8` for the second cache dtype):

```bash
for spec in "12 2 128 4 128" "16 4 256 8 256" "8 1 128 16 64" "24 2 512 4 1024"; do
  read -r heads kv dim ratio groups <<< "$spec"
  python scripts/benchmark_sparse_gqa_decode.py \
    --heads "$heads" --kv-heads "$kv" --dim "$dim" \
    --ratio "$ratio" --groups "$groups" \
    --batches 8 64 --contexts 8195 --page-size 128 --trace \
    --output "results/geometry-$heads-$kv-$dim-$ratio-$groups"
done
python scripts/benchmark_sparse_gqa_decode.py \
  --batches 1 64 256 --contexts 8195 --page-size 128 --trace \
  --output results/geometry-original
```

The tested source base is #1964 at `16d79f36b327f304952852198f4c0c765af01ef6`.

These measurements exclude index selection, KV updates, RoPE, projections and
full-model serving. Prefill is unchanged. No performance claim applies outside
the measured configurations.
