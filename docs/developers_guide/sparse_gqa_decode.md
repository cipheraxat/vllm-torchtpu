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

`is_supported_by_fused_decode` checks shapes and static eligibility; callers
must supply valid selected IDs, lengths and page-table contents. The direct
entry rejects unsupported inputs; QSA retains the existing RPA fallback.
Batch/context benchmark maxima are not dispatch limits. Launch chunking bounds
attention work, while page lookup and allocations still scale with input shapes.

## Page mapping and ownership

Physical page size is `S = cache.shape[1]`; a selected group is a virtual page
of `r = ratio` tokens. With `a = S/r`, group `g` maps to
`table[g//a]*a + g%a`. A compute tile holds `bkv_tokens/r` virtual pages.
These three sizes are distinct. Duplicates preserve order and multiplicity;
partial tails are included and inactive padding is masked.

Selected groups become virtual pages without materializing selected KV in HBM.
The kernel returns the aliased cache, preserving donation and persistent cache
ownership. Cache update, indexer and prefill dispatch are unchanged. Page mapping
and chunk execution share the existing QSA RPA adapter; page lookup indirectly
imports MLA/Torch.

## Launch planning and policy

One immutable `DecodeLaunch` holds the RPA config, policy and resource breakdown;
geometry comes from config/cache shape. Public entry signatures stay compatible.
Actual smaller tensor N rebuilds scalar/page shapes; scan retains masked idle
padding. No Array/Tracer static keys or global executable cache are introduced.

Launch capacity follows selected-group budget, ratio, native schedule/scalar
bytes, query/KV buffers, gate and hardware SMEM/VMEM capacity. A configuration
that cannot fit one query falls back to RPA; larger batches are chunked. The
inherited 32 KiB SMEM reserve and 0.8 VMEM fraction are conservative compiler
headroom, not measured peak use. The fallback retains its 64-query/160 KiB and
scan-unroll-4 policies.

The immutable default uses a 256-token compute tile and eight pipeline lanes.
These are a measured compromise across representative GQA/MQA, BF16/FP8 and
resource-stressing inputs, not hardware laws or a universal optimum. Smaller
tiles increased scheduling work in the measured chunked case; larger tiles or
more lanes excluded resource-heavy cases. No calibration-shape whitelist or
user/environment tuning knob controls eligibility.

## DMA and numerical contracts

The unchanged shared RPA `bref_override.py` wait ABI reshapes uint32 data into
128-word rows: the wait byte unit derives as `128*sizeof(uint32)` (512 bytes).
`abi.py` requires schedule copies to preserve complete rows. Copy starts at zero
and length is `min(align_up(live_prefix_words, policy_words), capacity_words)`;
zero steps skip DMA and tails stay capped. Start and wait use the same slices.
The default copy policy is 128 words. Inherited 1024-word overfetch was not a
proven hardware minimum; the chosen complete-row policy is not a silicon minimum.
Unknown schedule fields fail closed; aliases derive from named ABI/tree leaves.

The existing RPA `rpa_body`, `create_allocs` and
`pipeline.run_manual_triple_buffer` remain visible. The pipeline keeps input/
output slots 3/2, N+2 fetch/N-2 wait and LAST_TILE scratch carry/reset. Sigmoid,
explicit BF16 rounding and multiplication precede the final output DMA. Integer
RNE, zeroing and fences are unchanged; moving these boundaries can alter results.

## Validation and usage

Run numerical and lifecycle tests in the project environment; hardware tests
carry `tpu_test` for the TPU CI queue:

```bash
pytest tests/kernels/test_sparse_gqa_decode.py tests/kernels/test_qwen4_exp_qsa_rpa.py \
  tests/kernels/test_sparse_gqa_decode_contracts.py \
  tests/kernels/test_qwen4_exp_adaptive_prefill.py
```

Coverage includes dispatch/fallback, both cache dtypes, an independent float64
oracle, non-unit scales, causal/duplicate/tail cases, poisoned inactive storage,
chunk/page/resource boundaries, exact cache bytes and prefill-to-decode lifecycle.

Use the benchmark harness for matched inputs and synchronized latency samples:

```bash
python scripts/benchmark_sparse_gqa_decode.py --help
python scripts/benchmark_sparse_gqa_decode.py \
  --batches 1 8 --contexts 8192 --output results/sparse-decode
```

Outputs include raw samples, numerical errors, executable memory statistics,
compiled HLO and optional `--trace` XProf captures. Initial and per-completed-case
post-compilation provenance record actual loaded source paths/hashes outside
timing; failed cases do not claim completed bindings. Performance conclusions
apply to measured configurations, not full-model serving or every supported shape.
