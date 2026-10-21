# Reproduce the sparse prefill coefficient fit

This directory preserves the timings used to fit the 14 TC/SC coefficients. The
fitter runs on CPU with NumPy and SciPy; it does not import JAX or require TPU
access. Start with the [cost model overview][ref-1].

## Saved evidence

- [Calibration][ref-2]: 60 cases, work features, 12 durations per forced arm,
  and medians.
- [Expected fit][ref-3]: Original coefficients, predictions, and fitting errors.
- [Held-out validation][ref-4]: 72 cases with four arms and 30 durations per
  arm.
- [Full calls][ref-5]: Eight full-call or multichunk cases.
- [Context edges][ref-6]: Eight context-edge cases.
- [Runtime][ref-7]: Historical environment versions.
- [Frozen proof][ref-8]: Original calibration and fit hashes, and frozen
  coefficients.
- [Provenance][ref-9]: File hashes, calibration reference commit, and fitter
  changes.

All raw JSON values are unchanged; a final newline was added where missing.
`provenance.json` records both original and repository hashes. The frozen proof
retains the original hashes. The fitter retains the original expressions, case
selection, and working-directory I/O. Its changes are a license header,
formatting, and names for conditioning scales, the weight floor, and case count.

The data was measured on October 17, 2026, using TPU7x, Q24/KV2/D256, ratio
four, top-k 512, page size 1664, and unit K/V scales. The hardware sampling
protocol and validation limits are described in the [technical
appendix][ref-10]. These are historical standalone-kernel measurements, not
current serving results.

## Offline fitting

Use an existing Python environment with NumPy and SciPy. The recorded versions
are NumPy 2.5.4 and SciPy 1.18.1. Different versions may produce floating-point
differences. No installation or hardware measurement is needed by the script.

From this directory, create a separate output directory to preserve the inputs
and expected output:

```bash
REPRO_DIR="$PWD"
FIT_RUN_DIR="$(mktemp -d /tmp/sparse-prefill-fit.XXXXXX)"
mkdir "$FIT_RUN_DIR/calibration"
cp "$REPRO_DIR/calibration/results.json" "$FIT_RUN_DIR/calibration/results.json"
cd "$FIT_RUN_DIR"
python "$REPRO_DIR/fit.py"
```

Compare the generated `fit.json` with `expected-fit.json`: check
`coefficients_ms`, prediction errors, and per-case predictions under both `bf16`
and `fp8`, for both `tc` and `sc`. Preserve and report any differences along
with the environment. Do not overwrite the saved evidence or production
coefficients with the new output.

The original fitter skips a dtype unless it has exactly 30 cases. Verify input
hashes from `provenance.json`, then confirm that the output contains both
dtypes, all four fits, and all 14 coefficients. The fitter reads calibration
data only; it neither refits from validation data nor updates production code.

## Scope

An offline refit verifies coefficient provenance. It does not measure latency on
current hardware, validate later padding changes, or establish that every model
term is necessary. The historical TPU harness and binary/profile archives are
not included here. The existing `scripts/benchmark_qsa_prefill.py` and
`scripts/benchmark_qsa_prefill_ab.py` exercise repository kernels; they do not
regenerate these exact historical samples.

[ref-1]: ../../docs/developers_guide/sparse_gqa_prefill_cost_model.md
[ref-2]: calibration/results.json
[ref-3]: expected-fit.json
[ref-4]: validation/heldout-results.json
[ref-5]: validation/full-results.json
[ref-6]: validation/context-edges-results.json
[ref-7]: runtime.json
[ref-8]: frozen-coefficients-proof.json
[ref-9]: provenance.json
[ref-10]: ../../docs/developers_guide/sparse_gqa_prefill_cost_model_details.md
