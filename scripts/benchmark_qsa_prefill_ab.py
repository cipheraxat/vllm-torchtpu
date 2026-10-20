# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reproduce sparse-prefill A/B cases against the pinned pre-review checkout.

Includes dispatch, selection metadata, attention and gating. Excludes indexer
scoring, cache writes, setup, compilation and warmup. Output-only, no donation.
Run --list-cases for exact local geometry and request-padding configurations.
"""

import argparse
import functools
import gzip
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
from jax.stages import Compiled

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
)
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import get_sparse_prefill_tiles
from vllm_torchtpu.kernels.sparse_gqa_prefill.adaptive import (
    adaptive_prefill,
    is_supported_by_adaptive_prefill,
)
from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import select_sc

BASELINE_COMMIT = "d3b574ec1f560ec45bb71c76a9258b8af9084d2a"


CASES = [
    # name,T,context,pattern,dtype,H,KV,page,padded,baseline
    ("common-fp8-tc", 8192, 8192, "recent", True, 24, 2, 1664, False, "adaptive"),
    (
        "common-fp8-sc",
        8192,
        262144 - 8192,
        "uniform",
        True,
        24,
        2,
        1664,
        False,
        "adaptive",
    ),
    ("common-bf16-tc", 8192, 8192, "recent", False, 24, 2, 1664, False, "adaptive"),
    (
        "common-bf16-sc",
        8192,
        131072 - 8192,
        "uniform",
        False,
        24,
        2,
        1664,
        False,
        "adaptive",
    ),
    (
        "runner-padded-common",
        8192,
        262144 - 8192,
        "uniform",
        True,
        24,
        2,
        1664,
        True,
        "tc",
    ),
    ("tail8448-padded", 8448, 262144 - 8448, "uniform", True, 24, 2, 1664, True, "tc"),
    (
        "tail8448-gcd-control",
        8448,
        8192,
        "recent",
        False,
        24,
        2,
        1664,
        False,
        "adaptive",
    ),
    ("short512-bf16", 512, 8192, "recent", False, 24, 2, 896, True, "tc"),
    (
        "long16640-recent",
        16640,
        131072 - 16640,
        "recent",
        True,
        24,
        2,
        1664,
        True,
        "tc",
    ),
    (
        "long16640-uniform",
        16640,
        131072 - 16640,
        "uniform",
        False,
        24,
        2,
        1664,
        True,
        "tc",
    ),
    ("local12-kv1", 8192, 8192, "recent", True, 12, 1, 896, True, "tc"),
    ("local8-kv4", 512, 8192, "uniform", False, 8, 4, 128, True, "tc"),
    (
        "short256-fp8",
        256,
        262144 - 256,
        "uniform",
        True,
        24,
        2,
        1664,
        False,
        "adaptive",
    ),
    ("short17-bf16", 17, 2049, "recent", False, 24, 2, 896, True, "tc"),
]


def load(name: str, path: Path) -> ModuleType:
    s = importlib.util.spec_from_file_location(name, path)
    assert s is not None and s.loader is not None
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def make(case: tuple) -> tuple[jax.Array, ...]:
    name, tokens, context, pattern, fp8, heads, kv, page, padded, _ = case
    rng = np.random.default_rng(2018)
    dtype, packing = (jnp.float8_e4m3fn, 4) if fp8 else (jnp.bfloat16, 2)
    q = jnp.asarray(rng.normal(0, 0.3, (tokens, heads, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 2, q.shape), jnp.bfloat16)
    lengths = context + np.arange(1, tokens + 1, dtype=np.int32)
    blocks = np.full((tokens, 512), -1, np.int32)
    for row, n in enumerate(lengths):
        complete = n // 4
        count = min(512, complete)
        blocks[row, :count] = (
            rng.choice(complete, count, replace=False)
            if pattern == "uniform"
            else np.arange(complete - count, complete)
        )
    live = tokens - 3 if padded else tokens
    if padded:
        lengths[live:] = 0
        blocks[live:] = -1
        q = q.at[live:].set(jnp.nan)
        gate = gate.at[live:].set(jnp.nan)
    pages = math.ceil((context + tokens) / page)
    table = jnp.broadcast_to(
        jnp.asarray(rng.permutation(pages), jnp.int32), (tokens, pages)
    )
    shape = (pages, page, math.ceil(2 * kv / packing), packing, 256)
    cache = jax.jit(lambda key: jax.random.normal(key, shape).astype(dtype))(
        jax.random.key(1964)
    )
    starts = jnp.asarray([0] + [live] * (8 if padded else 1), jnp.int32)
    return (q, gate, jnp.asarray(blocks), jnp.asarray(lengths), table, cache, starts)


def dispatch(*args: jax.Array, kv_heads: int) -> tuple[jax.Array, jax.Array]:
    if is_supported_by_adaptive_prefill(
        *args, kv_heads=kv_heads, ratio=4, k_scale=1.0, v_scale=1.0
    ):
        return adaptive_prefill(*args)
    qb, kt = get_sparse_prefill_tiles(args[0], args[5], kv_heads=kv_heads, ratio=4)
    return block_sparse_gqa_attention(
        *args, kv_heads=kv_heads, ratio=4, query_block=qb, kv_tile=kt
    )


def info(compiled: Compiled, name: str, output_dir: Path) -> dict:
    text = compiled.as_text()
    m = compiled.memory_analysis()
    packed = gzip.compress(text.encode())
    if len(packed) < 8 * 1024**2:
        (output_dir / (name + ".hlo.txt.gz")).write_bytes(packed)
    return dict(
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        text_bytes=len(text),
        compressed_bytes=len(packed),
        paired="prefill_paired_pages" in text,
        sc="sc_compact_grouped_prefill" in text,
        temp_bytes=m.temp_size_in_bytes,
        lines=[
            s.strip()[:500]
            for s in text.splitlines()
            if ("sc_compact_grouped_prefill" in s or "prefill_paired_pages" in s)
        ][:12],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline-checkout", type=Path, help="Clean checkout at " + BASELINE_COMMIT
    )
    p.add_argument("--output", type=Path, help="New output directory (must not exist)")
    p.add_argument(
        "--cases",
        nargs="+",
        choices=[c[0] for c in CASES],
        help="Case names; defaults to all listed cases",
    )
    p.add_argument(
        "--list-cases",
        action="store_true",
        help="Print exact cases and exit without initializing a backend",
    )
    a = p.parse_args()
    if a.list_cases:
        print(json.dumps(CASES, indent=2))
        return
    if a.baseline_checkout is None or a.output is None:
        p.error("--baseline-checkout and --output are required")
    baseline = a.baseline_checkout.resolve()
    if not (baseline / "src/vllm_torchtpu").is_dir():
        p.error("baseline checkout must contain src/vllm_torchtpu")
    actual = subprocess.check_output(
        ["git", "-C", str(baseline), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != BASELINE_COMMIT:
        p.error("baseline checkout must be at " + BASELINE_COMMIT)
    if subprocess.check_output(
        ["git", "-C", str(baseline), "diff", "--name-only", "HEAD", "--", "src"],
        text=True,
    ).strip():
        p.error("baseline tracked source must be clean")
    repo = Path(__file__).resolve().parents[1]
    if (
        not Path(inspect.getfile(adaptive_prefill))
        .resolve()
        .is_relative_to(repo / "src")
    ):
        p.error("candidate must import this checkout; run with PYTHONPATH=src")
    shared = [
        "qwen4_exp/rounding.py",
        "sparse_gqa_prefill/grouped.py",
        "sparse_gqa_prefill/gather.py",
        "sparse_gqa_prefill/occupancy.py",
        "sparse_gqa_prefill/tuned_block_sizes.py",
        "experimental/batched_rpa/utils.py",
    ]
    for name in shared:
        rel = Path("src/vllm_torchtpu/kernels") / name
        if (repo / rel).read_bytes() != (baseline / rel).read_bytes():
            p.error("shared baseline dependency differs: " + str(rel))
    output_dir = a.output.resolve()
    if output_dir.exists():
        p.error("output directory must not exist; preserve prior samples")
    output_dir.mkdir(parents=True)
    old_tc = load(
        "original_tc",
        baseline / "src/vllm_torchtpu/kernels/qwen4_exp/block_sparse_gqa_prefill.py",
    )
    old_tiles = load(
        "original_tiles",
        baseline / "src/vllm_torchtpu/kernels/qwen4_exp/prefill_tiles.py",
    )
    old_adaptive = load(
        "original_adaptive",
        baseline / "src/vllm_torchtpu/kernels/sparse_gqa_prefill/adaptive.py",
    )
    old_adaptive.candidate = old_tc
    old_cost = load(
        "original_cost",
        baseline / "src/vllm_torchtpu/kernels/sparse_gqa_prefill/cost_model.py",
    )
    old_adaptive.select_sc = old_cost.select_sc
    old_adaptive.tc_always_wins = old_cost.tc_always_wins
    indices = [i for i, c in enumerate(CASES) if a.cases is None or c[0] in a.cases]
    paths = [
        "qwen4_exp/block_sparse_gqa_prefill.py",
        "qwen4_exp/prefill_tiles.py",
        "sparse_gqa_prefill/adaptive.py",
        "sparse_gqa_prefill/cost_model.py",
        *shared,
    ]
    source = {
        arm: {
            name: hashlib.sha256(
                (root / "src/vllm_torchtpu/kernels" / name).read_bytes()
            ).hexdigest()
            for name in paths
        }
        for arm, root in [("prior", baseline), ("review", repo)]
    }
    provenance = dict(
        baseline_head=actual,
        candidate_head=subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256=source,
        python=platform.python_version(),
        versions={
            n: importlib.metadata.version(n)
            for n in ["jax", "jaxlib", "libtpu", "torch", "torch-tpu", "vllm", "numpy"]
        },
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output_dir / "driver.py").write_bytes(Path(__file__).read_bytes())
    assert jax.default_backend() == "tpu", (
        "Run on an allocated TPU; this is not a CPU benchmark"
    )
    for idx in indices:
        case = CASES[idx]
        name, tokens, context, pattern, fp8, heads, kv, page, padded, baseline = case
        path = output_dir / f"benchmark-{name}.json"
        assert not path.exists()
        args = make(case)
        if baseline == "adaptive":
            prior = old_adaptive.adaptive_prefill
        else:
            qb, kt = old_tiles.get_sparse_prefill_tiles(
                args[0], args[5], kv_heads=kv, ratio=4
            )
            prior = functools.partial(
                old_tc.block_sparse_gqa_attention,
                kv_heads=kv,
                ratio=4,
                query_block=qb,
                kv_tile=kt,
            )

        def output_only(f, *xs):
            return f(*xs)[0]

        funcs = {
            "prior": jax.jit(functools.partial(output_only, prior))
            .lower(*args)
            .compile(),
            "review": jax.jit(
                functools.partial(output_only, functools.partial(dispatch, kv_heads=kv))
            )
            .lower(*args)
            .compile(),
        }
        details = {
            arm: info(fn, name + "-" + arm, output_dir) for arm, fn in funcs.items()
        }
        outputs = {}
        samples = {arm: [] for arm in funcs}
        for iteration in range(13):
            order = ["prior", "review"] if iteration % 2 == 0 else ["review", "prior"]
            for arm in order:
                jax.block_until_ready(args)
                start = time.perf_counter_ns()
                out = funcs[arm](*args)
                jax.block_until_ready(out)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if iteration >= 3:
                    samples[arm].append(elapsed)
                outputs[arm] = out
        actual = np.asarray(outputs["review"], np.float32)
        wanted = np.asarray(outputs["prior"], np.float32)
        assert np.isfinite(actual).all() and np.isfinite(wanted).all()
        error = float(np.linalg.norm(actual - wanted) / np.linalg.norm(wanted))
        assert error < 0.01, error
        summaries = {
            arm: dict(
                median_ms=statistics.median(v),
                min_ms=min(v),
                max_ms=max(v),
                p25_ms=float(np.quantile(v, 0.25)),
                p75_ms=float(np.quantile(v, 0.75)),
            )
            for arm, v in samples.items()
        }
        routes = []
        if is_supported_by_adaptive_prefill(
            *args, kv_heads=kv, ratio=4, k_scale=1.0, v_scale=1.0
        ):
            for i in range(0, tokens, 8192):
                ls = args[3][i : i + 8192]
                ids = args[2][i : i + 8192]
                pad = (-len(ls)) % 256
                ls = jnp.pad(ls, (0, pad))
                ids = jnp.pad(ids, ((0, pad), (0, 0)), constant_values=-1)
                routes.append(
                    bool(
                        jax.jit(
                            functools.partial(
                                select_sc,
                                num_tiles=math.ceil(args[4].shape[1] * page / 256),
                                dtype=args[5].dtype,
                            )
                        )(ids, ls)
                    )
                )
        record = dict(
            provenance=provenance,
            case=dict(
                name=name,
                tokens=tokens,
                context=context,
                pattern=pattern,
                fp8=fp8,
                heads=heads,
                kv_heads=kv,
                page=page,
                padded=padded,
                baseline=baseline,
            ),
            warmups_per_arm=3,
            measurements_per_arm=10,
            alternating_order=True,
            api=(
                "output-only; no donation; unchanged cache intentionally not consumed"
            ),
            samples_ms=samples,
            summary=summaries,
            relative_l2=error,
            hlo=details,
            sc_choices=routes,
            devices=[str(d) for d in jax.devices()],
            scope=(
                "dispatch + selection metadata + attention + gate; "
                "no indexer/cache writes/model layers"
            ),
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "case": name,
                    "summary": summaries,
                    "routes": routes,
                    "relative_l2": error,
                }
            ),
            flush=True,
        )
        # Representative execution traces are recorded separately, after raw timings.
        del funcs, args, outputs
        jax.clear_caches()


if __name__ == "__main__":
    main()
