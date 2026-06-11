"""
T14 Ablation Analyzer: paired t-test on beta-C ON vs OFF val accuracy.

Per R7 design (docs/superpowers/specs/2026-06-11-fractal-vit-classification-tokenization-synergy-r7-design.md)
Decision 9c and R7 Oracle report Q5 verdict.

Gating logic:
  - For each seed s, compute delta_s = val_acc_ON(s) - val_acc_OFF(s)
  - Run scipy.stats.ttest_rel(on, off) (paired, two-sided)
  - SHIP WITH beta-C iff:
        p_value < 0.05  AND  mean_ON > mean_OFF
  - Otherwise: SHIP WITHOUT beta-C (R6 closure path)

Usage:
    uv run python scripts/analyze_t14_ablation.py
    uv run python scripts/analyze_t14_ablation.py --results artifacts/t14/t14_results.json
    uv run python scripts/analyze_t14_ablation.py --results t14_results.json --verdict-out t14_verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# 统计依赖
try:
    import numpy as np
    from scipy import stats
except ImportError as e:  # pragma: no cover
    print(f"ERROR: missing dependency: {e}. Run `uv add scipy numpy`.", file=sys.stderr)
    sys.exit(2)

ALPHA = 0.05  # 配对 t 检验显著性阈值 (R7 Decision 9c, Q5)


def _load_results(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_paired_runs(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Group runs by seed; each seed must have both ON and OFF entries."""
    by_seed: Dict[int, Dict[str, float]] = {}
    for run in runs:
        if run.get("val_accuracy") is None:
            print(
                f"WARNING: skipping seed={run.get('seed')} beta-C={run.get('beta_c')} "
                f"(no val_accuracy; likely training failure)",
                file=sys.stderr,
            )
            continue
        seed = int(run["seed"])
        beta_c = bool(run["beta_c"])
        by_seed.setdefault(seed, {})["ON" if beta_c else "OFF"] = float(run["val_accuracy"])

    paired: Dict[str, Dict[str, float]] = {}
    for seed, accs in by_seed.items():
        if "ON" not in accs or "OFF" not in accs:
            print(
                f"WARNING: seed={seed} has only {list(accs.keys())}; skipping from paired test",
                file=sys.stderr,
            )
            continue
        paired[str(seed)] = accs
    return paired


def analyze(results_path: Path) -> Dict[str, Any]:
    data = _load_results(results_path)
    runs = data.get("runs", [])
    if not runs:
        raise ValueError(f"No runs in {results_path}")

    paired = _extract_paired_runs(runs)
    if len(paired) < 2:
        raise ValueError(
            f"T14 requires >=2 paired seeds (3 per R7). Found {len(paired)}: {list(paired.keys())}"
        )

    seeds_sorted = sorted(paired.keys(), key=int)
    on_vals = np.array([paired[s]["ON"] for s in seeds_sorted], dtype=float)
    off_vals = np.array([paired[s]["OFF"] for s in seeds_sorted], dtype=float)
    diffs = on_vals - off_vals  # positive => beta-C helped

    # Paired t-test, two-sided
    t_stat, p_value = stats.ttest_rel(on_vals, off_vals)
    # scipy returns nan if variance is zero (rare). Treat nan-p as 1.0 to fail safe.
    if np.isnan(p_value):
        p_value = 1.0
    if np.isnan(t_stat):
        t_stat = 0.0

    mean_on = float(np.mean(on_vals))
    mean_off = float(np.mean(off_vals))
    std_on = float(np.std(on_vals, ddof=1)) if len(on_vals) > 1 else 0.0
    std_off = float(np.std(off_vals, ddof=1)) if len(off_vals) > 1 else 0.0
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    mean_diff = float(np.mean(diffs))

    # ----- Verdict (R7 Decision 9c, Q5) -----
    # SHIP WITH beta-C iff: p<0.05 AND mean_ON > mean_OFF (one-sided spirit)
    # Two-sided p is conservative; we additionally require directional superiority
    # to avoid shipping beta-C on noise in the wrong direction.
    if p_value < ALPHA and mean_on > mean_off:
        verdict = "SHIP WITH beta-C"
        ship_with_beta_c = True
        verdict_reason = (
            f"Paired t-test: ON ({mean_on:.3f}%) > OFF ({mean_off:.3f}%), "
            f"p={p_value:.4f} < {ALPHA} -> beta-C provides statistically significant improvement"
        )
    else:
        verdict = "SHIP WITHOUT beta-C"
        ship_with_beta_c = False
        if mean_on <= mean_off:
            verdict_reason = (
                f"Paired t-test: ON ({mean_on:.3f}%) <= OFF ({mean_off:.3f}%); "
                f"beta-C does not improve mean val accuracy. R6 closure path: ship v1.3 with A only"
            )
        else:
            verdict_reason = (
                f"Paired t-test: ON ({mean_on:.3f}%) > OFF ({mean_off:.3f}%) but "
                f"p={p_value:.4f} >= {ALPHA} (n={len(diffs)} seeds, insufficient evidence). "
                f"R6 closure path: ship v1.3 with A only"
            )

    return {
        "protocol": "T14",
        "design_ref": "docs/superpowers/specs/2026-06-11-fractal-vit-classification-tokenization-synergy-r7-design.md",
        "oracle_ref": "docs/superpowers/specs/2026-06-11-fractal-vit-classification-tokenization-synergy-r7-oracle-report.md",
        "source_results": str(results_path),
        "n_seeds": len(diffs),
        "seeds": [int(s) for s in seeds_sorted],
        "per_seed": {
            s: {"on": paired[s]["ON"], "off": paired[s]["OFF"], "delta": paired[s]["ON"] - paired[s]["OFF"]}
            for s in seeds_sorted
        },
        "mean_ON": mean_on,
        "mean_OFF": mean_off,
        "mean_diff": mean_diff,
        "std_ON": std_on,
        "std_OFF": std_off,
        "std_diff": std_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "alpha": ALPHA,
        "verdict": verdict,
        "ship_with_beta_c": ship_with_beta_c,
        "verdict_reason": verdict_reason,
    }


def _print_human_summary(v: Dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print("T14 Ablation Verdict (beta-C ON vs OFF on CUB-200)")
    print("=" * 60)
    print(f"Seeds:               {v['n_seeds']} ({', '.join(str(s) for s in v['seeds'])})")
    print(f"Mean val_acc ON:     {v['mean_ON']:.4f} %  (std {v['std_ON']:.4f})")
    print(f"Mean val_acc OFF:    {v['mean_OFF']:.4f} %  (std {v['std_OFF']:.4f})")
    print(f"Mean delta (ON-OFF): {v['mean_diff']:+.4f} %  (std {v['std_diff']:.4f})")
    print(f"Paired t-stat:       {v['t_stat']:.4f}")
    print(f"p-value (two-sided): {v['p_value']:.4f}  (alpha={v['alpha']})")
    print()
    for s, accs in v["per_seed"].items():
        print(f"  seed {s}: ON={accs['on']:.4f}  OFF={accs['off']:.4f}  delta={accs['delta']:+.4f}")
    print()
    print(f"VERDICT: {v['verdict']}")
    print(f"Reason:  {v['verdict_reason']}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="T14 ablation paired t-test analyzer")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/t14/t14_results.json"),
        help="Path to t14_results.json from run_t14_ablation.ps1",
    )
    parser.add_argument(
        "--verdict-out",
        type=Path,
        default=Path("artifacts/t14/t14_verdict.json"),
        help="Path to write t14_verdict.json",
    )
    args = parser.parse_args()

    try:
        verdict = analyze(args.results)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    args.verdict_out.parent.mkdir(parents=True, exist_ok=True)
    with args.verdict_out.open("w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    print(f"Verdict written to: {args.verdict_out}")

    _print_human_summary(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
