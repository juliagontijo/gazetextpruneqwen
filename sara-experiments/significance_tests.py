"""
significance_tests.py — Statistical significance tests on evaluation results.

Tests:
  - Preference: binomial test (null: pruned wins 50% of the time)
  - Accuracy:   McNemar's test (paired, same samples across configs)
  - Judge/Coverage: Wilcoxon signed-rank test vs no_prune baseline

Usage:
    python significance_tests.py
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict
from scipy import stats
import numpy as np

HERE = Path(__file__).parent


def load(path):
    if not path.exists():
        print(f"WARNING: {path} not found")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def preference_significance(rows):
    print("\n" + "═"*65)
    print("  PREFERENCE — Binomial test (null: B wins 50% of the time)")
    print("  A = no_prune baseline wins  |  B = pruned wins")
    print("═"*65)
    print(f"  {'Config':<35}  {'B%':>5}  {'n':>4}  {'p-value':>9}  {'sig':>5}")
    print("  " + "-"*63)

    configs = defaultdict(lambda: {"A": 0, "B": 0, "Tie": 0})
    for r in rows:
        w = r.get("llm_pref_winner", "").strip()
        if w in ("A", "B", "Tie"):
            configs[r["config_tag"]][w] += 1

    for tag in sorted(configs.keys()):
        d = configs[tag]
        # Exclude ties for binomial test
        decisive = d["A"] + d["B"]
        if decisive == 0:
            continue
        b_wins = d["B"]
        # One-sided: is B winning MORE than 50%?
        p = stats.binomtest(b_wins, decisive, p=0.5, alternative="greater").pvalue
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print(f"  {tag:<35}  {b_wins/decisive*100:>4.0f}%  {decisive:>4}  {p:>9.4f}  {sig:>5}")


def accuracy_significance(rows):
    """McNemar's test: paired comparison of each config vs no_prune."""
    print("\n" + "═"*60)
    print("  ACCURACY — McNemar's test vs no_prune baseline (paired)")
    print("═"*60)
    print(f"  {'Config':<35}  {'Acc':>5}  {'p-value':>9}  {'sig':>5}")
    print("  " + "-"*58)

    # Build per-sample correct/incorrect for each config
    by_config = defaultdict(dict)
    for r in rows:
        c = r.get("correct", "").strip()
        if c in ("0", "1"):
            by_config[r["config_tag"]][int(r["sample_idx"])] = int(c)

    baseline = by_config.get("no_prune", {})
    if not baseline:
        print("  No no_prune rows found.")
        return

    for tag in sorted(by_config.keys()):
        if tag == "no_prune":
            continue
        comp = by_config[tag]
        shared = sorted(set(baseline.keys()) & set(comp.keys()))
        if len(shared) < 5:
            continue
        b = [baseline[i] for i in shared]
        c = [comp[i]     for i in shared]
        # McNemar: count discordant pairs
        b_right_c_wrong = sum(1 for x, y in zip(b, c) if x == 1 and y == 0)
        b_wrong_c_right = sum(1 for x, y in zip(b, c) if x == 0 and y == 1)
        n_disc = b_right_c_wrong + b_wrong_c_right
        acc = sum(c) / len(c)
        if n_disc < 2:
            print(f"  {tag:<35}  {acc:>4.2f}  {'n/a (no discordant pairs)':>9}")
            continue
        # Exact binomial on discordant pairs
        p = stats.binomtest(b_wrong_c_right, n_disc, p=0.5).pvalue
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print(f"  {tag:<35}  {acc:>4.2f}  {p:>9.4f}  {sig:>5}")


def score_significance(rows, score_col, label):
    """Wilcoxon signed-rank test vs no_prune for a score column."""
    print(f"\n{'═'*60}")
    print(f"  {label} — Wilcoxon signed-rank test vs no_prune")
    print(f"{'═'*60}")
    print(f"  {'Config':<35}  {'Mean':>6}  {'p-value':>9}  {'sig':>5}")
    print("  " + "-"*58)

    by_config = defaultdict(dict)
    for r in rows:
        v = r.get(score_col, "").strip()
        if v.lstrip("-").isdigit():
            by_config[r["config_tag"]][int(r["sample_idx"])] = int(v)

    baseline = by_config.get("no_prune", {})
    if not baseline:
        print("  No no_prune rows found.")
        return

    for tag in sorted(by_config.keys()):
        if tag == "no_prune":
            m = np.mean(list(baseline.values()))
            print(f"  {'no_prune':<35}  {m:>6.2f}  {'(baseline)':>9}")
            continue
        comp = by_config[tag]
        shared = sorted(set(baseline.keys()) & set(comp.keys()))
        if len(shared) < 5:
            continue
        b = [baseline[i] for i in shared]
        c = [comp[i]     for i in shared]
        m = np.mean(c)
        diffs = [x - y for x, y in zip(c, b)]
        if all(d == 0 for d in diffs):
            print(f"  {tag:<35}  {m:>6.2f}  {'identical':>9}")
            continue
        _, p = stats.wilcoxon(c, b, alternative="two-sided", zero_method="wilcox")
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print(f"  {tag:<35}  {m:>6.2f}  {p:>9.4f}  {sig:>5}")


def confidence_intervals(rows):
    """Wilson 95% CI for accuracy per config."""
    print("\n" + "═"*65)
    print("  ACCURACY — 95% Wilson Confidence Intervals")
    print("═"*65)
    print(f"  {'Config':<35}  {'Acc':>5}  {'95% CI':>15}  {'n':>4}")
    print("  " + "-"*63)

    configs = defaultdict(list)
    for r in rows:
        c = r.get("correct", "").strip()
        if c in ("0", "1"):
            configs[r["config_tag"]].append(int(c))

    for tag in sorted(configs.keys()):
        vals = configs[tag]
        n = len(vals)
        k = sum(vals)
        p_hat = k / n
        # Wilson interval
        z = 1.96
        denom = 1 + z**2 / n
        centre = (p_hat + z**2 / (2*n)) / denom
        margin = (z * (p_hat*(1-p_hat)/n + z**2/(4*n**2))**0.5) / denom
        lo, hi = max(0, centre - margin), min(1, centre + margin)
        print(f"  {tag:<35}  {p_hat:>4.2f}  [{lo:.2f}, {hi:.2f}]  {n:>4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",       type=Path,
                    default=HERE / "results" / "sara-experiments.csv")
    ap.add_argument("--reasoning",  type=Path,
                    default=HERE / "results" / "eval_reasoning.csv")
    ap.add_argument("--coverage",   type=Path,
                    default=HERE / "results" / "eval_coverage.csv")
    ap.add_argument("--preference", type=Path,
                    default=HERE / "results" / "eval_preference.csv")
    args = ap.parse_args()

    base_rows       = load(args.base)
    reasoning_rows  = load(args.reasoning)
    coverage_rows   = load(args.coverage)
    preference_rows = load(args.preference)

    if preference_rows:
        preference_significance(preference_rows)
    if base_rows:
        accuracy_significance(base_rows)
        confidence_intervals(base_rows)
    if reasoning_rows:
        score_significance(reasoning_rows, "llm_judge_score",
                           "LLM JUDGE SCORE (0–5)")
    if coverage_rows:
        score_significance(coverage_rows,  "llm_coverage_final",
                           "COVERAGE FINAL SCORE (0–5)")

    print()


if __name__ == "__main__":
    main()
