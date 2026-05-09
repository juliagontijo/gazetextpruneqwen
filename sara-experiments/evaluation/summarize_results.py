"""
summarize_results.py — Print summary tables from evaluation CSVs.

Usage:
    python summarize_results.py
    python summarize_results.py --reasoning results/eval_reasoning.csv \
                                 --coverage  results/eval_coverage.csv \
                                 --preference results/eval_preference.csv \
                                 --base      results/sara-experiments.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).parent


def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping.")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def fmt(v, decimals=2):
    return f"{v:.{decimals}f}" if v is not None else "--"


def accuracy_table(rows: list[dict]):
    """MCQ accuracy and decode speed per config from sara-experiments.csv."""
    configs = {}
    for r in rows:
        tag = r["config_tag"]
        if tag not in configs:
            configs[tag] = defaultdict(list)
        correct = r.get("correct", "")
        if correct not in ("", "0", "1"):
            continue  # caption row or unanswered
        qt = r.get("qa_type", "")
        configs[tag]["correct"].append(int(correct) if correct else 0)
        configs[tag][qt].append(int(correct) if correct else 0)
        mstok = r.get("decode_ms_per_token", "")
        if mstok:
            try:
                configs[tag]["mstok"].append(float(mstok))
            except ValueError:
                pass

    col_w = 35
    print("\n" + "═"*85)
    print("  ACCURACY & THROUGHPUT")
    print("═"*85)
    print(f"  {'Config':{col_w}}  {'Acc':>6}  {'Causal':>7}  {'Spatial':>7}  {'Temporal':>8}  {'ms/tok':>7}")
    print("  " + "-"*83)

    baseline_mstok = None
    order = ["no_prune", "random_l10_r0.5", "gaze_l10_r0.5",
             "text_l10_r0.5", "combined_a0.5_l10_r0.5",
             "text_l10_r0.10", "text_l10_r0.25", "text_l10_r0.50",
             "text_l10_r0.75", "text_l10_r0.90"]
    shown = set()
    for tag in order + sorted(configs.keys()):
        if tag not in configs or tag in shown:
            continue
        shown.add(tag)
        d = configs[tag]
        acc     = mean(d["correct"])
        causal  = mean(d.get("causal", []))
        spatial = mean(d.get("spatial", []))
        temporal= mean(d.get("temporal", []))
        mstok   = mean(d.get("mstok", []))
        if tag == "no_prune":
            baseline_mstok = mstok
        speedup = f"{baseline_mstok/mstok:.2f}×" if (baseline_mstok and mstok) else "--"
        print(f"  {tag:{col_w}}  {fmt(acc):>6}  {fmt(causal):>7}  {fmt(spatial):>7}  {fmt(temporal):>8}  {fmt(mstok,1):>7}  ({speedup})")


def reasoning_table(rows: list[dict]):
    """LLM judge reasoning scores per config."""
    configs = defaultdict(list)
    for r in rows:
        s = r.get("llm_judge_score", "").strip()
        if s.lstrip("-").isdigit():
            configs[r["config_tag"]].append(int(s))

    if not configs:
        print("\n  No reasoning scores found.")
        return

    col_w = 35
    print("\n" + "═"*60)
    print("  LLM JUDGE REASONING SCORE (0–5)")
    print("═"*60)
    print(f"  {'Config':{col_w}}  {'Mean':>6}  {'Std':>6}  {'n':>4}")
    print("  " + "-"*58)
    for tag in sorted(configs.keys()):
        vals = configs[tag]
        m = mean(vals)
        std = (sum((v - m)**2 for v in vals) / len(vals))**0.5 if len(vals) > 1 else 0
        print(f"  {tag:{col_w}}  {fmt(m):>6}  {fmt(std):>6}  {len(vals):>4}")


def coverage_table(rows: list[dict]):
    """Coverage breakdown per config."""
    configs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tag = r["config_tag"]
        for col, key in [("llm_coverage_scene",     "scene"),
                         ("llm_coverage_relevance",  "rel"),
                         ("llm_coverage_logic",      "logic"),
                         ("llm_coverage_final",      "final")]:
            v = r.get(col, "").strip()
            if v.lstrip("-").isdigit():
                configs[tag][key].append(int(v))

    if not configs:
        print("\n  No coverage scores found.")
        return

    col_w = 35
    print("\n" + "═"*75)
    print("  COVERAGE SCORES (0–5 per dimension)")
    print("═"*75)
    print(f"  {'Config':{col_w}}  {'Scene':>6}  {'Relev':>6}  {'Logic':>6}  {'Final':>6}  {'n':>4}")
    print("  " + "-"*73)
    for tag in sorted(configs.keys()):
        d = configs[tag]
        print(f"  {tag:{col_w}}  {fmt(mean(d['scene'])):>6}  {fmt(mean(d['rel'])):>6}  "
              f"{fmt(mean(d['logic'])):>6}  {fmt(mean(d['final'])):>6}  {len(d['final']):>4}")


def preference_table(rows: list[dict]):
    """Preference (A=baseline wins, B=pruned wins, Tie) per config."""
    configs = defaultdict(lambda: {"A": 0, "B": 0, "Tie": 0, "total": 0})
    for r in rows:
        w = r.get("llm_pref_winner", "").strip()
        if w in ("A", "B", "Tie"):
            tag = r["config_tag"]
            configs[tag][w] += 1
            configs[tag]["total"] += 1

    if not configs:
        print("\n  No preference results found.")
        return

    col_w = 35
    print("\n" + "═"*70)
    print("  PREFERENCE vs NO_PRUNE BASELINE  (A=baseline wins, B=pruned wins)")
    print("═"*70)
    print(f"  {'Config':{col_w}}  {'A%':>6}  {'B%':>6}  {'Tie%':>6}  {'n':>4}")
    print("  " + "-"*68)
    for tag in sorted(configs.keys()):
        d = configs[tag]
        n = d["total"]
        if n == 0:
            continue
        print(f"  {tag:{col_w}}  {d['A']/n*100:>5.0f}%  {d['B']/n*100:>5.0f}%  "
              f"{d['Tie']/n*100:>5.0f}%  {n:>4}")


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

    if base_rows:
        accuracy_table(base_rows)
    if reasoning_rows:
        reasoning_table(reasoning_rows)
    if coverage_rows:
        coverage_table(coverage_rows)
    if preference_rows:
        preference_table(preference_rows)

    print()


if __name__ == "__main__":
    main()
