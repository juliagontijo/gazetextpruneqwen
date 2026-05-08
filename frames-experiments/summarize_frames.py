"""
summarize_frames.py — Print comparison tables for the iso-cost frames experiment.

Reads results/frames-experiments.csv plus the three eval CSVs and prints:
  1. Accuracy + throughput per condition
  2. LLM judge reasoning scores
  3. Coverage breakdown
  4. Preference results (vs 4f_no_prune baseline)

The key question answered here:
  Does 8f_text_l10_r0.5 beat 4f_no_prune on reasoning/coverage
  at the same (or lower) decode cost?

Usage:
    python summarize_frames.py
"""

import csv
from collections import defaultdict
from pathlib import Path

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


def fmt(v, d=2):
    return f"{v:.{d}f}" if v is not None else "--"


# Display order that makes the iso-cost argument visible at a glance
CONDITION_ORDER = [
    "4f_no_prune",
    "8f_no_prune",
    "8f_text_l10_r0.5",
    "8f_gaze_l10_r0.5",
]

CONDITION_LABEL = {
    "4f_no_prune":       "4f  no-prune  (baseline)",
    "8f_no_prune":       "8f  no-prune  (upper bound)",
    "8f_text_l10_r0.5":  "8f  text-prune r=0.5  (iso-cost)",
    "8f_gaze_l10_r0.5":  "8f  gaze-prune r=0.5  (iso-cost)",
}


def accuracy_table(rows):
    configs = {}
    for r in rows:
        tag = r["config_tag"]
        if tag not in configs:
            configs[tag] = defaultdict(list)
        correct = r.get("correct", "")
        if correct not in ("0", "1"):
            continue
        configs[tag]["correct"].append(int(correct))
        configs[tag][r.get("qa_type", "")].append(int(correct))
        mstok = r.get("decode_ms_per_token", "")
        if mstok:
            try:
                configs[tag]["mstok"].append(float(mstok))
            except ValueError:
                pass

    baseline_mstok = None
    W = 36
    print("\n" + "═" * 88)
    print("  ACCURACY & THROUGHPUT")
    print("═" * 88)
    print(f"  {'Condition':{W}}  {'Acc':>5}  {'Causal':>6}  {'Spatial':>7}  {'Temporal':>8}  {'ms/tok':>7}  Speedup")
    print("  " + "-" * 86)

    for tag in CONDITION_ORDER:
        if tag not in configs:
            continue
        d = configs[tag]
        acc      = mean(d["correct"])
        causal   = mean(d.get("causal", []))
        spatial  = mean(d.get("spatial", []))
        temporal = mean(d.get("temporal", []))
        mstok    = mean(d.get("mstok", []))
        label    = CONDITION_LABEL.get(tag, tag)
        if tag == "4f_no_prune":
            baseline_mstok = mstok
        speedup = f"{baseline_mstok/mstok:.2f}×" if (baseline_mstok and mstok) else "--"
        print(f"  {label:{W}}  {fmt(acc):>5}  {fmt(causal):>6}  {fmt(spatial):>7}  {fmt(temporal):>8}  {fmt(mstok,1):>7}  {speedup}")


def reasoning_table(rows):
    configs = defaultdict(list)
    for r in rows:
        s = r.get("llm_judge_score", "").strip()
        if s.lstrip("-").isdigit():
            configs[r["config_tag"]].append(int(s))

    if not configs:
        print("\n  No reasoning scores found.")
        return

    W = 36
    print("\n" + "═" * 62)
    print("  LLM JUDGE REASONING SCORE (0–5)")
    print("═" * 62)
    print(f"  {'Condition':{W}}  {'Mean':>6}  {'Std':>6}  {'n':>4}")
    print("  " + "-" * 60)

    for tag in CONDITION_ORDER:
        if tag not in configs:
            continue
        vals = configs[tag]
        m = mean(vals)
        std = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0
        label = CONDITION_LABEL.get(tag, tag)
        print(f"  {label:{W}}  {fmt(m):>6}  {fmt(std):>6}  {len(vals):>4}")


def coverage_table(rows):
    configs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tag = r["config_tag"]
        for col, key in [
            ("llm_coverage_scene",    "scene"),
            ("llm_coverage_relevance","rel"),
            ("llm_coverage_logic",    "logic"),
            ("llm_coverage_final",    "final"),
        ]:
            v = r.get(col, "").strip()
            if v.lstrip("-").isdigit():
                configs[tag][key].append(int(v))

    if not configs:
        print("\n  No coverage scores found.")
        return

    W = 36
    print("\n" + "═" * 77)
    print("  COVERAGE SCORES (0–5 per dimension)")
    print("═" * 77)
    print(f"  {'Condition':{W}}  {'Scene':>6}  {'Relev':>6}  {'Logic':>6}  {'Final':>6}  {'n':>4}")
    print("  " + "-" * 75)

    for tag in CONDITION_ORDER:
        if tag not in configs:
            continue
        d = configs[tag]
        label = CONDITION_LABEL.get(tag, tag)
        print(f"  {label:{W}}  {fmt(mean(d['scene'])):>6}  {fmt(mean(d['rel'])):>6}  "
              f"{fmt(mean(d['logic'])):>6}  {fmt(mean(d['final'])):>6}  {len(d['final']):>4}")


def preference_table(rows):
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

    W = 36
    print("\n" + "═" * 72)
    print("  PREFERENCE vs 4f_no_prune  (A=baseline wins, B=condition wins)")
    print("═" * 72)
    print(f"  {'Condition':{W}}  {'A%':>6}  {'B%':>6}  {'Tie%':>6}  {'n':>4}")
    print("  " + "-" * 70)

    for tag in CONDITION_ORDER:
        if tag not in configs:
            continue
        d = configs[tag]
        n = d["total"]
        if n == 0:
            continue
        label = CONDITION_LABEL.get(tag, tag)
        print(f"  {label:{W}}  {d['A']/n*100:>5.0f}%  {d['B']/n*100:>5.0f}%  "
              f"{d['Tie']/n*100:>5.0f}%  {n:>4}")


def iso_cost_check(base_rows):
    """Print a note on whether 8f+prune is actually iso-cost vs 4f baseline."""
    by_tag = defaultdict(list)
    for r in base_rows:
        mstok = r.get("decode_ms_per_token", "")
        if mstok:
            try:
                by_tag[r["config_tag"]].append(float(mstok))
            except ValueError:
                pass

    base = mean(by_tag.get("4f_no_prune", []))
    text8 = mean(by_tag.get("8f_text_l10_r0.5", []))
    gaze8 = mean(by_tag.get("8f_gaze_l10_r0.5", []))

    if not (base and (text8 or gaze8)):
        return

    print("\n" + "═" * 60)
    print("  ISO-COST CHECK (decode ms/tok)")
    print("═" * 60)
    print(f"  4f no-prune baseline : {base:.1f} ms/tok")
    if text8:
        ratio = text8 / base
        verdict = "✓ iso-cost" if 0.85 <= ratio <= 1.25 else ("faster" if ratio < 0.85 else "slower")
        print(f"  8f text-prune r=0.5  : {text8:.1f} ms/tok  ({ratio:.2f}×  → {verdict})")
    if gaze8:
        ratio = gaze8 / base
        verdict = "✓ iso-cost" if 0.85 <= ratio <= 1.25 else ("faster" if ratio < 0.85 else "slower")
        print(f"  8f gaze-prune r=0.5  : {gaze8:.1f} ms/tok  ({ratio:.2f}×  → {verdict})")


def main():
    base_rows  = load(HERE / "results" / "frames-experiments.csv")
    reas_rows  = load(HERE / "results" / "eval_reasoning.csv")
    cov_rows   = load(HERE / "results" / "eval_coverage.csv")
    pref_rows  = load(HERE / "results" / "eval_preference.csv")

    if base_rows:
        accuracy_table(base_rows)
        iso_cost_check(base_rows)
    if reas_rows:
        reasoning_table(reas_rows)
    if cov_rows:
        coverage_table(cov_rows)
    if pref_rows:
        preference_table(pref_rows)

    print()


if __name__ == "__main__":
    main()
