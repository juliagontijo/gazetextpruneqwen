"""
merge_results.py — Merge no-prune-results.csv + sara_experiments1.csv
into a single sara-experiments.csv.

Takes all rows from no-prune-results.csv and all non-no_prune rows
from sara_experiments1.csv.

Usage:
    python merge_results.py
"""

import csv
from collections import Counter
from pathlib import Path

HERE            = Path(__file__).parent
NO_PRUNE_CSV    = HERE / "results" / "no-prune-results.csv"
MAIN_CSV        = HERE / "results" / "sara_experiments1.csv"
OUT_CSV         = HERE / "results" / "sara-experiments.csv"


def load(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    no_prune_rows = load(NO_PRUNE_CSV)
    main_rows     = load(MAIN_CSV)

    # Take only no_prune rows from the fresh file
    no_prune_kept = [r for r in no_prune_rows if r["config_tag"] == "no_prune"]

    # Take everything except no_prune from the main file
    other_kept    = [r for r in main_rows if r["config_tag"] != "no_prune"]

    merged = no_prune_kept + other_kept

    # Use the union of all columns, preserving order from main CSV
    all_cols = list(main_rows[0].keys()) if main_rows else list(no_prune_rows[0].keys())
    for col in no_prune_rows[0].keys():
        if col not in all_cols:
            all_cols.append(col)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for r in merged:
            writer.writerow({k: r.get(k, "") for k in all_cols})

    # Summary
    tag_counts = Counter(r["config_tag"] for r in merged)
    print(f"Written {len(merged)} rows → {OUT_CSV}")
    print("\nRows per config_tag:")
    for tag, n in sorted(tag_counts.items()):
        print(f"  {tag:45s}  {n}")


if __name__ == "__main__":
    main()
