"""
Offline BERTScore for caption-task rows in ablations.csv.

Reads the per-sample CSV produced by profile_forked_model.py --task caption
(where `correct_answer` = ground-truth narration, `predicted_answer` = model
output, `correct` = empty), computes BERTScore F1 per row, fills the `correct`
column with that F1, and writes back.

Usage:
    pip install bert-score
    python score_captions.py --csv results/ablations.csv
    python score_captions.py --csv results/ablations.csv --config-tag caption_text_l10_r0.10

By default, only rows whose `correct` field is empty are scored (so it is safe
to re-run on a CSV that mixes MCQ and caption rows).
"""
import argparse
import csv
from pathlib import Path

CSV_COLUMNS = [
    "config_tag", "sample_idx", "video_id", "qa_type",
    "prune_text", "prune_gaze", "prune_random", "prune_alpha", "prune_ratio", "prune_layer",
    "input_preprocessing_s", "vision_encoder_s", "decode_s", "tokens_generated",
    "decode_ms_per_token", "correct_answer", "predicted_answer", "correct",
    "cot_text", "narration_gt", "cot_coverage", "reasoning_score", "reasoning_explanation",
]


def is_caption_row(row: dict) -> bool:
    """A caption row has empty `correct` AND a long `correct_answer` (narrations)."""
    return row.get("correct", "") == "" and len(row.get("correct_answer", "")) > 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--config-tag", type=str, default=None,
                    help="Only score rows whose config_tag contains this substring.")
    ap.add_argument("--model", type=str, default="microsoft/deberta-xlarge-mnli",
                    help="BERTScore backbone (default = best-correlation per BERTScore docs).")
    ap.add_argument("--lang", type=str, default="en")
    ap.add_argument("--rescale", action="store_true",
                    help="Rescale F1 with baseline (more interpretable, F1≈0 for random text).")
    args = ap.parse_args()

    # Read all rows
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    targets = []
    for r in rows:
        if not is_caption_row(r):
            continue
        if args.config_tag and args.config_tag not in r["config_tag"]:
            continue
        targets.append(r)

    if not targets:
        print("No caption rows to score (correct already filled or no caption rows).")
        return

    print(f"Scoring {len(targets)} caption rows with BERTScore "
          f"({args.model})...")

    # Lazy import — only needed if we actually score
    from bert_score import score

    cands = [r["predicted_answer"] for r in targets]
    refs  = [r["correct_answer"]   for r in targets]

    P, R, F1 = score(
        cands, refs,
        model_type=args.model,
        lang=args.lang,
        rescale_with_baseline=args.rescale,
        verbose=True,
    )

    # Write F1 back into the original rows (match by identity)
    f1s = F1.tolist()
    for r, f1 in zip(targets, f1s):
        r["correct"] = f"{f1:.4f}"

    # Re-write the CSV
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_COLUMNS})

    # Per-config summary
    by_tag: dict[str, list[float]] = {}
    for r, f1 in zip(targets, f1s):
        by_tag.setdefault(r["config_tag"], []).append(f1)

    print("\nBERTScore F1 per config:")
    for tag, vals in sorted(by_tag.items()):
        m = sum(vals) / len(vals)
        print(f"  {tag:40s}  mean F1 = {m:.4f}   (n={len(vals)})")

    print(f"\nUpdated → {args.csv}")


if __name__ == "__main__":
    main()
