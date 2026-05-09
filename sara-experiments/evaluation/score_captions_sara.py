"""
score_captions_sara.py — Offline BERTScore for caption-task rows.

Same logic as the parent repo's score_captions.py but updated for the extended
CSV schema used by profile_sara.py (extra columns: cot_output, llm_judge_score,
llm_judge_reasoning).

Usage:
    pip install bert-score
    python score_captions_sara.py --csv results/sara_experiments.csv
    python score_captions_sara.py --csv results/sara_experiments.csv --config-tag caption_text_l10
"""
import argparse
import csv
from pathlib import Path

CSV_COLUMNS = [
    "config_tag", "sample_idx", "video_id", "qa_type",
    "prune_text", "prune_gaze", "prune_random", "prune_alpha", "prune_ratio", "prune_layer",
    "input_preprocessing_s", "vision_encoder_s", "decode_s", "tokens_generated",
    "decode_ms_per_token", "correct_answer", "predicted_answer", "correct",
    "cot_output", "llm_judge_score", "llm_judge_reasoning",
]


def is_caption_row(row: dict) -> bool:
    """Caption rows have an empty `correct` field and a long narration as ground truth."""
    return row.get("correct", "") == "" and len(row.get("correct_answer", "")) > 20


CAPTION_COLUMNS = [
    "config_tag", "sample_idx", "video_id", "qa_type",
    "prune_text", "prune_gaze", "prune_random", "prune_ratio", "prune_layer",
    "tokens_generated", "decode_ms_per_token",
    "correct_answer", "predicted_answer", "bertscore_f1",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",        type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, default=None,
                    help="Write clean caption results here (default: caption_scores.csv next to input).")
    ap.add_argument("--config-tag", type=str,  default=None,
                    help="Only score rows whose config_tag contains this substring.")
    ap.add_argument("--model",      type=str,  default="roberta-large",
                    help="BERTScore backbone.")
    ap.add_argument("--lang",       type=str,  default="en")
    ap.add_argument("--rescale",    action="store_true",
                    help="Rescale F1 with baseline (more interpretable).")
    args = ap.parse_args()
    if args.output_csv is None:
        args.output_csv = args.csv.parent / "caption_scores.csv"

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

    print(f"Scoring {len(targets)} caption rows with BERTScore ({args.model})...")

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

    f1s = F1.tolist()
    for r, f1 in zip(targets, f1s):
        r["bertscore_f1"] = f"{f1:.4f}"

    # Write clean output CSV with only caption-relevant columns
    out_cols = [c for c in CAPTION_COLUMNS if c in targets[0]]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for r in targets:
            writer.writerow({k: r.get(k, "") for k in out_cols})

    by_tag: dict[str, list[float]] = {}
    for r, f1 in zip(targets, f1s):
        by_tag.setdefault(r["config_tag"], []).append(f1)

    print("\nBERTScore F1 per config:")
    for tag in ["caption_no_prune", "caption_random_r0.5", "caption_text_r0.5", "caption_gaze_r0.5"]:
        vals = by_tag.get(tag, [])
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {tag:35s}  mean F1 = {m:.4f}   (n={len(vals)})")

    print(f"\nSaved → {args.output_csv}")


if __name__ == "__main__":
    main()
