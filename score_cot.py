"""
Offline CoT coverage scoring for MCQ rows in ablations.csv.

For each MCQ row the model produces a chain-of-thought (CoT) before writing
"Answer: X". Coverage measures how well that reasoning covers the ground-truth
scene content (the Ego4D narration text for the clip).

Metric: BERTScore **recall** — for each semantic unit in the reference narration,
how well is it represented in the CoT? High recall = the model's reasoning
mentions the things that were actually happening in the scene.

Rows already scored (cot_coverage non-empty) are skipped, so it is safe to
re-run on a CSV that grows over time.

Usage:
    python score_cot.py --csv results/ablations.csv
    python score_cot.py --csv results/ablations.csv --config-tag text_l10_r0.5
    python score_cot.py --csv results/ablations.csv --rescale
"""
import argparse
import csv
from pathlib import Path

CSV_COLUMNS = [
    "config_tag", "sample_idx", "video_id", "qa_type",
    "prune_text", "prune_gaze", "prune_random", "prune_alpha", "prune_ratio", "prune_layer",
    "input_preprocessing_s", "vision_encoder_s", "decode_s", "tokens_generated",
    "decode_ms_per_token", "correct_answer", "predicted_answer", "correct",
    "cot_text", "narration_gt", "cot_coverage",
]

BERT_MODEL = "microsoft/deberta-xlarge-mnli"


def is_scorable(row: dict) -> bool:
    """MCQ row with a non-empty CoT and narration GT, not yet scored."""
    return (
        row.get("cot_coverage", "") == ""
        and len(row.get("cot_text", "").strip()) > 10
        and len(row.get("narration_gt", "").strip()) > 10
    )


def summarise(rows: list[dict], targets: list[dict], scores: list[float]) -> None:
    """Print mean coverage per config_tag."""
    by_tag: dict[str, list[float]] = {}
    for r, s in zip(targets, scores):
        by_tag.setdefault(r["config_tag"], []).append(s)

    print("\nCoT coverage (BERTScore recall) per config:")
    print(f"  {'config_tag':<45} {'mean recall':>11}  {'n':>4}")
    print(f"  {'-'*45} {'-'*11}  {'-'*4}")
    for tag, vals in sorted(by_tag.items()):
        m = sum(vals) / len(vals)
        print(f"  {tag:<45} {m:>11.4f}  {len(vals):>4}")

    # Also show breakdown by qa_type within each config (only if multiple types present)
    qa_types = sorted({r["qa_type"] for r in targets})
    if len(qa_types) > 1:
        print()
        for tag, vals in sorted(by_tag.items()):
            tag_rows = [(r, s) for r, s in zip(targets, scores) if r["config_tag"] == tag]
            for qt in qa_types:
                qt_vals = [s for r, s in tag_rows if r["qa_type"] == qt]
                if qt_vals:
                    print(f"  {tag:<35} {qt:<10} {sum(qt_vals)/len(qt_vals):.4f}  (n={len(qt_vals)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Path to ablations.csv")
    ap.add_argument("--config-tag", type=str, default=None,
                    help="Only score rows whose config_tag contains this substring.")
    ap.add_argument("--model", type=str, default=BERT_MODEL,
                    help="BERTScore backbone model.")
    ap.add_argument("--lang", type=str, default="en")
    ap.add_argument("--rescale", action="store_true",
                    help="Rescale with baseline (F1≈0 for random text).")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    targets = []
    for r in rows:
        if not is_scorable(r):
            continue
        if args.config_tag and args.config_tag not in r["config_tag"]:
            continue
        targets.append(r)

    if not targets:
        print("No scorable CoT rows found (already scored, empty CoT, or empty narration_gt).")
        return

    print(f"Scoring {len(targets)} CoT rows with BERTScore recall ({args.model}) ...")

    from bert_score import score as bert_score

    cands = [r["cot_text"]    for r in targets]
    refs  = [r["narration_gt"] for r in targets]

    _P, R, _F1 = bert_score(
        cands, refs,
        model_type=args.model,
        lang=args.lang,
        rescale_with_baseline=args.rescale,
        verbose=True,
    )

    recall_scores = R.tolist()

    # Write recall back into rows
    for r, recall in zip(targets, recall_scores):
        r["cot_coverage"] = f"{recall:.4f}"

    # Re-write the CSV, preserving all rows and filling any missing columns
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_COLUMNS})

    summarise(rows, targets, recall_scores)
    print(f"\nUpdated → {args.csv}")


if __name__ == "__main__":
    main()
