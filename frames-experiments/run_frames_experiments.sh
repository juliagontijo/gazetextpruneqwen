#!/usr/bin/env bash
# ============================================================
#  run_frames_experiments.sh — Iso-cost frames experiment
#
#  Tests whether 8 frames + pruning r=0.5 gives better
#  reasoning than 4 frames with no pruning, at equal active
#  token count.
#
#  Conditions:
#    4f_no_prune          baseline (4 frames, no pruning)
#    8f_no_prune          upper bound (8 frames, no pruning, 2× slower)
#    8f_text_l10_r0.5     iso-cost with text-guided pruning
#    8f_gaze_l10_r0.5     iso-cost with gaze-guided pruning
#
#  All conditions write to results/frames-experiments.csv.
#  Already-completed rows are skipped automatically.
#
#  Run from frames-experiments/:
#    bash run_frames_experiments.sh
# ============================================================

set -e
cd "$(dirname "$0")"

RESULTS_CSV="results/frames-experiments.csv"
JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"
N=30
SEED=42

mkdir -p results

echo "=================================================="
echo "  PHASE 1: Inference — all 4 conditions, N=$N"
echo "=================================================="


python profile_frames.py \
  --condition 8f_no_prune \
  --num-samples $N --seed $SEED \
  --results-csv $RESULTS_CSV

python profile_frames.py \
  --condition 8f_text_l10_r0.5 \
  --num-samples $N --seed $SEED \
  --results-csv $RESULTS_CSV

python profile_frames.py \
  --condition 8f_gaze_l10_r0.5 \
  --num-samples $N --seed $SEED \
  --results-csv $RESULTS_CSV

echo ""
echo "=================================================="
echo "  PHASE 2: LLM-judge evaluation"
echo "=================================================="

SARA_DIR="$(dirname "$0")/../sara-experiments"

python "$SARA_DIR/evaluation.py" \
  --csv $RESULTS_CSV \
  --output-csv results/eval_reasoning.csv \
  --mode reasoning \
  --judge-model $JUDGE_MODEL

python "$SARA_DIR/evaluation.py" \
  --csv $RESULTS_CSV \
  --output-csv results/eval_coverage.csv \
  --mode coverage \
  --judge-model $JUDGE_MODEL

# Preference: compare each pruned / 8f condition vs 4f_no_prune baseline
python "$SARA_DIR/evaluation.py" \
  --csv $RESULTS_CSV \
  --output-csv results/eval_preference.csv \
  --mode preference \
  --judge-model $JUDGE_MODEL

echo ""
echo "=================================================="
echo "  PHASE 3: Summary tables"
echo "=================================================="

python summarize_frames.py

echo ""
echo "All done. Check results/ for output CSVs."
