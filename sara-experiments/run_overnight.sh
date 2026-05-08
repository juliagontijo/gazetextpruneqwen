#!/usr/bin/env bash
# ============================================================
#  run_overnight.sh — Extend method ablation to N=100 samples
#  and run all evaluations.
#
#  Resumes from existing results — already-done samples are
#  skipped automatically (resume logic in profile_sara.py).
#
#  Run from the sara-experiments/ directory:
#    bash run_overnight.sh
# ============================================================

set -e
cd "$(dirname "$0")"

RESULTS_CSV="results/sara-experiments.csv"
JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"
N=90           # 30 causal + 30 spatial + 30 temporal — all unique (clip, question) pairs
SEED=42
FRAMES=4
BEST_LAYER=10

mkdir -p results

echo "=================================================="
echo "  PHASE 1: Extend method ablation to N=$N samples"
echo "  (samples 0-29 already done, running 30-99 only)"
echo "=================================================="

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --config-tag no_prune \
  --results-csv $RESULTS_CSV

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-random --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag random_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-text --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag text_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-gaze --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag gaze_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-text --prune-gaze --prune-alpha 0.5 \
  --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag combined_a0.5_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-text --prune-layers $BEST_LAYER --prune-ratio 0.75 \
  --config-tag text_l${BEST_LAYER}_r0.75 \
  --results-csv $RESULTS_CSV

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-gaze --prune-layers $BEST_LAYER --prune-ratio 0.75 \
  --config-tag gaze_l${BEST_LAYER}_r0.75 \
  --results-csv $RESULTS_CSV

echo ""
echo "=================================================="
echo "  PHASE 2: Prompt ablation — gaze hint + scene oracle"
echo "=================================================="

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --gaze-prompt \
  --config-tag no_prune_gaze_hint \
  --results-csv $RESULTS_CSV

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --scene-prompt \
  --config-tag no_prune_scene_oracle \
  --results-csv $RESULTS_CSV

echo ""
echo "=================================================="
echo "  PHASE 4: Run all evaluations on new results"
echo "  (already-scored rows skipped automatically)"
echo "=================================================="

python evaluation.py \
  --csv $RESULTS_CSV \
  --output-csv results/eval_reasoning.csv \
  --mode reasoning \
  --judge-model $JUDGE_MODEL

python evaluation.py \
  --csv $RESULTS_CSV \
  --output-csv results/eval_coverage.csv \
  --mode coverage \
  --judge-model $JUDGE_MODEL

python evaluation.py \
  --csv $RESULTS_CSV \
  --output-csv results/eval_preference.csv \
  --mode preference \
  --judge-model $JUDGE_MODEL

echo ""
echo "=================================================="
echo "  PHASE 5: Summary + significance tests"
echo "=================================================="

python summarize_results.py
python significance_tests.py

echo ""
echo "All done. Check results/ for output CSVs."
