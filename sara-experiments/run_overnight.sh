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

echo ""
echo "=================================================="
echo "  PHASE 4: eval and start from coverage and preference "
echo "=================================================="


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
