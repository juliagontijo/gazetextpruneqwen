#!/usr/bin/env bash
# ============================================================
#  run_evaluation.sh — LLM-based CoT evaluation (all three modes)
#
#  Reads:  results/sara-experiments.csv
#  Writes: results/eval_reasoning.csv
#          results/eval_coverage.csv
#          results/eval_preference.csv
#
#  Run from the sara-experiments/ directory:
#    bash run_evaluation.sh
# ============================================================

set -e
cd "$(dirname "$0")"

INPUT_CSV="results/sara-experiments.csv"
JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"

mkdir -p results

echo "=== Mode 1: Reasoning quality (scene + question + CoT → 0–5 score) ==="
python evaluation.py \
  --csv "$INPUT_CSV" \
  --output-csv results/eval_reasoning.csv \
  --mode reasoning \
  --judge-model "$JUDGE_MODEL"

echo ""
echo "=== Mode 2: Coverage breakdown (scene grounding / relevance / logic) ==="
python evaluation.py \
  --csv "$INPUT_CSV" \
  --output-csv results/eval_coverage.csv \
  --mode coverage \
  --judge-model "$JUDGE_MODEL"

echo ""
echo "=== Mode 3: Preference vs no_prune baseline ==="
python evaluation.py \
  --csv "$INPUT_CSV" \
  --output-csv results/eval_preference.csv \
  --mode preference \
  --judge-model "$JUDGE_MODEL"

echo ""
echo "=== All evaluations done ==="
echo "  results/eval_reasoning.csv"
echo "  results/eval_coverage.csv"
echo "  results/eval_preference.csv"
