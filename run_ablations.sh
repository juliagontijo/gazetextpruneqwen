#!/usr/bin/env bash
# ============================================================
#  Ablation sweep — gaze-guided visual token pruning
#
#  Usage:
#    bash run_ablations.sh              # run all 4 steps
#    bash run_ablations.sh 1            # run only step 1
#    bash run_ablations.sh 2 3          # run steps 2 and 3
#
#  After step 1: set BEST_LAYER below and re-run from step 2.
#  After step 2: set BEST_METHOD below and re-run from step 3.
#  After step 3: set BEST_RATIO below and re-run step 4 (if combined).
#
#  Results:   results/ablations.csv
#  CoT score: results/cot_coverage.csv  (same CSV, cot_coverage column)
# ============================================================

set -e
cd "$(dirname "$0")"

RESULTS_CSV="results/ablations.csv"
N=30        # samples per condition (covers all 30 clips = 90 QA rows / 3 types)
SEED=42
FRAMES=4
PY="python"

mkdir -p results

# ── Best values — update after each step ─────────────────────
BEST_LAYER=10          # update after step 1
BEST_METHOD="text"     # update after step 2: text | gaze | combined_a0.5
BEST_RATIO=0.50        # update after step 3

# ── Which steps to run (default: all) ────────────────────────
STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
  STEPS=(1 2 3 4)
fi

run_condition() {
  # run_condition <config_tag> [extra flags...]
  local TAG="$1"; shift
  echo "  → $TAG"
  $PY profile_forked_model.py \
    --frames $FRAMES --num-samples $N --seed $SEED \
    --task mcq \
    --results-csv $RESULTS_CSV \
    --config-tag "$TAG" \
    "$@"
}

score_cot() {
  echo ""
  echo "  Scoring CoT coverage (BERTScore recall)..."
  $PY score_cot.py --csv $RESULTS_CSV
}

score_reasoning() {
  echo ""
  echo "  Scoring reasoning quality (LLM-as-Judge)..."
  $PY score_reasoning.py --csv $RESULTS_CSV
}

# ────────────────────────────────────────────────────────────
# STEP 1 — Layer ablation
#   Fixed: text-only, ratio=0.5
#   Vary:  layer ∈ {5, 10, 20}
#   Goal:  find the layer where pruning hurts accuracy least
#          while still providing a speedup
# ────────────────────────────────────────────────────────────
if [[ " ${STEPS[*]} " =~ " 1 " ]]; then
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  STEP 1 — Layer ablation  (text-only, ratio=0.5)"
  echo "════════════════════════════════════════════════════════════"

  run_condition "layer5_text_r0.5"  --prune-text --prune-layers 5  --prune-ratio 0.5
  run_condition "layer10_text_r0.5" --prune-text --prune-layers 10 --prune-ratio 0.5
  run_condition "layer20_text_r0.5" --prune-text --prune-layers 20 --prune-ratio 0.5

  score_cot
  score_reasoning

  echo ""
  echo "  Step 1 done. Set BEST_LAYER in this script, then run:"
  echo "    bash run_ablations.sh 2"
fi

# ────────────────────────────────────────────────────────────
# STEP 2 — Method ablation
#   Fixed: BEST_LAYER, ratio=0.5
#   Vary:  no-prune / random / text / gaze / combined (α=0.5)
#   Goal:  find which signal (or combo) keeps accuracy highest
#   Note:  baseline (no_prune) is only run here once
# ────────────────────────────────────────────────────────────
if [[ " ${STEPS[*]} " =~ " 2 " ]]; then
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  STEP 2 — Method ablation  (layer=$BEST_LAYER, ratio=0.5)"
  echo "════════════════════════════════════════════════════════════"

  run_condition "no_prune"
  run_condition "random_l${BEST_LAYER}_r0.5"           --prune-random --prune-layers $BEST_LAYER --prune-ratio 0.5
  run_condition "text_l${BEST_LAYER}_r0.5"             --prune-text   --prune-layers $BEST_LAYER --prune-ratio 0.5
  run_condition "gaze_l${BEST_LAYER}_r0.5"             --prune-gaze   --prune-layers $BEST_LAYER --prune-ratio 0.5
  run_condition "combined_a0.5_l${BEST_LAYER}_r0.5"   --prune-text --prune-gaze --prune-alpha 0.5 \
                                                        --prune-layers $BEST_LAYER --prune-ratio 0.5

  score_cot
  score_reasoning

  echo ""
  echo "  Step 2 done. Set BEST_METHOD in this script, then run:"
  echo "    bash run_ablations.sh 3"
fi

# ────────────────────────────────────────────────────────────
# STEP 3 — Ratio ablation  (main speed–accuracy trade-off)
#   Fixed: BEST_LAYER, BEST_METHOD
#   Vary:  ratio ∈ {0.10, 0.25, 0.50, 0.75, 0.90}
#   Goal:  plot the accuracy vs. decode-speed Pareto curve
# ────────────────────────────────────────────────────────────
if [[ " ${STEPS[*]} " =~ " 3 " ]]; then
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  STEP 3 — Ratio ablation  (layer=$BEST_LAYER, method=$BEST_METHOD)"
  echo "════════════════════════════════════════════════════════════"

  for RATIO in 0.10 0.25 0.50 0.75 0.90; do
    case $BEST_METHOD in
      text)             FLAGS="--prune-text" ;;
      gaze)             FLAGS="--prune-gaze" ;;
      combined_a0.5)    FLAGS="--prune-text --prune-gaze --prune-alpha 0.5" ;;
      *)  echo "Unknown BEST_METHOD=$BEST_METHOD"; exit 1 ;;
    esac

    run_condition "${BEST_METHOD}_l${BEST_LAYER}_r${RATIO}" \
      $FLAGS --prune-layers $BEST_LAYER --prune-ratio $RATIO
  done

  score_cot
  score_reasoning

  echo ""
  echo "  Step 3 done. Set BEST_RATIO in this script, then run:"
  echo "    bash run_ablations.sh 4   (only needed if BEST_METHOD=combined_a0.5)"
fi

# ────────────────────────────────────────────────────────────
# STEP 4 — Alpha ablation  (only when best method = combined)
#   Fixed: BEST_LAYER, BEST_RATIO
#   Vary:  α ∈ {0.0, 0.25, 0.50, 0.75, 1.0}
#   Goal:  find optimal text vs. gaze balance
# ────────────────────────────────────────────────────────────
if [[ " ${STEPS[*]} " =~ " 4 " ]]; then
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  STEP 4 — Alpha ablation  (layer=$BEST_LAYER, ratio=$BEST_RATIO)"
  echo "════════════════════════════════════════════════════════════"

  if [[ "$BEST_METHOD" == combined* ]]; then
    for ALPHA in 0.0 0.25 0.50 0.75 1.0; do
      run_condition "combined_a${ALPHA}_l${BEST_LAYER}_r${BEST_RATIO}" \
        --prune-text --prune-gaze --prune-alpha $ALPHA \
        --prune-layers $BEST_LAYER --prune-ratio $BEST_RATIO
    done
    score_cot
    score_reasoning
  else
    echo "  Skipped — BEST_METHOD='$BEST_METHOD' is not combined."
  fi

  echo ""
  echo "  Step 4 done."
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All requested steps complete."
echo "  Results → $RESULTS_CSV"
echo "  (CoT coverage is in the cot_coverage column)"
echo "════════════════════════════════════════════════════════════"
