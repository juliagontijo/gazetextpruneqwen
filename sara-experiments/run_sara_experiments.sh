#!/usr/bin/env bash
# ============================================================
#  Sara's ablation experiments — gaze-guided visual token pruning
#
#  What's different from run_ablations.sh:
#    - Layer ablation (Step 1) is SKIPPED — layer 10 is fixed as best.
#    - All conditions save the full CoT output (cot_output column).
#    - Results go to sara-experiments/results/sara_experiments.csv.
#    - After running, score with:
#        python llm_judge.py --csv results/sara_experiments.csv
#        python score_captions_sara.py --csv results/sara_experiments.csv
#
#  Run from the sara-experiments/ directory:
#    bash run_sara_experiments.sh
# ============================================================

set -e
cd "$(dirname "$0")"

RESULTS_CSV="results/sara_experiments.csv"
N=30
SEED=42
FRAMES=4
BEST_LAYER=10   # fixed — layer ablation already done

mkdir -p results


# ────────────────────────────────────────────────────────────
# STEP 2 — Method ablation
#   Fixed: layer=10, ratio=0.5
#   Vary:  no-prune / random / text / gaze / combined (alpha=0.5)
#   Measures: accuracy + CoT quality (LLM judge offline)
# ────────────────────────────────────────────────────────────
echo "=== STEP 2: Method ablation (layer=$BEST_LAYER) ==="

# Baseline — no pruning
python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --config-tag no_prune \
  --results-csv $RESULTS_CSV

# Random baseline
python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-random --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag random_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

# Text-only
python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-text --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag text_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

# Gaze-only
python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-gaze --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag gaze_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

# Combined text+gaze (alpha=0.5)
python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-text --prune-gaze --prune-alpha 0.5 \
  --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag combined_a0.5_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

echo "--- Step 2 done. Set BEST_METHOD below before continuing ---"


# ────────────────────────────────────────────────────────────
# STEP 3 — Ratio ablation  (speed-accuracy trade-off)
#   Fixed: layer=10, BEST_METHOD
#   Vary:  ratio (0.10 / 0.25 / 0.50 / 0.75 / 0.90)
#   Measures: accuracy + decode speed + CoT quality
# ────────────────────────────────────────────────────────────
BEST_METHOD="text"   # <-- update after Step 2: text / gaze / combined_a0.5

echo "=== STEP 3: Ratio ablation (layer=$BEST_LAYER, method=$BEST_METHOD) ==="

for RATIO in 0.10 0.25 0.50 0.75 0.90; do
  case $BEST_METHOD in
    text)
      FLAGS="--prune-text" ;;
    gaze)
      FLAGS="--prune-gaze" ;;
    combined_a0.5)
      FLAGS="--prune-text --prune-gaze --prune-alpha 0.5" ;;
    *)
      echo "Unknown BEST_METHOD=$BEST_METHOD"; exit 1 ;;
  esac

  python profile_sara.py \
    --frames $FRAMES --num-samples $N --seed $SEED \
    $FLAGS --prune-layers $BEST_LAYER --prune-ratio $RATIO \
    --config-tag ${BEST_METHOD}_l${BEST_LAYER}_r${RATIO} \
    --results-csv $RESULTS_CSV
done

echo "--- Step 3 done. Set BEST_RATIO below before continuing ---"


# ────────────────────────────────────────────────────────────
# STEP 4 — Alpha ablation  (only if best method = combined)
#   Fixed: layer=10, BEST_RATIO
#   Vary:  alpha (0.0 / 0.25 / 0.50 / 0.75 / 1.0)
#   Measures: CoT quality per qa_type (text vs gaze emphasis)
# ────────────────────────────────────────────────────────────
BEST_RATIO=0.50   # <-- update after Step 3

echo "=== STEP 4: Alpha ablation (layer=$BEST_LAYER, ratio=$BEST_RATIO) ==="

if [ "$BEST_METHOD" = "combined_a0.5" ] || [ "$BEST_METHOD" = "combined" ]; then
  for ALPHA in 0.0 0.25 0.50 0.75 1.0; do
    python profile_sara.py \
      --frames $FRAMES --num-samples $N --seed $SEED \
      --prune-text --prune-gaze --prune-alpha $ALPHA \
      --prune-layers $BEST_LAYER --prune-ratio $BEST_RATIO \
      --config-tag combined_a${ALPHA}_l${BEST_LAYER}_r${BEST_RATIO} \
      --results-csv $RESULTS_CSV
  done
else
  echo "  Skipped: best method is '$BEST_METHOD', not combined."
fi

echo ""
echo "=== All conditions done. Results in $RESULTS_CSV ==="
echo ""
echo "Next steps:"
echo "  1. Score CoT quality:"
echo "       python llm_judge.py --csv $RESULTS_CSV"
echo "  2. Score caption task (BERTScore):"
echo "       python score_captions_sara.py --csv $RESULTS_CSV"
