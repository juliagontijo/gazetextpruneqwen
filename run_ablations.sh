#!/usr/bin/env bash
# ============================================================
#  Ablation sweep for gaze-guided visual token pruning
#  Run from the Sara/ directory:  bash run_ablations.sh
#
#  Results are appended to results/ablations.csv.
#  Run each STEP block independently — Step N fixes the best
#  value found in Step N-1.
#
#  Edit BEST_LAYER / BEST_METHOD / BEST_RATIO below after each
#  step before running the next one.
# ============================================================

set -e
cd "$(dirname "$0")"

RESULTS_CSV="results/ablations.csv"
N=30          # samples per condition
SEED=42       # fixed shuffle seed — same 30 samples every condition
FRAMES=4

mkdir -p results

# ────────────────────────────────────────────────────────────
# STEP 1 — Layer ablation
#   Fixed: text-only, ratio=0.5
#   Vary: layer  (5 / 10 / 20)
#   Measures: accuracy + decode speed (layer affects both)
# ────────────────────────────────────────────────────────────
echo "=== STEP 1: Layer ablation ==="

python profile_forked_model.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-text --prune-layers 5  --prune-ratio 0.5 \
  --config-tag layer5_text_r0.5 \
  --results-csv $RESULTS_CSV

python profile_forked_model.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-text --prune-layers 10 --prune-ratio 0.5 \
  --config-tag layer10_text_r0.5 \
  --results-csv $RESULTS_CSV

python profile_forked_model.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-text --prune-layers 20 --prune-ratio 0.5 \
  --config-tag layer20_text_r0.5 \
  --results-csv $RESULTS_CSV

echo "--- Step 1 done. Check results/ablations.csv, pick best layer, set BEST_LAYER below ---"


# ────────────────────────────────────────────────────────────
# STEP 2 — Method ablation
#   Fixed: BEST_LAYER, ratio=0.5
#   Vary: no-prune / random / text / gaze / combined (alpha=0.5)
#   Measures: accuracy only (all pruned methods → same speedup)
# ────────────────────────────────────────────────────────────
BEST_LAYER=10   # <-- update after Step 1

echo "=== STEP 2: Method ablation (layer=$BEST_LAYER) ==="

# Baseline — no pruning
python profile_forked_model.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --config-tag no_prune \
  --results-csv $RESULTS_CSV

# Random
python profile_forked_model.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-random --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag random_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

# Text-only
python profile_forked_model.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-text --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag text_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

# Gaze-only
python profile_forked_model.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-gaze --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag gaze_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

# Combined text+gaze (alpha=0.5)
python profile_forked_model.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --prune-text --prune-gaze --prune-alpha 0.5 \
  --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag combined_a0.5_l${BEST_LAYER}_r0.5 \
  --results-csv $RESULTS_CSV

echo "--- Step 2 done. Pick best method, set BEST_METHOD below ---"


# ────────────────────────────────────────────────────────────
# STEP 3 — Ratio ablation  (speed-accuracy trade-off)
#   Fixed: BEST_LAYER, BEST_METHOD
#   Vary: ratio  (0.10 / 0.25 / 0.50 / 0.75 / 0.90)
#   Measures: accuracy + decode speed  ← main result plot
# ────────────────────────────────────────────────────────────
BEST_METHOD="text"   # <-- update after Step 2: text / gaze / combined_a0.5

echo "=== STEP 3: Ratio ablation (layer=$BEST_LAYER, method=$BEST_METHOD) ==="

for RATIO in 0.10 0.25 0.50 0.75 0.90; do
  # build the right flags for the chosen method
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

  python profile_forked_model.py \
    --frames $FRAMES --num-samples $N --seed $SEED \
    $FLAGS --prune-layers $BEST_LAYER --prune-ratio $RATIO \
    --config-tag ${BEST_METHOD}_l${BEST_LAYER}_r${RATIO} \
    --results-csv $RESULTS_CSV
done

echo "--- Step 3 done. Pick best ratio, set BEST_RATIO below ---"


# ────────────────────────────────────────────────────────────
# STEP 4 — Alpha ablation  (only if best method = combined)
#   Fixed: BEST_LAYER, ratio=BEST_RATIO
#   Vary: alpha  (0.0 / 0.25 / 0.50 / 0.75 / 1.0)
#   Measures: accuracy only (same ratio → same speedup)
# ────────────────────────────────────────────────────────────
BEST_RATIO=0.50   # <-- update after Step 3

echo "=== STEP 4: Alpha ablation (layer=$BEST_LAYER, ratio=$BEST_RATIO) ==="

if [ "$BEST_METHOD" = "combined_a0.5" ] || [ "$BEST_METHOD" = "combined" ]; then
  for ALPHA in 0.0 0.25 0.50 0.75 1.0; do
    python profile_forked_model.py \
      --frames $FRAMES --num-samples $N --seed $SEED \
      --prune-text --prune-gaze --prune-alpha $ALPHA \
      --prune-layers $BEST_LAYER --prune-ratio $BEST_RATIO \
      --config-tag combined_a${ALPHA}_l${BEST_LAYER}_r${BEST_RATIO} \
      --results-csv $RESULTS_CSV
  done
else
  echo "  Skipped: best method is '$BEST_METHOD', not combined."
fi

echo "=== All ablations done. Results in $RESULTS_CSV ==="
