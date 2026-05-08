#!/usr/bin/env bash
# ============================================================
#  run_captioning.sh — Caption task experiment (parallel)
#
#  Runs 4 conditions in parallel (2 at a time to avoid OOM).
#  Each condition gets its own process and log file.
#
#  Hypothesis: text pruning ≈ random (generic prompt),
#  gaze pruning outperforms both (scene-specific signal).
#
#  Run from sara-experiments/:
#    bash run_captioning.sh
# ============================================================

set -e
cd "$(dirname "$0")"

CAPTION_CSV="results/caption-experiments.csv"
N=90
SEED=42
FRAMES=8
BEST_LAYER=10

mkdir -p results logs

echo "=================================================="
echo "  Caption experiment — N=$N, $FRAMES frames"
echo "  Running 2 conditions at a time in parallel"
echo "=================================================="

# ── Batch 1: no_prune + random ────────────────────────────
echo "  [Batch 1] no_prune + random_r0.5 ..."

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --task caption \
  --config-tag caption_no_prune \
  --results-csv $CAPTION_CSV \
  > logs/caption_no_prune.log 2>&1 &
PID1=$!

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --task caption \
  --prune-random --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag caption_random_r0.5 \
  --results-csv $CAPTION_CSV \
  > logs/caption_random.log 2>&1 &
PID2=$!

wait $PID1 && echo "  no_prune done" || echo "  no_prune FAILED — check logs/caption_no_prune.log"
wait $PID2 && echo "  random done"   || echo "  random FAILED — check logs/caption_random.log"

# ── Batch 2: text + gaze ──────────────────────────────────
echo "  [Batch 2] text_r0.5 + gaze_r0.5 ..."

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --task caption \
  --prune-text --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag caption_text_r0.5 \
  --results-csv $CAPTION_CSV \
  > logs/caption_text.log 2>&1 &
PID3=$!

python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --task caption \
  --prune-gaze --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag caption_gaze_r0.5 \
  --results-csv $CAPTION_CSV \
  > logs/caption_gaze.log 2>&1 &
PID4=$!

wait $PID3 && echo "  text done" || echo "  text FAILED — check logs/caption_text.log"
wait $PID4 && echo "  gaze done" || echo "  gaze FAILED — check logs/caption_gaze.log"

echo ""
echo "=================================================="
echo "  Scoring with BERTScore (RoBERTa)"
echo "=================================================="

python score_captions_sara.py --csv $CAPTION_CSV

echo ""
echo "All done. Results → $CAPTION_CSV"
