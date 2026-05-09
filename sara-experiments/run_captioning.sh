#!/usr/bin/env bash
# ============================================================
#  run_captioning.sh — Caption task experiment (sequential)
#
#  Runs 4 conditions one at a time to avoid GPU OOM.
#  caption_no_prune already done — skipped automatically
#  via resume logic in profile_sara.py.
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
echo "  Running conditions sequentially (one GPU)"
echo "=================================================="

echo "  [1/4] no_prune ..."
python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --task caption \
  --config-tag caption_no_prune \
  --results-csv $CAPTION_CSV

echo "  [2/4] random_r0.5 ..."
python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --task caption \
  --prune-random --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag caption_random_r0.5 \
  --results-csv $CAPTION_CSV

echo "  [3/4] text_r0.5 ..."
python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --task caption \
  --prune-text --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag caption_text_r0.5 \
  --results-csv $CAPTION_CSV

echo "  [4/4] gaze_r0.5 ..."
python profile_sara.py \
  --frames $FRAMES --num-samples $N --seed $SEED \
  --task caption \
  --prune-gaze --prune-layers $BEST_LAYER --prune-ratio 0.5 \
  --config-tag caption_gaze_r0.5 \
  --results-csv $CAPTION_CSV

echo ""
echo "=================================================="
echo "  Scoring with BERTScore (RoBERTa)"
echo "=================================================="

python score_captions_sara.py --csv $CAPTION_CSV

echo ""
echo "All done. Results -> $CAPTION_CSV"
