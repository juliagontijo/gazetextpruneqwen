#!/bin/bash
#SBATCH --job-name=caption_pruning
#SBATCH --output=logs/caption_%A_%a.out
#SBATCH --error=logs/caption_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-3

cd "$(dirname "$0")"
mkdir -p logs results

CAPTION_CSV="results/caption-experiments.csv"
FRAMES=8
BEST_LAYER=10
N=90
SEED=42

# One condition per array task
CONDITIONS=("no_prune" "random" "text" "gaze")
CONFIG_TAGS=("caption_no_prune" "caption_random_r0.5" "caption_text_r0.5" "caption_gaze_r0.5")

CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
TAG=${CONFIG_TAGS[$SLURM_ARRAY_TASK_ID]}

echo "Running condition: $TAG"

case $CONDITION in
  no_prune)
    python profile_sara.py \
      --frames $FRAMES --num-samples $N --seed $SEED \
      --task caption \
      --config-tag $TAG \
      --results-csv $CAPTION_CSV
    ;;
  random)
    python profile_sara.py \
      --frames $FRAMES --num-samples $N --seed $SEED \
      --task caption \
      --prune-random --prune-layers $BEST_LAYER --prune-ratio 0.5 \
      --config-tag $TAG \
      --results-csv $CAPTION_CSV
    ;;
  text)
    python profile_sara.py \
      --frames $FRAMES --num-samples $N --seed $SEED \
      --task caption \
      --prune-text --prune-layers $BEST_LAYER --prune-ratio 0.5 \
      --config-tag $TAG \
      --results-csv $CAPTION_CSV
    ;;
  gaze)
    python profile_sara.py \
      --frames $FRAMES --num-samples $N --seed $SEED \
      --task caption \
      --prune-gaze --prune-layers $BEST_LAYER --prune-ratio 0.5 \
      --config-tag $TAG \
      --results-csv $CAPTION_CSV
    ;;
esac

echo "Done: $TAG"
