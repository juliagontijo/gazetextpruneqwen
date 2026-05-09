# GazeTextPruneQwen

## Overview

This project implements gaze-based text pruning using Qwen2-VL models for efficient video captioning and question answering on the EgoGazeVQA dataset and serves as an exploration between its efficiency and the models reasoning quality tradeoffs. It focuses on pruning irrelevant text tokens based on gaze data to improve model performance and reduce computational overhead.

## Features

- Gaze-aware text pruning for Qwen2-VL models
- Video frame preprocessing and captioning
- Ablation studies and evaluation metrics
- Support for EgoGazeVQA dataset
- Batch processing for efficiency

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gazetextpruneqwen
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the required data in the `data/` directory (see Data section below).

## Data

The project uses the EgoGazeVQA dataset. Place the dataset files in `data/EgoGazeVQA_full/`:
- `ego4d.json`
- `egoexo.json`
- `metadata.csv`
- `README.md`

## Usage

### Preprocessing Frames
```bash
python preprocess_frames.py
```

### Precomputing Captions
```bash
python precompute_captions.py
```

### Running Experiments
Use the scripts in `sara-experiments/` for various experiments:
- `run_captioning.sh`: Run captioning experiments
- `run_evaluation.sh`: Evaluate model performance
- `run_sara_experiments.sh`: Run full experiment suite

### Ablations
Run ablation studies using:
```bash
bash run_ablations.sh
```

### Visualization
Visualize gaze data:
```bash
python visualize_gaze.py
```

## Experiments

### Batch Experiments
Located in `sara-experiments/batched-experiments/`:
- `batch_speedup_test.py`: Test batch processing speedup

### Frames Experiments
Located in `sara-experiments/frames-experiments/`:
- `profile_frames.py`: Profile frame processing
- `run_frames_experiments.sh`: Run frame-based experiments

### GRPO Training
Located in `sara-experiments/GRPO/`:
- `train_grpo.py`: Train with GRPO (Generative Reward-based Policy Optimization)

## Evaluation

Evaluation scripts are in `sara-experiments/evaluation/`:
- `evaluation.py`: Main evaluation script
- `llm_judge.py`: LLM-based judging
- `score_captions_sara.py`: Score captions
- `significance_tests.py`: Statistical significance tests
- `summarize_results.py`: Summarize results

Results are stored in `sara-experiments/sara-results/` and `first-results/`.

## Profiling

Profile model performance:
```bash
python profile_model.py
python profile_forked_model.py
```

## Contributing

Please follow standard contribution guidelines. Ensure all experiments are reproducible and documented.

## License

[Add license information here]
