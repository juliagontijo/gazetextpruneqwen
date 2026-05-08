# Gaze-Text Prune on Qwen2-VL

Forked Qwen2-VL profiling workspace for gaze/text pruning experiments on EgoGazeVQA.

This branch is a **snapshot of the forked-model path**, including:
- the forked Qwen model implementation in `modeling_qwen2_vl.py`
- the forked profiler in `profile_forked_model.py`
- frame preprocessing in `preprocess_frames.py`
- lightweight dataset metadata/narration files
- saved profiling logs and pruning visualizations

It is an **experimental branch**, not a polished release.

## What Is In This Repo

- `profile_forked_model.py`
  - loads the local forked `Qwen2VLForConditionalGeneration`
  - profiles `vision_encoder`, `decoder_only`, and `full_run` separately
  - logs timings, inputs, outputs, predicted answer, and pruning config to `profile_forked_log.md`
- `modeling_qwen2_vl.py`
  - local Qwen2-VL fork with in-model pruning arguments
- `preprocess_frames.py`
  - builds frame folders plus `gaze.json` files from EgoGazeVQA narrations
- `viz/`
  - saved pruning visualizations from forked-model runs
- `data/EgoGazeVQA_full/`
  - lightweight metadata files included in this branch:
    - `metadata.csv`
    - `ego4d.json`
    - `egoexo.json`
    - `egtea.json`
    - dataset README / `.gitattributes`

## What Is Not Included

The full `data/` directory on the local machine is about **25 GB**, so the complete video payload and generated frame cache are **not** pushed in this branch.

That means this branch includes:
- metadata
- narration/gaze JSON files

but not:
- the full downloaded dataset videos
- the large pre-extracted frame cache

## Environment Setup

This project was run in a local Python 3.11 virtual environment named `llmgazepp`.

### 1. Create the environment

```bash
python3 -m venv llmgazepp
source llmgazepp/bin/activate
python -m pip install --upgrade pip
```

### 2. Install Python dependencies

```bash
pip install torch transformers accelerate qwen-vl-utils opencv-python pillow matplotlib numpy datasets huggingface_hub
```

Notes:
- this workspace was run on Apple Silicon with `mps`
- `matplotlib` is only needed if you want saved visualizations
- `accelerate` is required because the model is loaded with `device_map`

## Dataset

Dataset source:
- [taiyi09/EgoGazeVQA on Hugging Face](https://huggingface.co/datasets/taiyi09/EgoGazeVQA)

Important:
- the dataset is **gated**
- you must log in to Hugging Face and accept the dataset conditions before downloading

### 1. Log in

```bash
huggingface-cli login
```

### 2. Download the lightweight metadata/narration files

These are already included in this branch, but this is how they were obtained:

```bash
huggingface-cli download taiyi09/EgoGazeVQA \
  metadata.csv ego4d.json egoexo.json egtea.json README.md .gitattributes \
  --repo-type dataset \
  --local-dir data/EgoGazeVQA_full
```

### 3. Download the videos

Download only the subsets you want. Example:

```bash
huggingface-cli download taiyi09/EgoGazeVQA ego4d \
  --repo-type dataset \
  --local-dir data/EgoGazeVQA_full
```

You can repeat that for:
- `egoexo`
- `egtea`

Expected local structure:

```text
data/EgoGazeVQA_full/
├── metadata.csv
├── ego4d.json
├── egoexo.json
├── egtea.json
├── ego4d/
├── egoexo/
└── egtea/
```

## Metadata Note

`metadata.csv` is included in this repo because it was needed to reproduce the runs, and it was not present in your earlier GitHub snapshot. This branch keeps the lightweight dataset metadata together with the profiling code so the sample selection and evaluation flow are reproducible.

## Frame Preprocessing

Before profiling, build the gaze-aligned frame cache.

Example for Ego4D:

```bash
python preprocess_frames.py --dataset ego4d --force
```

This creates frame folders plus `gaze.json` files under:

```text
data/EgoGazeVQA_full/frames/ego4d/
```

You can do the same for the other subsets:

```bash
python preprocess_frames.py --dataset egoexo --force
python preprocess_frames.py --dataset egtea --force
```

## Running The Forked Profiler

Baseline:

```bash
python profile_forked_model.py --frames 4
```

This:
1. loads the local forked model from `modeling_qwen2_vl.py`
2. loads one sample from `metadata.csv`
3. prefers preprocessed frames if they exist
4. runs a warmup generate
5. profiles:
   - `vision_encoder`
   - `decoder_only`
   - `full_run`
6. logs the result to `profile_forked_log.md`

## Pruning Commands

### 1. Gaze-only pruning

```bash
python profile_forked_model.py --frames 4 --prune-gaze --prune-layers 10 --prune-ratio 0.5
```

Meaning:
- use 4 sampled frames
- compute gaze scores from the preprocessed `gaze.json`
- prune inside the forked model at decoder layer `10`
- keep `50%` of visual tokens at that pruning layer

### 2. Text-only pruning

```bash
python profile_forked_model.py --frames 4 --prune-text --prune-layers 10 --prune-ratio 0.5
```

Meaning:
- use SparseVLM-style text-guided pruning
- prune at decoder layer `10`
- keep `50%` of visual tokens

### 3. Combined gaze + text pruning

```bash
python profile_forked_model.py --frames 4 --prune-gaze --prune-text --prune-layers 10 --prune-ratio 0.5 --prune-alpha 0.5
```

Meaning:
- enable both gaze and text pruning
- prune at decoder layer `10`
- keep `50%` of visual tokens
- fuse gaze and text scores with `prune_alpha=0.5`

Interpretation of `prune_alpha` in this branch:
- `0.0` means rely more on gaze
- `1.0` means rely more on text
- `0.5` gives equal weight

## Current Snapshot Status

This branch is meant to preserve the current experimental state.

What is working:
- baseline forked-model profiling run
- separate timing for `vision_encoder`, `decoder_only`, and `full_run`
- logging to `profile_forked_log.md`
- saved pruning visualizations under `viz/`

What is still experimental:
- in-model pruning with cached decode, especially text-guided and combined paths
- layer-specific attention-mask/cache bookkeeping after pruning

If a prune-enabled run fails, that is part of the current saved state of this branch.

## Outputs

- profiling log:
  - `profile_forked_log.md`
- pruning visualizations:
  - `viz/`

## Example Workflow

```bash
source llmgazepp/bin/activate

python preprocess_frames.py --dataset ego4d --force

python profile_forked_model.py --frames 4
python profile_forked_model.py --frames 4 --prune-gaze --prune-layers 10 --prune-ratio 0.5
python profile_forked_model.py --frames 4 --prune-text --prune-layers 10 --prune-ratio 0.5
python profile_forked_model.py --frames 4 --prune-gaze --prune-text --prune-layers 10 --prune-ratio 0.5 --prune-alpha 0.5
```
