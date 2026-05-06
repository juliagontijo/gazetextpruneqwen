# Gaze + Text Visual Token Pruning for Qwen2-VL

This project investigates whether pruning visual tokens in a multimodal LLM — guided by human gaze and/or text relevance — can reduce decoding cost without hurting accuracy on egocentric video QA.

The model is **Qwen2-VL-2B-Instruct**, with a forked decoder that supports on-the-fly visual token pruning at specified transformer layers. The benchmark is **EgoGazeVQA**, an egocentric gaze-guided VQA dataset built on Ego4D, EgoExo, and EGTEA clips.

---

## How It Works

### The Core Idea

Qwen2-VL encodes video frames into a large number of visual tokens fed into the LLM decoder alongside text tokens. Most of those visual tokens are redundant for any given question. This project prunes the least useful ones at a chosen decoder layer, shrinking the KV cache and speeding up all subsequent decoding steps.

Pruning happens **once during the prefill pass**, before any tokens are generated. After pruning, the decoder operates on a shorter sequence for the rest of generation.

### Pruning Methods

Three signals can score visual tokens, alone or combined:

**Text-rater (`--prune-text`)**
Inspired by SparseVLM. After the prefill pass through the pruning layer, a subset of text tokens is selected as "raters" based on how much attention they pay to visual tokens. Each visual token is then scored by how strongly those raters attend to it — tokens that text pays attention to are kept.

**Gaze-guided (`--prune-gaze`)**
A 2D Gaussian heatmap is placed at the annotated gaze fixation point for each frame. Visual tokens closer to where the person was looking get higher scores. This is computed before any model forward pass and passed in as a `gaze_scores` tensor.

**Combined (`--prune-text --prune-gaze`)**
A weighted sum: `alpha * text_score + (1 - alpha) * gaze_score`. Alpha controls the balance (1 = pure text, 0 = pure gaze).

**Random (`--prune-random`)**
Baseline: tokens kept by uniform random scores. Reproducible per sample via a fixed seed. Mutually exclusive with the above two.

### Where Pruning Happens in the Model

The forked model is in [`modeling_qwen2_vl.py`](modeling_qwen2_vl.py). The key logic is in `Qwen2VLTextModel.forward()`:

1. The prefill runs normally up to the pruning layer
2. At the pruning layer, a standard forward pass runs first to get updated hidden states
3. Text-rater scores and/or gaze scores are computed from those hidden states
4. A boolean `visual_keep_mask` is built by top-K selection over the combined scores
5. Visual tokens failing the mask are removed from `hidden_states`, `position_ids`, and the attention mask
6. Remaining layers see only the kept tokens
7. During autoregressive decoding, the shorter KV cache is used — this is where the speedup accrues

---

## Dataset: EgoGazeVQA

**Source:** `taiyi09/EgoGazeVQA` on HuggingFace (gated — requires access request)

**Full size:** 1,761 QA rows across three source datasets:

| Dataset | Unique video IDs | Clips per video | QA rows |
|---------|-----------------|-----------------|---------|
| Ego4D   | 31              | ~10 (30 QA rows each, balanced 10×3) | 581 |
| EgoExo  | 161             | ~6 rows each    | 695 |
| EGTEA   | 82              | varies          | 485 |

Each clip has **3 QA rows** — one per question type:
- **Causal** — why did the person do X?
- **Spatial** — where is object X relative to Y?
- **Temporal** — what happened before/after X?

Each QA row is a 5-option multiple choice question (A–E).

**Gaze data:** Every clip has per-frame gaze fixation coordinates (`gaze_x`, `gaze_y` in [0,1]) and narration text sourced from `ego4d.json` / `egoexo.json`. These are written to per-clip `gaze.json` files during preprocessing.

### Why Ego4D for experiments

Each Ego4D video ID provides exactly 30 QA rows (10 per question type) — perfectly balanced. Downloading one video ID gives you a complete, balanced evaluation set. EgoExo only gives 6 rows per video ID, requiring many more downloads to reach the same coverage.

### Local Data Layout

```
data/EgoGazeVQA_full/
├── metadata.csv              # all 1761 QA rows
├── ego4d.json                # narrations + gaze for all Ego4D videos
├── egoexo.json               # narrations + gaze for all EgoExo videos
├── ego4d/                    # downloaded video clips
│   └── <video_id>/
│       └── <start>_<end>.mp4
└── frames/                   # extracted by preprocess_frames.py
    └── ego4d/
        └── <video_id>/
            └── <start>_<end>/
                ├── <frame_number>.jpg
                └── gaze.json
```

---

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate gazeprune
```

Or manually:

```bash
conda create -n gazeprune python=3.11 -y
pip install torch torchvision "transformers>=4.51.0" \
    huggingface_hub datasets qwen-vl-utils opencv-python \
    Pillow numpy matplotlib tqdm bert-score
```

### 2. Log in to HuggingFace

The dataset is gated. First request access at `huggingface.co/datasets/taiyi09/EgoGazeVQA`, then create a token at `huggingface.co/settings/tokens` with **"Read access to public gated repositories"** enabled.

```bash
hf auth login
```

---

## Data Pipeline

### Step 1 — Download videos

```bash
# Default: 3 Ego4D video IDs = 30 unique clips = 90 QA rows (30 per question type)
python download_ego4d.py

# Or specify different video IDs from metadata.csv
python download_ego4d.py --video-ids dafc891e-05f0-4734-88c6-1f818ac67a23 566ad4e5-1ce4-4679-9d19-ef63072c848c
```

Downloads go to `data/EgoGazeVQA_full/ego4d/<video_id>/<clip>.mp4`. Already-present clips are skipped so the script is safe to re-run.

### Step 2 — Extract frames and gaze

```bash
python preprocess_frames.py --dataset ego4d
```

For each clip this script:
1. Finds narration timestamps that fall within the clip's frame range
2. Extracts those exact frames as JPEGs
3. Writes a `gaze.json` with per-frame gaze coordinates and narration text

Output: `data/EgoGazeVQA_full/frames/ego4d/<video_id>/<clip_stem>/`

Use `--force` to re-extract even if already done.

---

## Running Experiments

### Single sample (debug / visualisation)

```bash
# Baseline — no pruning
python profile_forked_model.py

# Text-rater pruning at layer 10, keep 50% of visual tokens
python profile_forked_model.py --prune-text --prune-layers 10 --prune-ratio 0.5

# Gaze-guided pruning
python profile_forked_model.py --prune-gaze --prune-layers 10 --prune-ratio 0.5

# Combined (alpha=0.5 = equal weight between text and gaze)
python profile_forked_model.py --prune-text --prune-gaze --prune-alpha 0.5 \
    --prune-layers 10 --prune-ratio 0.5

# Caption task instead of MCQ
python profile_forked_model.py --task caption --prune-text --prune-layers 10 --prune-ratio 0.5
```

Single-sample runs save a pruning visualisation PNG to `viz/` and a markdown log entry to `profile_forked_log.md`.

### Full ablation sweep

```bash
bash run_ablations.sh
```

Runs in 4 sequential steps. After each step, inspect `results/ablations.csv`, update the `BEST_*` variable at the top of the next step block in `run_ablations.sh`, and continue.

| Step | Variable swept | Fixed | Measures |
|------|---------------|-------|----------|
| 1 — Layer | layer ∈ {5, 10, 20} | text-only, ratio=0.5 | accuracy + decode speed |
| 2 — Method | no-prune / random / text / gaze / combined | best layer, ratio=0.5 | accuracy |
| 3 — Ratio | ratio ∈ {0.10, 0.25, 0.50, 0.75, 0.90} | best layer + method | accuracy + decode speed |
| 4 — Alpha | α ∈ {0.0, 0.25, 0.50, 0.75, 1.0} | best layer + ratio | accuracy (combined only) |

All conditions use **N=30 samples, seed=42, 4 frames**. Results are appended to `results/ablations.csv` after each sample — safe to interrupt and resume.

### Key flags for `profile_forked_model.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--frames` | 4 | Frames sampled per clip |
| `--num-samples` | 1 | Number of clips to evaluate |
| `--seed` | 42 | Shuffle seed for sample selection |
| `--qa-types` | all | Filter: `causal`, `spatial`, `temporal` |
| `--task` | `mcq` | `mcq` (letter match) or `caption` (BERTScore, scored offline) |
| `--prune-text` | off | Enable text-rater pruning |
| `--prune-gaze` | off | Enable gaze-guided pruning |
| `--prune-random` | off | Random pruning baseline (mutually exclusive with text/gaze) |
| `--prune-layers` | [27] | Decoder layer index to prune at |
| `--prune-ratio` | 0.5 | Fraction of visual tokens to keep |
| `--prune-alpha` | 0.5 | Weight of text vs gaze (1 = pure text, 0 = pure gaze) |
| `--config-tag` | auto | Label written to the CSV for this condition |
| `--results-csv` | none | Path to append per-sample CSV rows |
| `--save-viz` | off | Save pruning overlay PNG for every sample |

---

## Output Files

### `results/ablations.csv`

One row per sample per condition. Key columns:

| Column | Description |
|--------|-------------|
| `config_tag` | Condition label, e.g. `text_l10_r0.5` |
| `prune_text/gaze/random` | Which method was active (0/1) |
| `prune_layer` | Layer index where pruning happened |
| `prune_ratio` | Fraction of tokens kept |
| `decode_s` | Total decode wall time in seconds |
| `decode_ms_per_token` | Decode speed |
| `correct` | 1/0 for MCQ; BERTScore F1 for caption (filled offline by `score_captions.py`) |
| `cot_text` | Chain-of-thought prefix extracted before "Answer:" |
| `narration_gt` | Ground-truth Ego4D narration for the clip |
| `cot_coverage` | BERTScore recall of CoT vs. narration (filled offline by `score_cot.py`) |
| `reasoning_score` | LLM-as-Judge rubric score 0–1 (filled offline by `score_reasoning.py`) |
| `reasoning_explanation` | Compact JSON with per-criterion `{"score": 0\|1, "reason": "..."}` |

### `viz/`

PNG files with a 3-panel pruning visualisation per sample:
- Row 1: original frames
- Row 2: frames overlaid with token score heatmap (RdYlGn, green = kept)
- Row 3: text-rater scores per token (bar chart, green bars = selected rater tokens)

### `profile_forked_log.md`

Markdown log with per-run timing table, model output, and pruning config for single-sample runs.

---

## Offline Scoring

All offline scoring scripts are safe to re-run on a CSV that has grown over time — they skip rows already scored.

### Caption scoring (`score_captions.py`)

Caption-task runs leave the `correct` column empty. Score them offline with BERTScore F1:

```bash
python score_captions.py --csv results/ablations.csv

# Score only a specific condition
python score_captions.py --csv results/ablations.csv --config-tag caption_text_l10_r0.5

# Rescale F1 so random text ≈ 0 (more interpretable)
python score_captions.py --csv results/ablations.csv --rescale
```

Uses `microsoft/deberta-xlarge-mnli` as the BERTScore backbone.

### CoT coverage (`score_cot.py`)

Measures how well the model's chain-of-thought covers the ground-truth scene content using BERTScore **recall** — for each semantic unit in the Ego4D narration, how well is it represented in the CoT? High recall means the model's reasoning mentions the things that were actually happening.

```bash
python score_cot.py --csv results/ablations.csv

# Score only a specific condition
python score_cot.py --csv results/ablations.csv --config-tag text_l10_r0.5
```

Writes the recall score to `cot_coverage`. Called automatically by `run_ablations.sh` after each step.

### Reasoning quality (`score_reasoning.py`)

LLM-as-Judge evaluation using `Qwen2.5-7B-Instruct` running locally — no API key needed, runs on the same GPU as the ablations.

```bash
python score_reasoning.py --csv results/ablations.csv

# Score only a specific condition
python score_reasoning.py --csv results/ablations.csv --config-tag text_l10_r0.5

# Dry-run to see what would be scored without running inference
python score_reasoning.py --csv results/ablations.csv --dry-run

# Use a smaller model if VRAM is tight
python score_reasoning.py --csv results/ablations.csv --model Qwen/Qwen2.5-3B-Instruct
```

Each MCQ row's CoT is evaluated on 5 binary criteria specific to the question type:

| QA type | Criteria |
|---------|----------|
| causal | action_observation · context_identification · causal_inference · logical_consistency · visual_grounding |
| spatial | object_identification · spatial_description · spatial_inference · logical_consistency · visual_grounding |
| temporal | event_identification · sequence_description · temporal_inference · logical_consistency · visual_grounding |

The mean of the 5 binary scores (0 or 1 each) is written to `reasoning_score` (range 0.0–1.0). A second column, `reasoning_explanation`, stores a compact JSON object with the per-criterion score **and** a one-sentence reason for that score — making it easy to audit why a particular CoT was rated as it was. The system prompt is cached across API calls to reduce cost. Progress is written to disk after every row so the script is safe to interrupt and resume.

**Hypothesis**: pruning degrades `reasoning_score` before it affects raw `correct` accuracy — making this metric a more sensitive early signal of quality degradation.

---

## Gaze Visualisation

Inspect gaze heatmaps overlaid on frames without running any model:

```bash
python visualize_gaze.py --frames 4 --sigma 0.1
python visualize_gaze.py --frames 4 --sigma 0.05 --save   # saves to gaze_vis.png
```

`sigma` controls the Gaussian width around the gaze fixation point (as a fraction of the token grid dimensions).

---

## Repository Structure

```
.
├── modeling_qwen2_vl.py      # forked Qwen2-VL decoder with visual token pruning
├── profile_forked_model.py   # main evaluation + profiling script
├── run_ablations.sh          # 4-step ablation sweep
├── download_ego4d.py         # download Ego4D clips from HuggingFace
├── preprocess_frames.py      # extract frames + gaze.json from raw clips
├── score_captions.py         # offline BERTScore for caption-task rows
├── score_cot.py              # offline BERTScore recall for CoT coverage
├── score_reasoning.py        # offline LLM-as-Judge reasoning quality scorer
├── visualize_gaze.py         # gaze heatmap visualisation tool
├── environment.yml           # conda environment spec
├── data/EgoGazeVQA_full/     # dataset (videos + frames + metadata)
├── results/                  # ablations.csv written here
└── viz/                      # pruning visualisation PNGs
```
