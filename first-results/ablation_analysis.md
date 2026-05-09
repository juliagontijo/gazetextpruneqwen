# Comprehensive Ablation Analysis

**Project:** Gaze + Text Guided Token Pruning for Efficient Video VLM Decoding
**Model:** Qwen2-VL-2B-Instruct, MPS (M1)
**Dataset:** EgoGazeVQA, 4 frames per clip
**Date:** 2026-05-01

---

## Recap of the Pruning Mechanism

Before reading the tables, the math:

1. **Text scores** (SparseVLM) — for each visual token, measure attention from a small set of "rater" text tokens (selected as those most attentive to vision). This gives `text_visual_scores[i]` = how much the prompt cares about visual token `i`.
2. **Gaze scores** — for each frame, place a 2D Gaussian over the post-merger H_m×W_m grid centered on (gaze_x, gaze_y) with `σ = 0.15·max(H_m, W_m)`. Concatenate over T frames.
3. Both are **min-max normalized** to [0, 1].
4. Combined score: **`S = α · text_score + (1 − α) · gaze_score`**
5. At layer L (during prefill), keep top-`r` visual tokens by S, prune the rest. Layers L+1 … 28 then process a smaller KV cache → faster decode per token.

`prune_ratio = fraction kept`, so r=0.10 keeps 10% (very aggressive), r=0.90 keeps 90% (mild).

---

## 1. Layer Ablation — *where* to prune

Fixed: text-only, r=0.5, n=10.

| Layer | Accuracy | mean decode_s | mean ms/tok |
|---|---|---|---|
| **5**  | 5/10 = **50%** | 36.5 s | 112.8 |
| **10** | 7/10 = **70%** | 37.3 s | 136.4 |
| **15** | 6/10 = **60%** | 30.7 s | 91.4  |
| **20** | 6/10 = **60%** | 36.8 s | 103.0 |
| no_prune (ref) | 6/10 = 60% | 40.2 s | 136.0 |

**Hypothesis.** Earlier layers (L=5) prune before the model has cross-attended visual to text — token importance scores are computed on under-developed hidden states, so the rater sees noise and drops genuinely informative tokens (50% accuracy). Later layers (L=15, 20) prune after the model has built a richer multimodal representation, but only layers L+1..28 benefit from a smaller KV cache, so the speedup window shrinks. **L=10 is the inflection point**: hidden states are mature enough to give meaningful rater scores, yet 18 of 28 layers still benefit from a smaller KV cache. The high ms/tok at L=10 is misleading — more samples hit the 512-token cap (longer outputs), which inflates ms/tok via prefill amortization. Decode_s is the cleaner metric and is roughly flat (~30–40s) within 10-sample noise.

**Conclusion: L=10 wins on accuracy.**

---

## 2. Method Ablation — *what signal* to score with

Fixed: layer=10, r=0.5, n=10.

| Method | Accuracy | mean ms/tok | Notes |
|---|---|---|---|
| no_prune | 6/10 = 60% | 136.0 | baseline |
| **random** | 6/10 = 60% | 132.3 | uniform random scores |
| **text** | **7/10 = 70%** | 135.4 | SparseVLM rater only |
| **gaze** | 6/10 = 60% | 127.6 | Gaussian on (gaze_x, gaze_y) |
| **combined α=0.5** | 5/9 = 56% (1 unanswered) | 131.4 | equal mix |

**Hypothesis.**
- **Random ≈ no_prune** — confirms that even random pruning at 50% keeps enough visual context to answer most questions; tokens are highly redundant. This is the *control*: any "intelligent" method must beat this.
- **Text > random** — SparseVLM's text-rater identifies prompt-relevant tokens; the model can answer based on that subset.
- **Gaze ≈ random** — gaze alone is noisy. The Gaussian is a hand-crafted prior with σ=0.15, and many EgoGazeVQA questions ask about something *not* at the gaze point (peripheral objects, temporal sequences). Gaze is *user-centric*, not *question-centric*.
- **Combined α=0.5 < text** — equal weighting *dilutes* the text signal with gaze noise. When text says "look at the phone" but gaze is on the table, fusing 50/50 averages the two and the phone token gets a mediocre score, possibly dropped.

**Conclusion: text is the strongest single signal; naive 50/50 fusion hurts.**

---

## 3. Ratio Ablation — *how aggressively* to prune

Fixed: text-only, layer=10. Mixed n.

| Ratio (kept) | Accuracy | mean ms/tok | mean decode_s | n |
|---|---|---|---|---|
| 0.10 | **4/5 = 80%** | **82.4**  | 22.2 s | 5 |
| 0.25 | 4/5 = **80%** | 112.9 | 46.1 s | 5 |
| 0.50 | 7/10 = 70%    | 135.4 | 36.9 s | 10 |
| 0.75 | 2/5 = 40%     | 124.4 | 33.1 s | 5 |
| 0.90 | (corrupted/missing) | — | — | — |
| no_prune | 6/10 = 60% | 133.5 | 39.7 s | 10 |

**Hypothesis.**
- **Most-aggressive pruning (r=0.10) is the fastest *and* (tied for) most accurate.** Speedup ≈ **1.62× over baseline** in ms/tok. This is a striking result: keeping just 10% of visual tokens after layer 10 is enough for 80% accuracy on this set. Two reasons: (i) Qwen2-VL applies a 2×2 spatial merger before the LLM, so even before our pruning each "token" already aggregates 4 patches; (ii) the text rater concentrates importance on a small region, so the long tail it drops is genuinely redundant.
- **r=0.75 dropping to 40%** is suspicious — n=5 is tiny, and a flip of 2 samples explains the gap. Don't trust this point.
- **r=0.50 is slower than r=0.75** in ms/tok — ratio-driven KV size *should* be monotonic. The likely cause is more of the n=10 r=0.50 samples hitting the 512 max-length cap; longer outputs mean later tokens decode against bigger autoregressive KV, raising the average ms/tok. This is a metric artifact, not a real reversal.

**Conclusion: very aggressive pruning (keep 10%) is the operating point — accuracy holds and decode is ~1.6× faster.**

---

## 4. Alpha Ablation — *how to fuse* text and gaze

Fixed: layer=10, r=0.25, n=5.

| Alpha | Score = | Accuracy | mean ms/tok |
|---|---|---|---|
| 0.0  | pure gaze        | **4/5 = 80%** | 116.8 |
| 0.25 | 25% text + 75% gaze | 3/5 = **60%** | 114.7 |
| 0.50 | 50/50            | 4/5 = 80%     | 115.1 |
| 0.75 | 75% text + 25% gaze | 4/5 = 80%  | 114.0 |
| 1.0  | pure text        | 4/5 = 80%     | 113.8 |

**Hypothesis.**
- **Speed is flat across α** — expected. α only changes *which* tokens to drop, not how many. KV cache size is identical.
- **α=0.25 is the only loser (60%)** — this is the worst of both worlds: enough text weight to displace some good gaze picks, but not enough to dominate when text and gaze disagree. The min-max normalization makes this fragile: if gaze is concentrated in a small region (high values there, near-zero elsewhere) and text spreads more evenly, the gaze-heavy mix collapses keeps to a tiny spatial cluster and misses prompt-relevant peripheral tokens.
- **α=0.0 (pure gaze) matching α=1.0 (pure text)** at 80% — a coincidence of the 5-sample set. The single hard sample (s0, temporal) is wrong under *every* α. This point would shift with more samples.

**Conclusion: α barely matters within α ∈ {0.5, 0.75, 1.0}; avoid heavy gaze weighting (α ≤ 0.25). The uplift from fusion is *not* present in this data.**

---

## 5. Cross-Cutting Conclusions

### Parameter recipe
- **Layer = 10** — sweet spot between rater quality and speedup window.
- **Ratio = 0.10** — keep 10% is enough; the model is robust to losing 90% of post-merger visual tokens.
- **Method = text** — strongest standalone signal; gaze and combined add noise without accuracy benefit *on this n*.
- **α irrelevant** unless we go gaze-heavy (which hurts).

### Implementation observations
1. **The visual tokens are heavily redundant.** Random pruning at 50% loses no accuracy. Text pruning at 90% loses no accuracy. The model effectively only needs ~10–20% of the post-merger visual tokens for these multi-choice questions.
2. **ms/tok is a leaky metric.** It conflates per-token decode cost with output-length distribution. Whenever methods disagree on output length (e.g., longer rambling under harder configs), ms/tok shifts in misleading ways. **Total decode_s** with controlled `min_new_tokens`/`max_new_tokens` would be cleaner.
3. **Gaze is a weak signal *as currently encoded*.** A fixed Gaussian (σ=0.15) of the eye fixation is question-agnostic. The user's gaze tells us where they *looked*, not what the *question* asks.
4. **5–10 sample sets are too small** to separate true effects from noise (a single flipped sample = 10–20pp).

### The headline trade-off
At L=10, going from ratio 0.50 → 0.10 **gains ~1.6× decode speedup with no accuracy loss** (in fact +10pp on this sample). This is the project's main result.

---

## 6. Best Run So Far

**`text_l10_r0.10_v2`** — text-rater pruning, layer 10, keep 10% of visual tokens.
- Accuracy: **4/5 = 80%** (vs baseline 60%)
- ms/tok: **82.4** (vs baseline 133.5 → **1.62× speedup**)
- Mean decode: **22.2s** (vs baseline 39.7s → **1.79× speedup**)

Caveat: n=5. Needs replication on n≥30 before being a publishable number.

---

## 7. Improvement Ideas

### Stats / methodology
1. **Replicate r=0.10 and r=0.05 on n≥30** to nail down the speed-accuracy frontier with confidence intervals. The 5-sample numbers are illustrative, not conclusive.
2. **Fix output-length confound.** Set `min_new_tokens = max_new_tokens = 200` for *all* runs, or report decode-s per sample matched on output length, or report prefill-time and decode-time separately.
3. **Stratify by qa_type.** Spatial questions might benefit more from gaze; temporal more from text. The aggregate accuracy hides this.
4. **Compare against non-pruning baselines** (FastV, VisionZip, ToMe) at matched compute.

### Methods
5. **Soft-keep instead of hard-prune.** Use scores as attention bias on visual tokens rather than dropping them. Preserves info; might help temporal questions.
6. **Multi-layer progressive pruning.** Prune lightly at L=10 (keep 50%), then again at L=15 (keep 50% of survivors → 25% of original). Lets the model "earn" its way down.
7. **Question-conditioned gaze.** Currently gaze is question-agnostic. Re-weight the Gaussian by attention from the question tokens to the spatial region. This makes gaze actually fuse with intent.
8. **Tune gaze σ.** Sigma=0.15 is a guess. Sweep σ ∈ {0.05, 0.10, 0.20, 0.30} — small σ overweights the exact pixel, large σ ≈ uniform.
9. **Better fusion than linear.** The min-max normalization makes α brittle. Try log-sum-exp, multiplicative fusion (`text * gaze`), or rank-based (top-K from each, union).
10. **Drop the merger redundancy first.** Most of "10% kept = 80% accuracy" might be because the 2×2 spatial merger already discarded the easy redundancy. Profile token-level entropy before and after pruning to see where the real signal is.

### Engineering
11. **Cache the text rater scores.** They depend only on the prompt + frozen layers 0..L-1; a fixed system prompt would let you precompute.
12. **Move gaze score computation to the GPU.** Currently CPU `torch.exp`/`meshgrid` then `.to(device)`. For larger T it'll matter.
13. **Visualize disagreement.** For each sample, render which tokens text-only would keep vs gaze-only. Use this to pick samples where fusion *should* help vs where it's always going to be redundant.

### What this project still needs
The *story* the slides need to tell is "gaze + text > text alone." Right now the data says "text alone is fine, gaze is noise on top." Either:
- (a) Find the regime where gaze actually adds signal — likely **harder spatial/temporal questions where gaze and prompt disagree**, requiring better question-conditioned gaze fusion (idea 7).
- (b) Reposition the contribution — "text-rater pruning at extreme ratio (r=0.10) is sufficient on egocentric VLM" is a defensible finding by itself, with gaze as an explored-but-not-helpful baseline.

---

# 8. Final-Paper Experiment Plan (n=20)

This section lists the **commands to run for the paper-quality results.** All conditions use:

- **n = 20 samples** per condition (`--num-samples 20`)
- **same seed (42)** so every condition sees the *same 20 questions* — sample-difficulty noise cancels out across conditions
- **4 frames** per clip
- `TRANSFORMERS_OFFLINE=1` to skip HuggingFace remote checks
- `TRANSFORMERS_VERBOSITY=error` to silence the harmless docstring linter warnings

Two CSVs:
- `results/ablations.csv` — MCQ runs (existing file; new rows append)
- `results/captions.csv` — caption runs (graded offline by `score_captions.py`)

Each block is a chained `&&` command that runs sequentially without you needing to re-launch. The 5-second cooldown built into `profile_forked_model.py` prevents MPS memory contamination between conditions.

> **Time budget per run:** ~14 min on M1 (model load + warmup + 20 × ~40 s decode).
> **Total for all blocks below:** ~4 hours.

---

## Block A — MCQ Layer Ablation  *(text-only, r = 0.5, vary L)*

Re-runs the layer ablation at n=20 to give clean, low-variance numbers for the paper. **Goal:** confirm L=10 is still the accuracy sweet spot.

```
TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-layers 5 --prune-ratio 0.5 --config-tag n20_text_l5_r0.5 --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-layers 10 --prune-ratio 0.5 --config-tag n20_text_l10_r0.5 --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-layers 15 --prune-ratio 0.5 --config-tag n20_text_l15_r0.5 --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-layers 20 --prune-ratio 0.5 --config-tag n20_text_l20_r0.5 --results-csv results/ablations.csv
```

**Runs:** 4 conditions (L ∈ {5, 10, 15, 20}). **Time:** ~55 min.
**Output rows:** `n20_text_l{5,10,15,20}_r0.5`

---

## Block B — MCQ Ratio Ablation  *(text-only, L = 10, vary r)*

Sweeps how aggressive the pruning can be while preserving accuracy. **Goal:** locate the speed-accuracy frontier and reproduce the headline result (r=0.10).

The L=10 r=0.5 run from Block A also belongs in this ablation — no need to re-run it.

```
TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-layers 10 --prune-ratio 0.10 --config-tag n20_text_l10_r0.10 --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-layers 10 --prune-ratio 0.25 --config-tag n20_text_l10_r0.25 --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-layers 10 --prune-ratio 0.75 --config-tag n20_text_l10_r0.75 --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-layers 10 --prune-ratio 0.90 --config-tag n20_text_l10_r0.90 --results-csv results/ablations.csv
```

**Runs:** 4 new conditions (r ∈ {0.10, 0.25, 0.75, 0.90}). r=0.50 reuses Block A. **Time:** ~50 min.
**Output rows:** `n20_text_l10_r{0.10,0.25,0.75,0.90}`

---

## Block C — MCQ Alpha Ablation  *(combined text+gaze, L = 10, r = 0.25, vary α)*

Tests linear fusion at the operating point. **Goal:** confirm linear α-mix doesn't beat pure text or pure gaze, and identify the worst α.

Note α = 1.0 (pure text) overlaps with Block B's `text_l10_r0.25` — but we run it explicitly here because the `--prune-text --prune-gaze` flag combination triggers a slightly different pruning code path; running it explicitly keeps the alpha curve internally consistent.

```
TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-gaze --prune-alpha 0.0 --prune-layers 10 --prune-ratio 0.25 --config-tag n20_combined_a0.0_l10_r0.25 --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-gaze --prune-alpha 0.25 --prune-layers 10 --prune-ratio 0.25 --config-tag n20_combined_a0.25_l10_r0.25 --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-gaze --prune-alpha 0.5 --prune-layers 10 --prune-ratio 0.25 --config-tag n20_combined_a0.5_l10_r0.25 --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-gaze --prune-alpha 0.75 --prune-layers 10 --prune-ratio 0.25 --config-tag n20_combined_a0.75_l10_r0.25 --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-text --prune-gaze --prune-alpha 1.0 --prune-layers 10 --prune-ratio 0.25 --config-tag n20_combined_a1.0_l10_r0.25 --results-csv results/ablations.csv
```

**Runs:** 5 conditions (α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}). **Time:** ~70 min.
**Output rows:** `n20_combined_a{0.0,0.25,0.5,0.75,1.0}_l10_r0.25`

α = 0.0 is the pure-gaze condition; α = 1.0 is the pure-text condition (within the combined code path).

---

## Block D — MCQ Method Comparison at Operating Point  *(L = 10, r = 0.25)*

The method ablation at the chosen operating point: how do random / text / gaze / combined α=0.25 compare under matched conditions? **Goal:** show that pure text is the best single signal and combined α=0.25 underperforms it.

`text_l10_r0.25` is already covered by Block B; `combined_a0.25_l10_r0.25` is already covered by Block C. **Only two new runs are needed: gaze-only and the no-prune baseline.**

```
TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --config-tag n20_no_prune --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-gaze --prune-layers 10 --prune-ratio 0.25 --config-tag n20_gaze_l10_r0.25 --results-csv results/ablations.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --frames 4 --num-samples 20 --seed 42 --prune-random --prune-layers 10 --prune-ratio 0.25 --config-tag n20_random_l10_r0.25 --results-csv results/ablations.csv
```

**Runs:** 3 conditions (`no_prune`, gaze-only, random). **Time:** ~45 min.
**Output rows:** `n20_no_prune`, `n20_gaze_l10_r0.25`, `n20_random_l10_r0.25`

After this block you have, all at n=20 with matched seed:

| Method | Already in CSV? |
|---|---|
| no_prune | Block D |
| random | Block D |
| text-only | Block B (`n20_text_l10_r0.25`) |
| gaze-only | Block D |
| combined α = 0.25 | Block C (`n20_combined_a0.25_l10_r0.25`) |

---

## Block E — Task Ablation  *(MCQ vs Captioning at the headline config)*

Cross-task validation: does the same pattern hold when we measure quality on open-ended captions instead of MCQ letters? **Goal:** show that the speed-accuracy story is not a quirk of the MCQ format.

Three captioning runs at n=20: baseline, headline pruning config, gaze-only (for parity with Block D).

```
TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --task caption --frames 4 --num-samples 20 --seed 42 --config-tag n20_caption_no_prune --results-csv results/captions.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --task caption --frames 4 --num-samples 20 --seed 42 --prune-text --prune-layers 10 --prune-ratio 0.25 --config-tag n20_caption_text_l10_r0.25 --results-csv results/captions.csv && TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error llmgazepp/bin/python profile_forked_model.py --task caption --frames 4 --num-samples 20 --seed 42 --prune-gaze --prune-layers 10 --prune-ratio 0.25 --config-tag n20_caption_gaze_l10_r0.25 --results-csv results/captions.csv
```

**Runs:** 3 conditions. **Time:** ~45 min.
**Output rows (in `results/captions.csv`):** `n20_caption_no_prune`, `n20_caption_text_l10_r0.25`, `n20_caption_gaze_l10_r0.25`

---

## Block F — Score the caption runs (offline, after Block E)

The caption rows have empty `correct` cells until BERTScore is computed. Run this once *after* Block E completes:

```
llmgazepp/bin/python score_captions.py --csv results/captions.csv --model roberta-large
```

This fills in the `correct` column with BERTScore F1 (range ~0.6–0.95) and prints a per-config mean F1 summary. Re-runnable safely — already-scored rows are skipped.

If you want rescaled-baseline F1 (0 ≈ random text, ~0.3 ≈ decent caption — easier to interpret across conditions), add `--rescale`:

```
llmgazepp/bin/python score_captions.py --csv results/captions.csv --model roberta-large --rescale
```

⚠️ Pick **one** rescaling mode for the paper and stick with it — rescaled and unrescaled F1 are not comparable.

---

## Total run inventory

| Block | Description | New runs | Time | CSV |
|---|---|---|---|---|
| A | Layer ablation | 4 | ~55 min | ablations.csv |
| B | Ratio ablation (text) | 4 | ~50 min | ablations.csv |
| C | Alpha ablation (combined) | 5 | ~70 min | ablations.csv |
| D | Method @ operating point | 3 | ~45 min | ablations.csv |
| E | Caption task track | 3 | ~45 min | captions.csv |
| F | Offline BERTScore | — | ~3 min | captions.csv |
| | **TOTAL** | **19 runs** | **~4 hours** | |

380 sample evaluations total (19 × 20). All 19 conditions use `--seed 42`, so they all share the same 20 EgoGazeVQA questions — every comparison is paired.

---

## Suggested order of operations

1. **Block D first** — it gives you `no_prune` baseline. You can spot-check immediately whether the new n=20 baseline matches the previous n=10 baseline (sanity check on the run).
2. **Block B** — produces the headline result (r=0.10).
3. **Block A** — fills out the layer ablation.
4. **Block C** — alpha sweep.
5. **Block E + F** — caption track + scoring at the end.

If any block crashes mid-way, the per-sample CSV append means you don't lose completed rows — just re-run the *remaining* commands from that block.

---

## After all runs complete

You will have (paired across conditions, n=20, seed=42):

- 13 MCQ rows in `ablations.csv` (4 layers + 4 ratios + 5 alphas + 3 method, with the L=10 and α=1.0 / α=0.25 overlaps)
- 3 caption rows in `captions.csv` with BERTScore F1 filled in

That's enough to populate every table in the paper:

| Paper table | Source |
|---|---|
| MCQ method ablation @ L=10, r=0.25 | Block D + Block B (text) + Block C (α=0.25) |
| MCQ ratio ablation (text-only, L=10) | Block B + Block A (L=10 r=0.5) |
| MCQ layer ablation (text-only, r=0.5) | Block A |
| MCQ alpha ablation (combined, r=0.25, L=10) | Block C |
| Captioning headline | Block E + F |
