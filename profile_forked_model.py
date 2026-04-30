import argparse
import contextlib
import csv
import datetime
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATA_DIR = Path(__file__).parent / "data" / "EgoGazeVQA_full"
METADATA_CSV = DATA_DIR / "metadata.csv"
FRAMES_DIR = DATA_DIR / "frames"
FORKED_MODEL_PATH = Path(__file__).parent / "modeling_qwen2_vl.py"
LOG_FILE = Path(__file__).parent / "profile_forked_log.md"
VIZ_DIR = Path(__file__).parent / "viz"


def load_forked_qwen2vl_class():
    module_name = "transformers.models.qwen2_vl.modeling_qwen2_vl"
    spec = importlib.util.spec_from_file_location(module_name, FORKED_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "transformers.models.qwen2_vl"
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Qwen2VLForConditionalGeneration


Qwen2VLForConditionalGeneration = load_forked_qwen2vl_class()


def mps_sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def mps_allocated_mb() -> float:
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0


@contextlib.contextmanager
def timed_block(label: str, results: dict):
    mps_sync()
    mem_before = mps_allocated_mb()
    t0 = time.perf_counter()
    yield
    mps_sync()
    elapsed = time.perf_counter() - t0
    mem_after = mps_allocated_mb()
    results[label] = {
        "time_s": elapsed,
        "mem_delta_mb": mem_after - mem_before,
        "mem_peak_mb": mem_after,
    }


def extract_frames(video_path: str, n_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1

    indices = [int(i * (total - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    return frames


def load_sample() -> tuple[str, dict, str]:
    with open(METADATA_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        local_path = DATA_DIR / row["file_name"]
        if not local_path.exists():
            continue

        options = row["answer_options"].split("|")
        options_str = "\n".join(o.strip() for o in options)

        output_log = f"""
Sample  file:      {row['file_name']}
        qa_type:   {row['qa_type']}
        question:  {row['question']}
        options:   {options_str}
        answer:    {row['correct_answer']}
"""

        print(f"Sample  file:      {row['file_name']}")
        print(f"        qa_type:   {row['qa_type']}")
        print(f"        question:  {row['question']}")
        print(f"        options:   {options_str}")
        print(f"        answer:    {row['correct_answer']}")
        return str(local_path), row, output_log

    raise FileNotFoundError("No locally available clips match metadata.csv entries.")


def load_preprocessed_frames(file_name: str, n_frames: int):
    parts = Path(file_name)
    dataset = parts.parts[0]
    scene_id = parts.parts[1]
    clip_stem = parts.stem

    frames_dir = FRAMES_DIR / dataset / scene_id / clip_stem
    if not frames_dir.exists():
        return None, None

    jpgs = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not jpgs:
        return None, None

    n = len(jpgs)
    indices = [int(i * (n - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
    selected = [jpgs[min(i, n - 1)] for i in indices]

    gaze_json = frames_dir / "gaze.json"
    gaze_data = {}
    if gaze_json.exists():
        with open(gaze_json) as f:
            gaze_data = json.load(f)

    frames = [Image.open(p).convert("RGB") for p in selected]
    frame_gaze = [gaze_data.get(p.stem) for p in selected]
    print(f"        frames from preprocess: {len(frames)}  (gaze available: {sum(1 for g in frame_gaze if g)})")
    return frames, frame_gaze


def build_inputs(processor, n_frames: int):
    video_path, row, output_log = load_sample()

    frames, frame_gaze = load_preprocessed_frames(row["file_name"], n_frames)
    if frames is None:
        frames = extract_frames(video_path, n_frames)
        frame_gaze = None
        print(f"        frames extracted (cv2): {len(frames)}")

    # Use the actual VQA question so outputs are checkable against ground truth.
    # Asking for step-by-step reasoning first produces multi-token output (needed
    # for meaningful decode-time profiling) while "Answer: X" at the end is easy
    # to parse for accuracy evaluation.
    options = row["answer_options"].split("|")
    options_str = "\n".join(o.strip() for o in options)
    prompt = (
        f"Watch the video carefully and answer the following multiple-choice question.\n"
        f"Think step by step about what you observe, then end your response with "
        f"'Answer: X' where X is the letter of the correct option.\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{options_str}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    inputs = inputs.to(device)

    pixel_values_videos = getattr(inputs, "pixel_values_videos", None)
    pixel_shape = pixel_values_videos.shape if pixel_values_videos is not None else "?"
    print(f"        input_ids shape: {inputs.input_ids.shape}  pixel_values shape: {pixel_shape}")

    return inputs, row, output_log, frame_gaze, frames


def build_pruning_kwargs(
    prune_text: bool,
    prune_gaze: bool,
    prune_layers: list[int],
    prune_ratio: float,
    prune_alpha: float = 0.5,
) -> dict:
    """Build the kwargs dict that gets unpacked into model.generate().
    gaze_scores is NOT included here — it is added to the dict after being
    computed from frame_gaze + video_grid_thw (which are only available after
    build_inputs runs).
    """
    if not prune_text and not prune_gaze:
        return {}
    kwargs: dict = {
        "pruning_layers": prune_layers,
        "pruning_ratios": {layer: prune_ratio for layer in prune_layers},
    }
    if prune_text:
        kwargs["prune_text"] = True
    if prune_gaze:
        kwargs["prune_gaze"] = True
        kwargs["prune_alpha"] = prune_alpha
    return kwargs


def flatten_feature_output(pooler_output):
    if isinstance(pooler_output, (tuple, list)):
        return torch.cat(pooler_output, dim=0)
    return pooler_output


def extract_video_embeds(model, inputs):
    pixel_values_videos = getattr(inputs, "pixel_values_videos", None)
    if pixel_values_videos is None:
        return None

    video_outputs = model.get_video_features(
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=inputs.video_grid_thw,
    )
    video_embeds = flatten_feature_output(video_outputs.pooler_output)
    return video_embeds.to(device=inputs.input_ids.device, dtype=model.dtype)


def build_prefill_inputs(model, inputs, video_embeds=None):
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    mm_token_type_ids = inputs.mm_token_type_ids
    image_grid_thw = getattr(inputs, "image_grid_thw", None)
    video_grid_thw = getattr(inputs, "video_grid_thw", None)

    inputs_embeds = model.model.get_input_embeddings()(input_ids)

    if video_embeds is not None:
        _, video_mask = model.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    position_ids = model.model.compute_3d_position_ids(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        mm_token_type_ids=mm_token_type_ids,
    )

    return {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
    }


def compute_gaze_scores(
    frame_gaze: list,
    video_grid_thw: torch.Tensor,
    sigma_frac: float = 0.15,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a flat (T * H_m * W_m,) gaze-score tensor aligned with the LLM's visual
    token sequence (frame-by-frame, row-major within each frame).

    Args:
        frame_gaze:     list[dict | None], one entry per sampled frame.
                        Each dict must contain "gaze_x" and "gaze_y" in [0, 1]
                        (normalised to original image size).
        video_grid_thw: tensor of shape (num_videos, 3) = [T, H_pre, W_pre].
                        T   = number of frames.
                        H_pre / W_pre = pre-merger patch grid per frame.
                        Post-merger: H_m = H_pre // 2, W_m = W_pre // 2.
        sigma_frac:     Gaussian sigma expressed as a fraction of max(H_m, W_m).
                        Default 0.15 → roughly 15 % of the grid width.
        device:         Target device string (e.g. "mps", "cpu").

    Returns:
        gaze_scores: FloatTensor of shape (T * H_m * W_m,).
                     Frames with no gaze data get uniform scores of 1.0
                     (no gaze preference = keep all tokens equally).
    """
    T_raw, H_pre, W_pre = video_grid_thw[0].tolist()
    T, H_pre, W_pre = int(T_raw), int(H_pre), int(W_pre)
    H_m = H_pre // 2
    W_m = W_pre // 2
    sigma = sigma_frac * max(H_m, W_m)

    # Coordinate grids over the post-merger token grid — shape (H_m, W_m) each
    rows = torch.arange(H_m, dtype=torch.float32)
    cols = torch.arange(W_m, dtype=torch.float32)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")

    frame_score_list = []
    for t in range(T):
        gaze = frame_gaze[t] if t < len(frame_gaze) else None
        if gaze is not None and "gaze_x" in gaze and "gaze_y" in gaze:
            # Map normalised [0, 1] image coordinates → post-merger token-grid indices
            gc = float(gaze["gaze_x"]) * (W_m - 1)   # column in token grid
            gr = float(gaze["gaze_y"]) * (H_m - 1)   # row    in token grid
            dist_sq = (grid_r - gr) ** 2 + (grid_c - gc) ** 2
            scores = torch.exp(-dist_sq / (2.0 * sigma ** 2))  # (H_m, W_m)
        else:
            # No gaze for this frame → uniform scores (gaze plays no role)
            scores = torch.ones(H_m, W_m, dtype=torch.float32)

        frame_score_list.append(scores.reshape(-1))   # (H_m * W_m,)

    gaze_scores = torch.cat(frame_score_list)          # (T * H_m * W_m,)
    return gaze_scores.to(device)


def run_decoder_only_generate(model, prefill_inputs, generation_kwargs, max_new_tokens: int, min_new_tokens: int):
    return model.generate(
        input_ids=prefill_inputs["input_ids"],
        inputs_embeds=prefill_inputs["inputs_embeds"],
        attention_mask=prefill_inputs["attention_mask"],
        position_ids=prefill_inputs["position_ids"],
        mm_token_type_ids=prefill_inputs["mm_token_type_ids"],
        image_grid_thw=prefill_inputs["image_grid_thw"],
        video_grid_thw=prefill_inputs["video_grid_thw"],
        use_cache=True,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        **generation_kwargs,
    )


def decode_output(processor, inputs, generated_ids):
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0]


def parse_answer(output: str) -> str | None:
    """Extract the predicted letter from 'Answer: X' at the end of the output."""
    m = re.search(r"Answer:\s*([A-E])", output, re.IGNORECASE)
    return m.group(1).upper() if m else None


def visualize_pruning(
    frames: list,
    video_grid_thw: torch.Tensor,
    pruning_debug: dict,
    input_ids: torch.Tensor,
    processor,
    prune_layer: int,
    save_path: Path,
):
    """
    Three-panel figure:
      Row 0 — original frames
      Row 1 — frames with per-token keep/score overlay (RdYlGn, green=kept/high)
      Row 2 — bar chart of text-rater scores (green bar = selected as rater)
    """
    T_raw, H_pre, W_pre = video_grid_thw[0].tolist()
    T, H_pre, W_pre = int(T_raw), int(H_pre), int(W_pre)
    H_m, W_m = H_pre // 2, W_pre // 2

    debug = pruning_debug.get(prune_layer, {})
    visual_keep_mask  = debug.get("visual_keep_mask")         # (T*H_m*W_m,) bool
    combined_scores   = debug.get("combined_visual_scores")   # (T*H_m*W_m,) float | None
    text_rater_scores = debug.get("text_rater_scores")        # (Lt,) float | None
    rater_mask        = debug.get("rater_mask")               # (Lt,) bool  | None
    text_pos          = debug.get("text_pos")                 # (Lt,) int   | None

    # T = temporal slots in LLM token grid (may differ from len(frames) due to
    # Qwen2-VL temporal compression: 2 input frames → 1 temporal slot).
    # Map each slot to the most representative input frame for display.
    n_input = len(frames)
    slot_to_frame = [
        min(int(round(t * (n_input - 1) / max(T - 1, 1))), n_input - 1)
        for t in range(T)
    ]

    has_text = text_rater_scores is not None and text_pos is not None

    fig = plt.figure(figsize=(4 * T, 11 if has_text else 7))
    gs  = gridspec.GridSpec(3 if has_text else 2, T, figure=fig, hspace=0.35, wspace=0.05)

    for t in range(T):
        frame_np = np.array(frames[slot_to_frame[t]])

        # ── Row 0: original ──────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, t])
        ax.imshow(frame_np)
        ax.set_title(f"Frame {t + 1} — original", fontsize=8)
        ax.axis("off")

        # ── Row 1: token overlay ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, t])
        ax.imshow(frame_np)

        if visual_keep_mask is not None:
            tok_s = t * H_m * W_m
            tok_e = tok_s + H_m * W_m

            # Use continuous scores when available, otherwise binary keep mask
            if combined_scores is not None:
                # cast to float32 — numpy doesn't support bfloat16
                grid = combined_scores[tok_s:tok_e].reshape(H_m, W_m).float().numpy()
            else:
                grid = visual_keep_mask[tok_s:tok_e].reshape(H_m, W_m).numpy().astype(float)

            # Resize grid to match frame pixel dimensions (nearest = sharp edges)
            grid_img = Image.fromarray((grid * 255).astype(np.uint8)).resize(
                (frame_np.shape[1], frame_np.shape[0]), Image.NEAREST
            )
            rgba = plt.cm.RdYlGn(np.array(grid_img) / 255.0)
            rgba[..., 3] = 0.50  # alpha
            ax.imshow(rgba)

            kept  = int(visual_keep_mask[tok_s:tok_e].sum().item())
            total = H_m * W_m
            ax.set_title(f"Frame {t + 1} — kept {kept}/{total}", fontsize=8)
        ax.axis("off")

    # ── Row 2: text rater scores ─────────────────────────────────────────────
    if has_text:
        ax = fig.add_subplot(gs[2, :])
        scores_np = text_rater_scores.float().numpy()
        rater_np  = rater_mask.numpy() if rater_mask is not None else np.zeros(len(scores_np), bool)

        MAX_T = 80  # keep chart readable
        tok_ids  = input_ids[0, text_pos].tolist()
        tok_strs = [
            processor.tokenizer.decode([tid], skip_special_tokens=False)
            .replace(" ", "·").replace("\n", "↵")[:10]
            for tid in tok_ids
        ]
        if len(scores_np) > MAX_T:
            scores_np = scores_np[:MAX_T]
            rater_np  = rater_np[:MAX_T]
            tok_strs  = tok_strs[:MAX_T]

        colors = ["#2ca02c" if r else "#1f77b4" for r in rater_np]
        ax.bar(range(len(scores_np)), scores_np, color=colors, width=0.85)
        ax.set_xticks(range(len(tok_strs)))
        ax.set_xticklabels(tok_strs, rotation=90, fontsize=5)
        ax.set_ylabel("Rater score")
        ax.set_title(
            f"Text-token rater scores  "
            f"(green = selected rater, {int(rater_np.sum())}/{len(scores_np)} selected)",
            fontsize=9,
        )

    fig.suptitle(f"Pruning visualisation — layer {prune_layer}", fontsize=11, y=1.01)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"        Visualisation saved: {save_path}")


def print_timing_summary(results):
    print("\n── Profiling Summary ───────────────────────────────────────────────")
    for key in ["model_load", "input_preprocessing", "vision_encoder", "decoder_only", "full_run"]:
        r = results.get(key, {})
        print(
            f"  {key:<20} {r.get('time_s', 0) * 1000:8.1f} ms"
            f"  mem Δ {r.get('mem_delta_mb', 0):+.1f} MB"
        )


def write_log(
    device,
    n_frames,
    results,
    inputs,
    sample_log,
    prune_text,
    prune_gaze,
    prune_layers,
    prune_ratio,
    prune_alpha,
    decoder_output,
    full_output,
    correct_answer,
    predicted_answer,
    viz_path=None,
):
    run_number = 1
    if LOG_FILE.exists():
        content = LOG_FILE.read_text()
        run_number = content.count("## Run ") + 1

    pixel_values_videos = getattr(inputs, "pixel_values_videos", None)
    pixel_shape = tuple(pixel_values_videos.shape) if pixel_values_videos is not None else "n/a"

    accuracy_str = "n/a"
    if predicted_answer is not None:
        correct = predicted_answer == correct_answer
        accuracy_str = f"{predicted_answer} {'✓' if correct else '✗'} (correct: {correct_answer})"

    viz_str = f"`{viz_path}`" if viz_path else "none"

    entry = f"""
## Run {run_number}
**Date:** {datetime.date.today()}
**Model:** {MODEL_ID}
**Forked model file:** {FORKED_MODEL_PATH}
**Device:** {device.upper()} | torch {torch.__version__}
**Frames:** {n_frames}
**Prune text:** {prune_text}
**Prune gaze:** {prune_gaze}
**Prune layers:** {prune_layers}
**Prune ratio:** {prune_ratio}
**Prune alpha:** {prune_alpha}
**Predicted answer:** {accuracy_str}
**Visualisation:** {viz_str}
**Input shape:** input_ids {tuple(inputs.input_ids.shape)} | pixel_values {pixel_shape}
**Input logs:**
{sample_log}

### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | {results.get("model_load", {}).get("time_s", 0) * 1000:.1f} | {results.get("model_load", {}).get("mem_delta_mb", 0):+.1f} |
| input_preprocessing | {results.get("input_preprocessing", {}).get("time_s", 0) * 1000:.1f} | {results.get("input_preprocessing", {}).get("mem_delta_mb", 0):+.1f} |
| vision_encoder | {results.get("vision_encoder", {}).get("time_s", 0) * 1000:.1f} | {results.get("vision_encoder", {}).get("mem_delta_mb", 0):+.1f} |
| decoder_only | {results.get("decoder_only", {}).get("time_s", 0) * 1000:.1f} | {results.get("decoder_only", {}).get("mem_delta_mb", 0):+.1f} |
| full_run | {results.get("full_run", {}).get("time_s", 0) * 1000:.1f} | {results.get("full_run", {}).get("mem_delta_mb", 0):+.1f} |

### Outputs
**Decoder-only output**
> {decoder_output.replace(chr(10), ' ')}

**Full-run output**
> {full_output.replace(chr(10), ' ')}

---
"""

    with open(LOG_FILE, "a") as f:
        if run_number == 1:
            f.write("# Forked Qwen2-VL Profiling Log\n")
        f.write(entry)

    print(f"\nLogged to {LOG_FILE}  (run #{run_number})")


def main(
    n_frames: int = 4,
    prune_text: bool = False,
    prune_gaze: bool = False,
    prune_layers: list[int] | None = None,
    prune_ratio: float = 0.5,
    prune_alpha: float = 0.5,
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    prune_layers = prune_layers or [27]
    generation_kwargs = build_pruning_kwargs(prune_text, prune_gaze, prune_layers, prune_ratio, prune_alpha)

    print(f"Device: {device}  |  torch {torch.__version__}  |  frames: {n_frames}")

    results = {}

    # Flush MPS cache so leftover allocations from previous runs don't
    # push the model into disk-offload territory.
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    with timed_block("model_load", results):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map={"": device},   # force ALL weights onto the chosen device
            attn_implementation="eager",
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID)

    mem_after_load = mps_allocated_mb()
    print(
        f"Model loaded in {results['model_load']['time_s']:.2f}s  "
        f"({results['model_load']['mem_peak_mb']:.0f} MB MPS)"
    )

    # Hard check: if the model is below the expected full-MPS footprint it
    # was partially offloaded and the run will be artificially slow. Abort
    # early rather than waste minutes on a meaningless timing.
    FULL_LOAD_MB = 4000  # Qwen2-VL-2B in bfloat16 ≈ 4418 MB
    if torch.backends.mps.is_available() and mem_after_load < FULL_LOAD_MB:
        raise RuntimeError(
            f"Model only allocated {mem_after_load:.0f} MB — likely disk-offloaded. "
            f"Close other apps to free MPS memory and retry."
        )

    with timed_block("input_preprocessing", results):
        inputs, row, sample_log, frame_gaze, frames = build_inputs(processor, n_frames)

    # gaze scores — needs both frame_gaze AND inputs.video_grid_thw
    if prune_gaze:
        gaze_scores = compute_gaze_scores(
            frame_gaze,
            inputs.video_grid_thw,
            sigma_frac=0.15,
            device=str(inputs.input_ids.device),
        )
        generation_kwargs["gaze_scores"] = gaze_scores
        n_nonuniform = int((gaze_scores < 0.99).sum().item())
        print(f"        gaze_scores shape: {tuple(gaze_scores.shape)}  focused tokens: {n_nonuniform}")

    print("\nWarming up...")
    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=16,
            min_new_tokens=8,
            **generation_kwargs,
        )
    mps_sync()
    print("Warmup done.\n")

    print("\nEncoding video...")
    with torch.inference_mode():
        with timed_block("vision_encoder", results):
            video_embeds = extract_video_embeds(model, inputs)

    print("Encoding video done.\n")

    prefill_inputs = build_prefill_inputs(model, inputs, video_embeds=video_embeds)

    print("\Decoding...")
    with torch.inference_mode():
        with timed_block("decoder_only", results):
            decoder_generated_ids = run_decoder_only_generate(
                model,
                prefill_inputs,
                generation_kwargs,
                max_new_tokens=512,
                min_new_tokens=200,
            )
    print("Decoding done.\n")

    print("\Running full run...")
    with torch.inference_mode():
        with timed_block("full_run", results):
            full_generated_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=512,
                min_new_tokens=200,
                **generation_kwargs,
            )
    print("Full run done.\n")

    decoder_output = decode_output(processor, inputs, decoder_generated_ids)
    full_output    = decode_output(processor, inputs, full_generated_ids)

    predicted_answer = parse_answer(decoder_output)
    correct_answer   = row["correct_answer"].strip()
    correct          = predicted_answer == correct_answer if predicted_answer else None

    print_timing_summary(results)
    print(f"\nReference answer : {correct_answer}")
    print(f"Predicted answer : {predicted_answer}  {'✓' if correct else '✗' if correct is not None else '?'}")
    print(f"\nDecoder-only output: {decoder_output[:400]}")
    print(f"\nFull-run output:     {full_output[:400]}")

    # ── Pruning visualisation ────────────────────────────────────────────────
    viz_path = None
    pruning_active = prune_text or prune_gaze
    if pruning_active:
        pruning_debug = getattr(
            model.model.language_model, "_pruning_debug", {}
        )
        if pruning_debug:
            run_number = (
                LOG_FILE.read_text().count("## Run ") + 1 if LOG_FILE.exists() else 1
            )
            config_tag = f"{'t' if prune_text else ''}{'g' if prune_gaze else ''}_l{prune_layers[0]}_r{prune_ratio}"
            viz_path = VIZ_DIR / f"run{run_number:02d}_{config_tag}.png"
            visualize_pruning(
                frames=frames,
                video_grid_thw=inputs.video_grid_thw.cpu(),
                pruning_debug=pruning_debug,
                input_ids=inputs.input_ids.cpu(),
                processor=processor,
                prune_layer=prune_layers[0],
                save_path=viz_path,
            )

    write_log(
        device=device,
        n_frames=n_frames,
        results=results,
        inputs=inputs,
        sample_log=sample_log,
        prune_text=prune_text,
        prune_gaze=prune_gaze,
        prune_layers=prune_layers,
        prune_ratio=prune_ratio,
        prune_alpha=prune_alpha,
        decoder_output=decoder_output,
        full_output=full_output,
        correct_answer=correct_answer,
        predicted_answer=predicted_answer,
        viz_path=viz_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4, help="Number of frames to sample from the clip.")
    parser.add_argument("--prune-text", action="store_true", help="Enable forked in-model SparseVLM text pruning.")
    parser.add_argument("--prune-gaze", action="store_true", help="Enable forked in-model gaze pruning.")
    parser.add_argument("--prune-layers", type=int, nargs="+", default=[27], help="Decoder layer indices to prune.")
    parser.add_argument("--prune-ratio", type=float, default=0.5, help="Ratio of visual tokens to keep.")
    parser.add_argument("--prune-alpha", type=float, default=0.5, help="Ratio of gaze pruning importance vs text pruning importance")
    args = parser.parse_args()

    main(
        n_frames=args.frames,
        prune_text=args.prune_text,
        prune_gaze=args.prune_gaze,
        prune_layers=args.prune_layers,
        prune_ratio=args.prune_ratio,
        prune_alpha=args.prune_alpha,
    )
