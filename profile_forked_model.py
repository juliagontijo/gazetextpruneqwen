import argparse
import contextlib
import csv
import datetime
import importlib.util
import json
import random
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

CSV_COLUMNS = [
    "config_tag",
    "sample_idx",
    "video_id",
    "qa_type",
    "prune_text",
    "prune_gaze",
    "prune_random",
    "prune_alpha",
    "prune_ratio",
    "prune_layer",
    "input_preprocessing_s",
    "vision_encoder_s",
    "decode_s",
    "tokens_generated",
    "decode_ms_per_token",
    "correct_answer",
    "predicted_answer",
    "correct",
]


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


def load_samples(
    num_samples: int,
    seed: int,
    qa_types: list[str] | None = None,
) -> list[dict]:
    """
    Return a reproducibly-shuffled list of `num_samples` rows from metadata.csv
    that have local frame folders on disk.  If `qa_types` is given, only rows
    whose qa_type is in that list are considered.  Rows are balanced across
    qa_types when possible (round-robin).
    """
    with open(METADATA_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))

    # Filter to rows that have local pre-processed frame folders
    available: list[dict] = []
    for row in all_rows:
        fn = row["file_name"].replace(".mp4", "")
        folder = FRAMES_DIR / fn
        if folder.exists() and list(folder.glob("*.jpg")):
            available.append(row)

    # Optional qa_type filter
    if qa_types:
        available = [r for r in available if r["qa_type"] in qa_types]

    if not available:
        raise FileNotFoundError("No locally available clips match the filters.")

    # Reproducible shuffle
    rng = random.Random(seed)
    rng.shuffle(available)

    # Balance across qa_types (round-robin interleave)
    if qa_types and len(qa_types) > 1:
        buckets: dict[str, list[dict]] = {qt: [] for qt in qa_types}
        for row in available:
            qt = row["qa_type"]
            if qt in buckets:
                buckets[qt].append(row)
        balanced: list[dict] = []
        iters = [iter(v) for v in buckets.values()]
        while len(balanced) < num_samples:
            added = False
            for it in iters:
                try:
                    balanced.append(next(it))
                    added = True
                    if len(balanced) >= num_samples:
                        break
                except StopIteration:
                    pass
            if not added:
                break
        return balanced[:num_samples]

    return available[:num_samples]


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
    return frames, frame_gaze


def build_inputs(processor, row: dict, n_frames: int):
    """
    Prepare model inputs for a single metadata row.
    Returns (inputs, sample_log, frame_gaze, frames).
    """
    local_path = DATA_DIR / row["file_name"]

    frames, frame_gaze = load_preprocessed_frames(row["file_name"], n_frames)
    if frames is None:
        if not local_path.exists():
            raise FileNotFoundError(f"No video or frames for {row['file_name']}")
        frames = extract_frames(str(local_path), n_frames)
        frame_gaze = [None] * n_frames

    options = row["answer_options"].split("|")
    options_str = "\n".join(o.strip() for o in options)
    prompt = (
        f"Watch the video carefully and answer the following multiple-choice question.\n"
        f"Think step by step about what you observe, then end your response with "
        f"'Answer: X' where X is the letter of the correct option.\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{options_str}"
    )

    sample_log = (
        f"  file:     {row['file_name']}\n"
        f"  qa_type:  {row['qa_type']}\n"
        f"  question: {row['question']}\n"
        f"  answer:   {row['correct_answer']}"
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

    return inputs, sample_log, frame_gaze, frames


def build_pruning_kwargs(
    prune_text: bool,
    prune_gaze: bool,
    prune_random: bool,
    prune_layers: list[int],
    prune_ratio: float,
    prune_alpha: float = 0.5,
) -> dict:
    """
    Build the kwargs dict passed to model.generate().
    prune_random: keep tokens by uniform random scores — uses the gaze pathway
                  internally (alpha=0.0, pure "gaze") but with random scores
                  injected before the call.  Mutually exclusive with prune_gaze.
    """
    any_pruning = prune_text or prune_gaze or prune_random
    if not any_pruning:
        return {}

    kwargs: dict = {
        "pruning_layers": prune_layers,
        "pruning_ratios": {layer: prune_ratio for layer in prune_layers},
    }
    if prune_text and not prune_random:
        kwargs["prune_text"] = True
    if prune_gaze and not prune_random:
        kwargs["prune_gaze"] = True
        kwargs["prune_alpha"] = prune_alpha
    if prune_random:
        # Random uses the gaze pathway; scores are injected separately.
        # alpha=0.0 → combined = 1.0 * gaze_scores (pure random), text ignored.
        kwargs["prune_gaze"] = True
        kwargs["prune_alpha"] = 0.0
    return kwargs


def compute_gaze_scores(
    frame_gaze: list,
    video_grid_thw: torch.Tensor,
    sigma_frac: float = 0.15,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a flat (T * H_m * W_m,) gaze-score tensor aligned with the LLM's
    visual token sequence (frame-major, row-major within each frame).
    """
    T_raw, H_pre, W_pre = video_grid_thw[0].tolist()
    T, H_pre, W_pre = int(T_raw), int(H_pre), int(W_pre)
    H_m = H_pre // 2
    W_m = W_pre // 2
    sigma = sigma_frac * max(H_m, W_m)

    rows_t = torch.arange(H_m, dtype=torch.float32)
    cols_t = torch.arange(W_m, dtype=torch.float32)
    grid_r, grid_c = torch.meshgrid(rows_t, cols_t, indexing="ij")

    frame_score_list = []
    for t in range(T):
        gaze = frame_gaze[t] if t < len(frame_gaze) else None
        if gaze is not None and "gaze_x" in gaze and "gaze_y" in gaze:
            gc = float(gaze["gaze_x"]) * (W_m - 1)
            gr = float(gaze["gaze_y"]) * (H_m - 1)
            dist_sq = (grid_r - gr) ** 2 + (grid_c - gc) ** 2
            scores = torch.exp(-dist_sq / (2.0 * sigma ** 2))
        else:
            scores = torch.ones(H_m, W_m, dtype=torch.float32)
        frame_score_list.append(scores.reshape(-1))

    return torch.cat(frame_score_list).to(device)


def compute_random_scores(
    video_grid_thw: torch.Tensor,
    sample_idx: int,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Uniform random scores for visual tokens, reproducible per sample.
    Shape: (T * H_m * W_m,).
    """
    T_raw, H_pre, W_pre = video_grid_thw[0].tolist()
    T, H_m, W_m = int(T_raw), int(H_pre) // 2, int(W_pre) // 2
    gen = torch.Generator()
    gen.manual_seed(seed * 10_000 + sample_idx)
    return torch.rand(T * H_m * W_m, generator=gen).to(device)


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


def run_decoder_only_generate(model, prefill_inputs, generation_kwargs, max_new_tokens, min_new_tokens):
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
    m = re.search(r"Answer:\s*([A-E])", output, re.IGNORECASE)
    return m.group(1).upper() if m else None


def append_csv_row(csv_path: Path, row: dict):
    """Append one result row to the CSV; write header if the file is new."""
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


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
      Row 0 — original frames (T temporal slots)
      Row 1 — frames with per-token keep/score overlay (RdYlGn)
      Row 2 — bar chart of text-rater scores (green = selected rater)
    """
    T_raw, H_pre, W_pre = video_grid_thw[0].tolist()
    T, H_pre, W_pre = int(T_raw), int(H_pre), int(W_pre)
    H_m, W_m = H_pre // 2, W_pre // 2

    debug = pruning_debug.get(prune_layer, {})
    visual_keep_mask  = debug.get("visual_keep_mask")
    combined_scores   = debug.get("combined_visual_scores")
    text_rater_scores = debug.get("text_rater_scores")
    rater_mask        = debug.get("rater_mask")
    text_pos          = debug.get("text_pos")

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

        ax = fig.add_subplot(gs[0, t])
        ax.imshow(frame_np)
        ax.set_title(f"Slot {t + 1} — original", fontsize=8)
        ax.axis("off")

        ax = fig.add_subplot(gs[1, t])
        ax.imshow(frame_np)
        if visual_keep_mask is not None:
            tok_s = t * H_m * W_m
            tok_e = tok_s + H_m * W_m
            if combined_scores is not None:
                grid = combined_scores[tok_s:tok_e].reshape(H_m, W_m).float().numpy()
            else:
                grid = visual_keep_mask[tok_s:tok_e].reshape(H_m, W_m).numpy().astype(float)
            grid_img = Image.fromarray((grid * 255).astype(np.uint8)).resize(
                (frame_np.shape[1], frame_np.shape[0]), Image.NEAREST
            )
            rgba = plt.cm.RdYlGn(np.array(grid_img) / 255.0)
            rgba[..., 3] = 0.50
            ax.imshow(rgba)
            kept  = int(visual_keep_mask[tok_s:tok_e].sum().item())
            ax.set_title(f"Slot {t + 1} — kept {kept}/{H_m * W_m}", fontsize=8)
        ax.axis("off")

    if has_text:
        ax = fig.add_subplot(gs[2, :])
        scores_np = text_rater_scores.float().numpy()
        rater_np  = rater_mask.numpy() if rater_mask is not None else np.zeros(len(scores_np), bool)
        MAX_T = 80
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
            f"(green = selected rater, {int(rater_np.sum())}/{len(scores_np)} shown)",
            fontsize=9,
        )

    fig.suptitle(f"Pruning visualisation — layer {prune_layer}", fontsize=11, y=1.01)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"        viz → {save_path}")


def run_sample(
    model,
    processor,
    row: dict,
    sample_idx: int,
    n_frames: int,
    prune_text: bool,
    prune_gaze: bool,
    prune_random: bool,
    prune_layers: list[int],
    prune_ratio: float,
    prune_alpha: float,
    config_tag: str,
    device: str,
    seed: int,
    save_viz: bool = False,
    viz_suffix: str = "",
) -> dict:
    """
    Run inference on a single sample row.
    Returns a dict matching CSV_COLUMNS (plus 'decoder_output' for markdown log).
    """
    results: dict = {}

    with timed_block("input_preprocessing", results):
        inputs, sample_log, frame_gaze, frames = build_inputs(processor, row, n_frames)

    # Base pruning kwargs (no gaze_scores yet)
    generation_kwargs = build_pruning_kwargs(
        prune_text, prune_gaze, prune_random, prune_layers, prune_ratio, prune_alpha
    )

    # Inject gaze or random scores into kwargs
    if prune_gaze and not prune_random:
        gaze_scores = compute_gaze_scores(
            frame_gaze,
            inputs.video_grid_thw,
            sigma_frac=0.15,
            device=device,
        )
        generation_kwargs["gaze_scores"] = gaze_scores

    if prune_random:
        rand_scores = compute_random_scores(
            inputs.video_grid_thw,
            sample_idx=sample_idx,
            seed=seed,
            device=device,
        )
        generation_kwargs["gaze_scores"] = rand_scores

    with torch.inference_mode():
        with timed_block("vision_encoder", results):
            video_embeds = extract_video_embeds(model, inputs)

    prefill_inputs = build_prefill_inputs(model, inputs, video_embeds=video_embeds)

    with torch.inference_mode():
        with timed_block("decode", results):
            generated_ids = run_decoder_only_generate(
                model,
                prefill_inputs,
                generation_kwargs,
                max_new_tokens=512,
                min_new_tokens=200,
            )

    decoder_output  = decode_output(processor, inputs, generated_ids)
    predicted_answer = parse_answer(decoder_output)
    correct_answer   = row["correct_answer"].strip()
    is_correct       = int(predicted_answer == correct_answer) if predicted_answer else None

    tokens_generated     = generated_ids.shape[1] - inputs.input_ids.shape[1]
    decode_s             = results["decode"]["time_s"]
    decode_ms_per_token  = (decode_s * 1000 / tokens_generated) if tokens_generated > 0 else None

    correct_sym = "✓" if is_correct else ("✗" if is_correct is not None else "?")
    print(
        f"  [{sample_idx:03d}] {row['qa_type']:8s}  "
        f"pred={predicted_answer or '?'} {correct_sym}  "
        f"decode={decode_s:.2f}s  {decode_ms_per_token:.1f}ms/tok  "
        f"toks={tokens_generated}"
    )

    # Optional visualisation (only for single-sample runs or when explicitly requested)
    if save_viz and (prune_text or prune_gaze or prune_random):
        pruning_debug = getattr(model.model.language_model, "_pruning_debug", {})
        if pruning_debug:
            tag = config_tag.replace("/", "_")
            viz_path = VIZ_DIR / f"s{sample_idx:03d}_{tag}{viz_suffix}.png"
            visualize_pruning(
                frames=frames,
                video_grid_thw=inputs.video_grid_thw.cpu(),
                pruning_debug=pruning_debug,
                input_ids=inputs.input_ids.cpu(),
                processor=processor,
                prune_layer=prune_layers[0],
                save_path=viz_path,
            )

    csv_row = {
        "config_tag":            config_tag,
        "sample_idx":            sample_idx,
        "video_id":              row["video_id"],
        "qa_type":               row["qa_type"],
        "prune_text":            int(prune_text and not prune_random),
        "prune_gaze":            int(prune_gaze and not prune_random),
        "prune_random":          int(prune_random),
        "prune_alpha":           prune_alpha if (prune_text or prune_gaze) and not prune_random else "",
        "prune_ratio":           prune_ratio if (prune_text or prune_gaze or prune_random) else "",
        "prune_layer":           prune_layers[0] if (prune_text or prune_gaze or prune_random) else "",
        "input_preprocessing_s": round(results["input_preprocessing"]["time_s"], 4),
        "vision_encoder_s":      round(results["vision_encoder"]["time_s"], 4),
        "decode_s":              round(decode_s, 4),
        "tokens_generated":      tokens_generated,
        "decode_ms_per_token":   round(decode_ms_per_token, 3) if decode_ms_per_token else "",
        "correct_answer":        correct_answer,
        "predicted_answer":      predicted_answer or "",
        "correct":               is_correct if is_correct is not None else "",
    }
    # stash for markdown (not written to CSV)
    csv_row["_decoder_output"]  = decoder_output
    csv_row["_sample_log"]      = sample_log
    csv_row["_input_ids_shape"] = tuple(inputs.input_ids.shape)
    pv = getattr(inputs, "pixel_values_videos", None)
    csv_row["_pixel_shape"]     = tuple(pv.shape) if pv is not None else "n/a"

    return csv_row


def write_log_entry(
    device: str,
    n_frames: int,
    csv_row: dict,
    prune_text: bool,
    prune_gaze: bool,
    prune_random: bool,
    prune_layers: list[int],
    prune_ratio: float,
    prune_alpha: float,
):
    """Write a single-sample markdown entry (used when --num-samples 1)."""
    run_number = 1
    if LOG_FILE.exists():
        run_number = LOG_FILE.read_text().count("## Run ") + 1

    pred  = csv_row["predicted_answer"]
    corr  = csv_row["correct_answer"]
    is_ok = csv_row["correct"]
    acc_str = f"{pred} {'✓' if is_ok else '✗'} (correct: {corr})" if pred else "n/a"

    method_str = (
        "random" if prune_random
        else ("text+gaze" if (prune_text and prune_gaze)
              else ("text" if prune_text
                    else ("gaze" if prune_gaze else "none")))
    )

    entry = f"""
## Run {run_number}
**Date:** {datetime.date.today()}
**Model:** {MODEL_ID}
**Device:** {device.upper()} | torch {torch.__version__}
**Frames:** {n_frames}
**Method:** {method_str}
**Prune layers:** {prune_layers}
**Prune ratio:** {prune_ratio}
**Prune alpha:** {prune_alpha if (prune_text or prune_gaze) and not prune_random else "n/a"}
**Predicted answer:** {acc_str}
**Input shape:** input_ids {csv_row['_input_ids_shape']} | pixel_values {csv_row['_pixel_shape']}
**Sample:**
{csv_row['_sample_log']}

### Timings
| Stage | Time (ms) |
|---|---:|
| input_preprocessing | {csv_row['input_preprocessing_s'] * 1000:.1f} |
| vision_encoder | {csv_row['vision_encoder_s'] * 1000:.1f} |
| decode | {csv_row['decode_s'] * 1000:.1f} |
| tokens_generated | {csv_row['tokens_generated']} |
| decode_ms_per_token | {csv_row['decode_ms_per_token']} |

### Output
> {csv_row['_decoder_output'].replace(chr(10), ' ')[:600]}

---
"""

    with open(LOG_FILE, "a") as f:
        if run_number == 1:
            f.write("# Forked Qwen2-VL Profiling Log\n")
        f.write(entry)
    print(f"Logged → {LOG_FILE}  (run #{run_number})")


def print_batch_summary(config_tag: str, rows: list[dict]):
    """Print aggregate stats at the end of a multi-sample run."""
    n = len(rows)
    answered  = [r for r in rows if r["correct"] != ""]
    n_correct  = sum(r["correct"] for r in answered)
    accuracy   = n_correct / len(answered) if answered else float("nan")
    times      = [r["decode_ms_per_token"] for r in rows if r["decode_ms_per_token"] != ""]
    mean_time  = sum(times) / len(times) if times else float("nan")

    print(f"\n{'═'*60}")
    print(f"  Config:   {config_tag}")
    print(f"  Samples:  {n}  (answered: {len(answered)})")
    print(f"  Accuracy: {n_correct}/{len(answered)} = {accuracy:.1%}")
    print(f"  Mean decode: {mean_time:.1f} ms/tok")

    # breakdown by qa_type
    for qt in sorted({r["qa_type"] for r in rows}):
        subset = [r for r in answered if r["qa_type"] == qt]
        nc = sum(r["correct"] for r in subset)
        print(f"    {qt:10s}  {nc}/{len(subset)} = {nc/len(subset):.1%}" if subset else f"    {qt}: no data")
    print(f"{'═'*60}\n")


def main(
    n_frames: int = 4,
    num_samples: int = 1,
    seed: int = 42,
    qa_types: list[str] | None = None,
    prune_text: bool = False,
    prune_gaze: bool = False,
    prune_random: bool = False,
    prune_layers: list[int] | None = None,
    prune_ratio: float = 0.5,
    prune_alpha: float = 0.5,
    config_tag: str = "",
    results_csv: Path | None = None,
    save_viz: bool = False,
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    prune_layers = prune_layers or [27]

    # Auto-generate config_tag if not provided
    if not config_tag:
        if prune_random:
            method = "random"
        elif prune_text and prune_gaze:
            method = f"combined_a{prune_alpha}"
        elif prune_text:
            method = "text"
        elif prune_gaze:
            method = "gaze"
        else:
            method = "no_prune"
        active = prune_text or prune_gaze or prune_random
        config_tag = f"{method}_l{prune_layers[0]}_r{prune_ratio}" if active else "no_prune"

    print(f"{'─'*60}")
    print(f"  Config tag : {config_tag}")
    print(f"  Device     : {device}  |  torch {torch.__version__}")
    print(f"  Samples    : {num_samples}  (seed={seed})")
    print(f"  Frames     : {n_frames}")
    print(f"  Pruning    : text={prune_text}  gaze={prune_gaze}  random={prune_random}")
    if prune_text or prune_gaze or prune_random:
        print(f"  Layers/ratio: {prune_layers} / {prune_ratio}  alpha={prune_alpha}")
    print(f"{'─'*60}")

    # ── Load model ────────────────────────────────────────────────────────────
    # Brief pause to let the OS reclaim MPS memory from any previous process
    # that may not have fully released it yet.
    if torch.backends.mps.is_available():
        import time as _time
        _time.sleep(5)
        torch.mps.empty_cache()

    t_load = time.perf_counter()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    mps_sync()
    load_s = time.perf_counter() - t_load

    mem_after_load = mps_allocated_mb()
    print(f"Model loaded in {load_s:.2f}s  ({mem_after_load:.0f} MB)")

    FULL_LOAD_MB = 4300  # full bfloat16 load = ~4418 MB; abort if significantly below
    if torch.backends.mps.is_available() and mem_after_load < FULL_LOAD_MB:
        raise RuntimeError(
            f"Model only allocated {mem_after_load:.0f} MB (expected ~4418 MB) — "
            f"likely partial disk-offload from residual MPS memory of a previous run. "
            f"Wait 15s and retry, or restart your terminal."
        )

    # ── Load samples ─────────────────────────────────────────────────────────
    samples = load_samples(num_samples, seed, qa_types)
    print(f"Loaded {len(samples)} samples.")

    # ── Warmup (single forward, not timed, uses first sample) ────────────────
    print("\nWarming up...")
    warmup_inputs, _, warmup_frame_gaze, _ = build_inputs(processor, samples[0], n_frames)
    generation_kwargs_warmup = build_pruning_kwargs(
        prune_text, prune_gaze, prune_random, prune_layers, prune_ratio, prune_alpha
    )
    if prune_random:
        generation_kwargs_warmup["gaze_scores"] = compute_random_scores(
            warmup_inputs.video_grid_thw, sample_idx=0, seed=seed, device=device
        )
    elif prune_gaze:
        generation_kwargs_warmup["gaze_scores"] = compute_gaze_scores(
            warmup_frame_gaze, warmup_inputs.video_grid_thw, device=device
        )
    with torch.inference_mode():
        _ = model.generate(
            **warmup_inputs,
            use_cache=True,
            max_new_tokens=8,
            **generation_kwargs_warmup,
        )
    mps_sync()
    del warmup_inputs
    print("Warmup done.\n")

    # ── Sample loop ───────────────────────────────────────────────────────────
    all_results: list[dict] = []

    for idx, row in enumerate(samples):
        csv_row = run_sample(
            model=model,
            processor=processor,
            row=row,
            sample_idx=idx,
            n_frames=n_frames,
            prune_text=prune_text,
            prune_gaze=prune_gaze,
            prune_random=prune_random,
            prune_layers=prune_layers,
            prune_ratio=prune_ratio,
            prune_alpha=prune_alpha,
            config_tag=config_tag,
            device=device,
            seed=seed,
            save_viz=save_viz or (num_samples == 1),
            viz_suffix="",
        )
        all_results.append(csv_row)

        # Always write CSV row immediately (safe even if run crashes mid-way)
        if results_csv:
            clean = {k: v for k, v in csv_row.items() if not k.startswith("_")}
            append_csv_row(results_csv, clean)

        # Markdown log for single-sample runs (backward compat)
        if num_samples == 1:
            write_log_entry(
                device=device,
                n_frames=n_frames,
                csv_row=csv_row,
                prune_text=prune_text,
                prune_gaze=prune_gaze,
                prune_random=prune_random,
                prune_layers=prune_layers,
                prune_ratio=prune_ratio,
                prune_alpha=prune_alpha,
            )

    print_batch_summary(config_tag, all_results)
    if results_csv:
        print(f"Results appended → {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile forked Qwen2-VL with visual token pruning.")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--frames",      type=int,   default=4,    help="Frames per clip.")
    parser.add_argument("--num-samples", type=int,   default=1,    help="Number of samples to evaluate.")
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed for sample selection.")
    parser.add_argument("--qa-types",    nargs="+",  default=None,
                        choices=["causal", "spatial", "temporal"],
                        help="QA types to include (default: all).")

    # ── Pruning ───────────────────────────────────────────────────────────────
    parser.add_argument("--prune-text",   action="store_true", help="Text-rater pruning.")
    parser.add_argument("--prune-gaze",   action="store_true", help="Gaze-guided pruning.")
    parser.add_argument("--prune-random", action="store_true", help="Random pruning baseline.")
    parser.add_argument("--prune-layers", type=int, nargs="+", default=[27],
                        help="Decoder layer indices to prune at.")
    parser.add_argument("--prune-ratio",  type=float, default=0.5,
                        help="Fraction of visual tokens to keep (0–1).")
    parser.add_argument("--prune-alpha",  type=float, default=0.5,
                        help="Alpha: weight of text scores vs gaze scores (1=pure text, 0=pure gaze).")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--config-tag",   type=str, default="",
                        help="Label for this condition in the results CSV.")
    parser.add_argument("--results-csv",  type=Path, default=None,
                        help="Path to append per-sample results CSV.")
    parser.add_argument("--save-viz",     action="store_true",
                        help="Save pruning visualisation PNG for every sample.")

    args = parser.parse_args()

    # Validate: prune-random is mutually exclusive with prune-text/prune-gaze
    if args.prune_random and (args.prune_text or args.prune_gaze):
        parser.error("--prune-random is mutually exclusive with --prune-text and --prune-gaze.")

    main(
        n_frames=args.frames,
        num_samples=args.num_samples,
        seed=args.seed,
        qa_types=args.qa_types,
        prune_text=args.prune_text,
        prune_gaze=args.prune_gaze,
        prune_random=args.prune_random,
        prune_layers=args.prune_layers,
        prune_ratio=args.prune_ratio,
        prune_alpha=args.prune_alpha,
        config_tag=args.config_tag,
        results_csv=args.results_csv,
        save_viz=args.save_viz,
    )
