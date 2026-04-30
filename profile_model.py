"""
VLM Profiler for Qwen2-VL on Apple M1 (MPS backend)

Profiled components:
  1. vision_encoder     — model.visual (ViT forward, all frames batched)
  2. token_merger       — model.visual.merger (projector, reduces visual tokens)
  3. prefill            — first decoder forward (processes visual + text tokens)
  4. decode             — subsequent decoder forwards (autoregressive steps)
  5. end_to_end         — full model.generate()

Video reader: cv2 frame extraction (bypasses broken torchvision.io.read_video)

Usage:
  python profile_model.py                  # all layers + chrome trace
  python profile_model.py --no-trace       # skip chrome trace (faster)
  python profile_model.py --frames N       # number of frames to sample (default 4)
"""

import time
import contextlib
import argparse
import json
import os
from pathlib import Path

import csv
import datetime

import cv2
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID      = "Qwen/Qwen2-VL-2B-Instruct"
DATA_DIR      = Path(__file__).parent / "data" / "EgoGazeVQA_full"
EGO4D_JSON    = DATA_DIR / "ego4d.json"
EGO4D_VIDEOS  = DATA_DIR / "ego4d"
METADATA_CSV  = DATA_DIR / "metadata.csv"
FRAMES_DIR    = DATA_DIR / "frames"
VIDEO_PAD_TOKEN_ID = 151656
SPARSEVLM_KEEP_RATIO = 0.50
SPARSEVLM_PROBE_LAYER = -1
GAZE_PRUNE_THRESHOLD = 0.30
TEXT_RATER_LOG_PATH = Path(__file__).parent / "text_raters.json"


# ── helpers ──────────────────────────────────────────────────────────────────

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


# ── video reader (cv2-based, no torchvision.io dependency) ───────────────────

def extract_frames(video_path: str, n_frames: int) -> list[Image.Image]:
    """
    Uniformly sample n_frames from a video clip using cv2.
    Returns a list of PIL images.
    """
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


# ── sample loading ────────────────────────────────────────────────────────────

def load_sample() -> tuple[str, dict]:
    """
    Returns (local_video_path, row) for the first metadata.csv entry whose
    video clip exists locally. Row contains: question, answer_options,
    correct_answer, qa_type, dataset, video_id.
    """
    with open(METADATA_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        # file_name is like "ego4d/video_id/clip.mp4"
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


# ── input builder ─────────────────────────────────────────────────────────────


def get_input_value(inputs, key: str):
    if isinstance(inputs, dict):
        return inputs.get(key)
    return getattr(inputs, key, None)


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    s_min, s_max = scores.min(), scores.max()
    if (s_max - s_min) <= 1e-8:
        return torch.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def get_visual_token_positions(input_ids: torch.Tensor, video_token_id: int = VIDEO_PAD_TOKEN_ID) -> torch.Tensor:
    return (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]


def get_question_text_positions(inputs, visual_pos: torch.Tensor) -> torch.Tensor:
    mm_token_type_ids = get_input_value(inputs, "mm_token_type_ids")
    if mm_token_type_ids is None:
        raise ValueError("Qwen inputs are missing `mm_token_type_ids`; cannot compute SparseVLM text raters.")

    seq_len = get_input_value(inputs, "input_ids").shape[1]
    seq_pos = torch.arange(seq_len, device=mm_token_type_ids.device)
    text_mask = mm_token_type_ids[0] == 0

    if visual_pos.numel() > 0:
        text_after_video = text_mask & (seq_pos > visual_pos[-1])
        if text_after_video.any():
            return seq_pos[text_after_video]

    return seq_pos[text_mask]


def decode_token_piece(tokenizer, token_id: int) -> tuple[str, str]:
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    piece = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return token, piece


def log_text_raters(
    processor,
    inputs,
    text_pos: torch.Tensor,
    token_scores: torch.Tensor,
    rater_mask: torch.Tensor,
    save: bool = False,
    top_k: int = 20,
):
    tokenizer = processor.tokenizer
    input_ids = get_input_value(inputs, "input_ids")[0, text_pos].tolist()
    score_values = token_scores.detach().float().cpu().tolist()
    rater_values = rater_mask.detach().bool().cpu().tolist()

    entries = []
    for seq_index, token_id, score, is_rater in zip(text_pos.tolist(), input_ids, score_values, rater_values):
        token, piece = decode_token_piece(tokenizer, int(token_id))
        entries.append(
            {
                "seq_index": int(seq_index),
                "token_id": int(token_id),
                "token": token,
                "piece": piece,
                "score": float(score),
                "is_rater": bool(is_rater),
            }
        )

    top_entries = sorted(entries, key=lambda item: item["score"], reverse=True)[:top_k]
    print(f"        Top {min(top_k, len(top_entries))} text-token scores:")
    for entry in top_entries:
        flag = "*" if entry["is_rater"] else " "
        print(
            f"          {flag} idx={entry['seq_index']:>4} "
            f"score={entry['score']:.4f} token={entry['token']!r} piece={entry['piece']!r}"
        )

    if save:
        payload = {
            "text_span": tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "top_k_printed": top_k,
            "entries": entries,
        }
        with open(TEXT_RATER_LOG_PATH, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"        Saved text rater log → {TEXT_RATER_LOG_PATH}")

def load_preprocessed_frames(file_name: str, n_frames: int) -> list[Image.Image] | None:
    """
    Load n_frames evenly spaced from pre-extracted frames directory (if it exists).
    Returns None if preprocessing hasn't been run yet.
    file_name is like "ego4d/scene_id/123_1205.mp4"
    """
    parts = Path(file_name)
    dataset = parts.parts[0]
    scene_id = parts.parts[1]
    clip_stem = parts.stem  # "123_1205"

    frames_dir = FRAMES_DIR / dataset / scene_id / clip_stem
    if not frames_dir.exists():
        return None

    jpgs = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not jpgs:
        return None

    # Sample n_frames evenly
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

    # Prefer pre-extracted frames (gaze-aligned) over live cv2 extraction
    frames, frame_gaze = load_preprocessed_frames(row["file_name"], n_frames)
    if frames is None:
        frames = extract_frames(video_path, n_frames)
        frame_gaze = None
        print(f"        frames extracted (cv2): {len(frames)}")


    prompt = (
        # "Describe in detail what is happening in this egocentric video. "
        # "What objects are visible? What is the person doing with their hands? "
        # "Where are they looking and why? What activity are they performing?"
        "Describe in detail: what is the person doing with their hands? "
    )

    # Pass frames as a list of PIL images — qwen_vl_utils accepts this directly,
    # bypassing torchvision.io.read_video entirely.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text",  "text": prompt},
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

    n_visual = (inputs.pixel_values_videos is not None and
                inputs.pixel_values_videos.shape) if hasattr(inputs, "pixel_values_videos") else "?"
    print(f"        input_ids shape: {inputs.input_ids.shape}  "
          f"pixel_values shape: {n_visual}")
    return inputs, frames, frame_gaze, output_log


# ── 5-component profiler ──────────────────────────────────────────────────────

class ComponentProfiler:
    """
    Hooks into 4 named sub-components plus tracks prefill vs decode separately.

    Components:
      vision_encoder  — model.visual          (ViT, fires once)
      token_merger    — model.visual.merger   (projector, fires once)
      prefill         — model.model           (1st call: all tokens)
      decode          — model.model           (subsequent calls: 1 token each)
    """

    def __init__(self, model):
        self.model = model
        self._timings: dict[str, list[float]] = {}
        self._handles = []
        self._decoder_call_count = 0

    def _make_hooks(self, name: str):
        state = {}

        def pre_hook(module, args):
            mps_sync()
            state["mem_before"] = mps_allocated_mb()
            state["t0"] = time.perf_counter()

        def post_hook(module, args, output):
            mps_sync()
            elapsed = time.perf_counter() - state["t0"]
            mem_delta = mps_allocated_mb() - state["mem_before"]
            self._timings.setdefault(name, []).append(elapsed)
            self._timings.setdefault(f"{name}_mem_mb", []).append(mem_delta)

        return pre_hook, post_hook

    def _make_decoder_hooks(self):
        """Separate hook that splits first call (prefill) from the rest (decode)."""
        state = {}

        def pre_hook(module, args):
            mps_sync()
            state["mem_before"] = mps_allocated_mb()
            state["t0"] = time.perf_counter()

        def post_hook(module, args, output):
            mps_sync()
            elapsed = time.perf_counter() - state["t0"]
            mem_delta = mps_allocated_mb() - state["mem_before"]
            key = "prefill" if self._decoder_call_count == 0 else "decode"
            self._decoder_call_count += 1
            self._timings.setdefault(key, []).append(elapsed)
            self._timings.setdefault(f"{key}_mem_mb", []).append(mem_delta)

        return pre_hook, post_hook

    def attach(self):
        visual = self.model.model.visual
        components = [("vision_encoder", visual)]
        if hasattr(visual, "merger"):
            components.append(("token_merger", visual.merger))

        for name, module in components:
            pre, post = self._make_hooks(name)
            self._handles.append(module.register_forward_pre_hook(pre))
            self._handles.append(module.register_forward_hook(post))

        pre, post = self._make_decoder_hooks()
        self._handles.append(self.model.model.language_model.register_forward_pre_hook(pre))
        self._handles.append(self.model.model.language_model.register_forward_hook(post))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._decoder_call_count = 0

    def report(self):
        print("\n── Component Timing (hook-based, MPS-synced) ───────────────────────")

        enc_t    = sum(self._timings.get("vision_encoder", [0]))
        merger_t = sum(self._timings.get("token_merger",   [0]))
        pre_t    = sum(self._timings.get("prefill",        [0]))
        dec_times = self._timings.get("decode", [])
        dec_t    = sum(dec_times)
        dec_steps = len(dec_times)
        dec_avg  = dec_t / dec_steps if dec_steps else 0

        # token_merger is a sub-component of vision_encoder — exclude from total
        grand_total = enc_t + pre_t + dec_t

        def row(label, t, extra=""):
            pct = t / grand_total * 100 if grand_total else 0
            mems = self._timings.get(f"{label}_mem_mb", [0])
            mem_avg = sum(mems) / len(mems)
            print(f"  {label:<22}  {t*1000:8.1f} ms  ({pct:5.1f}%)  mem Δ {mem_avg:+.1f} MB  {extra}")

        row("vision_encoder", enc_t)
        # merger indented as sub-component of encoder
        merger_pct = merger_t / enc_t * 100 if enc_t else 0
        mems = self._timings.get("token_merger_mem_mb", [0])
        print(f"    {'└─ token_merger':<20}  {merger_t*1000:8.1f} ms  ({merger_pct:5.1f}% of enc)  "
              f"mem Δ {sum(mems)/len(mems):+.1f} MB")
        row("prefill", pre_t)
        dec_extra = f"{dec_steps} steps  avg {dec_avg*1000:.1f} ms/tok"
        row("decode", dec_t, dec_extra)
        print(f"  {'TOTAL':<22}  {grand_total*1000:8.1f} ms")


# ── torch.profiler trace ──────────────────────────────────────────────────────

def run_with_torch_profiler(fn, trace_path: str):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
    ) as prof:
        fn()

    prof.export_chrome_trace(trace_path)
    print(f"\n── torch.profiler chrome trace ──────────────────────────────────────")
    print(f"  Saved:  {trace_path}")
    print(f"  Open:   chrome://tracing  or  ui.perfetto.dev")
    print("\n  Top 15 ops by self CPU time:")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=15, max_name_column_width=40,
    ))


# ── logger ───────────────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).parent / "profile_log.md"

def write_log(device, n_frames, results, timings, inputs, output_text, build_inputs_logs):
    enc_t     = sum(timings.get("vision_encoder", [0]))
    merger_t  = sum(timings.get("token_merger",   [0]))
    pre_t     = sum(timings.get("prefill",        [0]))
    dec_times = timings.get("decode", [])
    dec_t     = sum(dec_times)
    dec_steps = len(dec_times)
    dec_avg   = dec_t / dec_steps if dec_steps else 0
    total_t   = enc_t + pre_t + dec_t

    merger_pct = merger_t / enc_t * 100 if enc_t else 0

    input_ids = get_input_value(inputs, "input_ids")
    pixel_values_videos = get_input_value(inputs, "pixel_values_videos")
    pixel_shape = (
        str(tuple(pixel_values_videos.shape))
        if pixel_values_videos is not None
        else "n/a"
    )

    run_number = 1
    if LOG_FILE.exists():
        content = LOG_FILE.read_text()
        run_number = content.count("## Run ") + 1

    entry = f"""
## Run {run_number}
**Date:** {datetime.date.today()}
**Model:** {MODEL_ID}
**Device:** {device.upper()} | torch {torch.__version__}
**Frames:** {n_frames}
**Input shape:** input_ids {tuple(input_ids.shape)} | pixel_values {pixel_shape}
**Input logs:**
{build_inputs_logs}

### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | {enc_t*1000:.1f} | {enc_t/total_t*100:.1f}% | mem Δ {sum(timings.get("vision_encoder_mem_mb",[0]))/max(len(timings.get("vision_encoder_mem_mb",[1])),1):+.1f} MB |
| └─ token_merger | {merger_t*1000:.1f} | {merger_pct:.1f}% of enc | mem Δ {sum(timings.get("token_merger_mem_mb",[0]))/max(len(timings.get("token_merger_mem_mb",[1])),1):+.1f} MB |
| prefill | {pre_t*1000:.1f} | {pre_t/total_t*100:.1f}% | mem Δ {sum(timings.get("prefill_mem_mb",[0]))/max(len(timings.get("prefill_mem_mb",[1])),1):+.1f} MB |
| decode | {dec_t*1000:.1f} | {dec_t/total_t*100:.1f}% | {dec_steps} steps, avg {dec_avg*1000:.1f} ms/tok |
| **TOTAL** | **{total_t*1000:.1f}** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | {results.get("input_preprocessing", {}).get("time_s", 0)*1000:.1f} |
| probe_prefill | {results.get("probe_prefill", {}).get("time_s", 0)*1000:.1f} |
| end_to_end | {results.get("end_to_end", {}).get("time_s", 0)*1000:.1f} |

### Model Output
> {output_text[:].replace(chr(10), ' ')}...

---
"""

    with open(LOG_FILE, "a") as f:
        if run_number == 1:
            f.write("# VLM Profiling Log\n")
        f.write(entry)

    print(f"\nLogged to {LOG_FILE}  (run #{run_number})")


# ── gaze visualization ───────────────────────────────────────────────────────

def visualize_gaze_scores(pil_frames: list, frame_gaze: list, scores: torch.Tensor, sigma: float = 0.1, save: bool = False):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    T = len(pil_frames)
    fig, axes = plt.subplots(1, T, figsize=(5 * T, 5))
    if T == 1:
        axes = [axes]

    fig.suptitle(f"Gaze token scores  (σ={sigma})", fontsize=13)
    cmap = matplotlib.colormaps["hot"]
    T_scores = scores.shape[0]

    for t, (frame, gaze, ax) in enumerate(zip(pil_frames, frame_gaze, axes)):
        img = np.array(frame)
        h_img, w_img = img.shape[:2]

        t_score = min(round(t * (T_scores - 1) / max(len(pil_frames) - 1, 1)), T_scores - 1)
        score_np = scores[t_score].cpu().numpy()
        heatmap = cv2.resize(score_np, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        overlay = (0.45 * colored + 0.55 * img).astype(np.uint8)

        ax.imshow(overlay)

        if gaze is not None and gaze.get("gaze_x") is not None:
            ax.plot(gaze["gaze_x"] * w_img, gaze["gaze_y"] * h_img,
                    "c+", markersize=18, markeredgewidth=2.5)
            ax.plot(gaze["gaze_x"] * w_img, gaze["gaze_y"] * h_img,
                    "co", markersize=8, alpha=0.7)
            ax.set_xlabel(gaze.get("narration_text", "")[:60], fontsize=7)

        ax.set_title(f"frame {t}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    if save:
        out = Path(__file__).parent / "gaze_vis.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()

# ── pruning visualization ────────────────────────────────────────────────────

def visualize_pruning(
    pil_frames: list,
    mask: torch.Tensor,       # (N_visual,) bool — True = kept
    video_grid_thw: tuple,    # (T, H, W)
    save: bool = False,
    save_path: str = "pruning_vis.png",
):
    """
    For each frame: show original side-by-side with pruned version.
    Pruned tokens are faded out; kept tokens remain at full opacity.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    T, H, W = video_grid_thw
    n_frames = len(pil_frames)

    # reshape flat mask back to (T, H, W)
    mask_grid = mask.cpu().reshape(T, H, W).float()  # 1.0=keep, 0.0=prune

    fig, axes = plt.subplots(n_frames, 2, figsize=(10, 5 * n_frames))
    if n_frames == 1:
        axes = [axes]  # make iterable

    kept_total   = mask.sum().item()
    pruned_total = (~mask).sum().item()
    fig.suptitle(
        f"Token pruning  —  kept {kept_total} / {kept_total + pruned_total}  "
        f"({100 * kept_total / (kept_total + pruned_total):.1f}%)",
        fontsize=13,
    )

    for t, (frame, ax_row) in enumerate(zip(pil_frames, axes)):
        img = np.array(frame).astype(np.float32)
        h_img, w_img = img.shape[:2]

        ax_orig, ax_pruned = ax_row

        # ── left: original ───────────────────────────────────────────────
        ax_orig.imshow(frame)
        ax_orig.set_title(f"frame {t}  —  original", fontsize=9)
        ax_orig.axis("off")

        # ── right: pruned ────────────────────────────────────────────────
        # map temporal slice to this frame
        t_score = min(round(t * (T - 1) / max(n_frames - 1, 1)), T - 1)
        token_keep = mask_grid[t_score].numpy()  # (H, W)  1=keep 0=prune

        # upsample token mask to image resolution
        import cv2
        keep_map = cv2.resize(token_keep, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        # faded image: pruned regions → 20% brightness, kept → full
        fade = np.where(keep_map[..., None] > 0.5, img, img * 0.2).astype(np.uint8)

        # draw thin grid lines at token boundaries
        tile_h = h_img / H
        tile_w = w_img / W
        fade_lined = fade.copy()
        for r in range(1, H):
            y = int(r * tile_h)
            fade_lined[y, :] = [80, 80, 80]
        for c in range(1, W):
            x = int(c * tile_w)
            fade_lined[:, x] = [80, 80, 80]

        n_kept_frame = int(token_keep.sum())
        ax_pruned.imshow(fade_lined)
        ax_pruned.set_title(
            f"frame {t}  —  pruned  (kept {n_kept_frame}/{H*W} tokens)",
            fontsize=9,
        )
        ax_pruned.axis("off")

    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ── encoder feature capture ──────────────────────────────────────────────────

class FeatureCapture:
    """
    Forward-hook utility that captures the output of model.model.visual.merger
    — i.e. the post-merger visual features of shape (N_visual, hidden_dim),
    where N_visual = T * (H//2) * (W//2).

    Usage:
        with FeatureCapture(model) as cap:
            <run something that triggers the visual encoder>
        features = cap.features          # tensor or None
    """

    def __init__(self, model):
        self.model = model
        self.features = None
        self._handle = None

    def __enter__(self):
        def hook(module, args, output):
            self.features = output.detach()
        self._handle = self.model.model.visual.merger.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def capture_visual_features(model, inputs):
    """
    Run only the vision encoder + merger (no LLM) and return the post-merger
    visual features via a forward hook on the merger.

    Returns: tensor of shape (N_visual, hidden_dim).
    """
    with FeatureCapture(model) as cap:
        with torch.inference_mode():
            _ = model.model.visual(inputs.pixel_values_videos, inputs.video_grid_thw)
    return cap.features


# ── get gaze score ───────────────────────────────────────────────────────────

def get_gaze_score(inputs, frame_gaze, pil_frames, sigma: float = 0.1,
                   save_vis: bool = False, features: torch.Tensor | None = None):
    """
    Compute per-token gaze score and return a flat (N_visual,) tensor aligned
    with the LLM's visual token positions (post-merger).

    Steps:
      1. Build a Gaussian heatmap (T, H, W) over the pre-merger grid from gaze
         coordinates — one slice per temporal step.
      2. (optional) visualize the heatmap on the original frames.
      3. Avg-pool 2x2 spatially to (T, H//2, W//2) — post-merger size.
      4. Flatten and min-max normalize to [0, 1].
      5. (future) blend with feature-based signal (e.g. L2 norm of `features`).
    """
    import torch.nn.functional as F

    T, H, W = inputs.video_grid_thw[0].tolist()

    # ── 1. geometric Gaussian heatmap on the pre-merger grid ──────────────────
    hs = torch.arange(H)
    ws = torch.arange(W)
    grid_h, grid_w = torch.meshgrid(hs, ws, indexing="ij")
    token_x = (grid_w + 0.5) / W
    token_y = (grid_h + 0.5) / H

    gaze_points = []
    for g in frame_gaze:
        if g is not None and g.get("gaze_x") is not None:
            gaze_points.append((g["gaze_x"], g["gaze_y"]))
        else:
            gaze_points.append((0.5, 0.5))  # center fallback
    gaze_tensor = torch.tensor(gaze_points)  # (T, 2)

    scores = torch.zeros(T, H, W)
    for t in range(T):
        gx, gy = gaze_tensor[t]
        dist_sq = (token_x - gx) ** 2 + (token_y - gy) ** 2
        scores[t] = torch.exp(-dist_sq / (2 * sigma ** 2))

    # ── 2. visualize at full pre-merger resolution ────────────────────────────
    visualize_gaze_scores(pil_frames, frame_gaze, scores, sigma=sigma, save=save_vis)

    # ── 3. pool 2x2 to post-merger size, 4. flatten, normalize ────────────────
    scores_post = F.avg_pool2d(
        scores.unsqueeze(1).float(),   # (T, 1, H, W)
        kernel_size=2, stride=2,
    ).squeeze(1)                       # (T, H//2, W//2)
    scores_flat = scores_post.reshape(-1)  # (N_visual,)

    s_min, s_max = scores_flat.min(), scores_flat.max()
    if (s_max - s_min) > 1e-8:
        scores_flat = (scores_flat - s_min) / (s_max - s_min)

    # ── 5. future: blend with encoder feature norm ────────────────────────────
    # if features is not None:
    #     l2 = features.float().norm(dim=-1)                       # (N_visual,)
    #     l2_norm = (l2 - l2.min()) / (l2.max() - l2.min() + 1e-8)
    #     scores_flat = torch.maximum(scores_flat, l2_norm)        # "rescue" high-info peripheral tokens

    return scores_flat


def run_text_probe_prefill(model, inputs):
    with torch.inference_mode():
        return model(
            **inputs,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )


def get_text_scores_attention(
    model,
    processor,
    inputs,
    probe_layer: int = SPARSEVLM_PROBE_LAYER,
    save_debug: bool = False,
):
    """
    Single-pass SparseVLM-style text pruning signal.

    1. Run one full prefill on the dense visual prefix.
    2. Use hidden-state similarity to select "text raters" exactly like SparseVLM:
         m_v_t = softmax(V @ T^T, dim=text).mean(dim=visual)
         raters = text tokens whose score is above the mean.
    3. Use mean self-attention from those raters to visual tokens as the final
       visual-token pruning score.

    Returns a flat (N_visual,) score tensor aligned with post-merger video
    tokens in the Qwen sequence.
    """
    probe_outputs = run_text_probe_prefill(model, inputs)

    attn_layers = probe_outputs.attentions
    hidden_layers = probe_outputs.hidden_states
    resolved_layer = probe_layer if probe_layer >= 0 else len(attn_layers) + probe_layer
    if resolved_layer < 0 or resolved_layer >= len(attn_layers):
        raise ValueError(f"Invalid SparseVLM probe layer {probe_layer} for {len(attn_layers)} available layers.")

    input_ids = get_input_value(inputs, "input_ids")
    visual_pos = get_visual_token_positions(input_ids, model.config.video_token_id)
    text_pos = get_question_text_positions(inputs, visual_pos)

    if visual_pos.numel() == 0:
        raise ValueError("Could not find any <|video_pad|> tokens in the Qwen prompt sequence.")
    if text_pos.numel() == 0:
        raise ValueError("Could not find any question text tokens after the video block.")

    hidden_states = hidden_layers[resolved_layer].float()
    attn = attn_layers[resolved_layer].float()

    v_t = hidden_states[:, visual_pos, :]
    t_t = hidden_states[:, text_pos, :]

    m_v_t = torch.matmul(v_t, t_t.transpose(1, 2))
    m_v_t = torch.softmax(m_v_t, dim=2).mean(dim=1)
    rater_mask = m_v_t > m_v_t.mean(dim=1, keepdim=True)
    if not rater_mask.any():
        top_idx = torch.argmax(m_v_t, dim=1, keepdim=True)
        rater_mask = torch.zeros_like(m_v_t, dtype=torch.bool)
        rater_mask.scatter_(1, top_idx, True)

    text_rater_scores = m_v_t.squeeze(0)
    selected_text_pos = text_pos[rater_mask[0]]
    attn_mean = attn.mean(dim=1)
    relation_vis_text = attn_mean[:, selected_text_pos, :][:, :, visual_pos].mean(dim=1).squeeze(0)
    text_scores = normalize_scores(relation_vis_text)

    log_text_raters(
        processor,
        inputs,
        text_pos,
        text_rater_scores,
        rater_mask.squeeze(0),
        save=save_debug,
    )

    print(
        f"        SparseVLM text raters: {selected_text_pos.numel()}/{text_pos.numel()} "
        f"(layer {resolved_layer}, keep ratio {SPARSEVLM_KEEP_RATIO:.2f})"
    )
    return text_scores


# ── build pruning function ────────────────────────────────────────────────────

def get_pruning_scores(gaze_pruning, text_pruning, gaze_scores, text_scores, alpha):
    # weighted sum function
    if gaze_pruning and text_pruning:
        return alpha * gaze_scores + (1 - alpha) * text_scores
    elif gaze_pruning:
        return gaze_scores
    elif text_pruning:
        return text_scores


    return None

# ── pruning mask + application ────────────────────────────────────────────────

def get_pruning_mask(prune_scores, threshold: float | None = None, keep_ratio: float | None = None):
    if keep_ratio is not None:
        n_total = prune_scores.numel()
        n_keep = max(1, min(n_total, int(round(n_total * keep_ratio))))
        mask = torch.zeros_like(prune_scores, dtype=torch.bool)
        topk = torch.topk(prune_scores, n_keep).indices
        mask[topk] = True
        return mask

    if threshold is None:
        raise ValueError("Either `threshold` or `keep_ratio` must be provided to get_pruning_mask().")

    return prune_scores > threshold


def apply_pruning_mask(inputs, mask):
    """
    Apply pruning mask by zeroing out attention for pruned visual tokens.

    Pruned tokens are hidden from the LLM by setting their positions to 0 in
    attention_mask.  This keeps input_ids, pixel_values_videos, video_grid_thw
    and rope_deltas all intact — the vision encoder still runs on all patches,
    but the language model cannot attend to the suppressed visual positions.

    mask : (N_visual,) bool  —  True = keep, False = prune
           N_visual = T * (H//2) * (W//2)  (post-merger token count)
    """
    VIDEO_PAD_TOKEN = 151656  # <|video_pad|> in Qwen2-VL tokenizer

    input_ids  = inputs.input_ids[0]
    visual_pos = (input_ids == VIDEO_PAD_TOKEN).nonzero(as_tuple=True)[0]

    n_found    = len(visual_pos)
    n_expected = mask.shape[0]

    # ── diagnostic: verify token ID is correct ────────────────────────────────
    if n_found == 0:
        from collections import Counter
        id_counts = Counter(input_ids.tolist())
        top = id_counts.most_common(10)
        print(f"  [WARN] VIDEO_PAD_TOKEN {VIDEO_PAD_TOKEN} not found in input_ids!")
        print(f"  [WARN] Top-10 token IDs by frequency: {top}")
        print(f"  [WARN] Skipping pruning to avoid sequence corruption.")
        return inputs

    if n_found != n_expected:
        print(f"  [WARN] visual_pos count ({n_found}) != mask size ({n_expected}) — skipping pruning")
        return inputs

    # ── suppress pruned visual positions via attention_mask ───────────────────
    pruned_pos = visual_pos[~mask]

    if inputs.get("attention_mask") is not None:
        attn = inputs.attention_mask.clone()
    else:
        attn = torch.ones(1, len(input_ids), dtype=torch.long, device=input_ids.device)

    attn[0, pruned_pos] = 0
    inputs.attention_mask = attn

    n_kept  = mask.sum().item()
    n_total = mask.shape[0]
    print(f"        pruned: kept {n_kept}/{n_total} visual tokens ({100*n_kept/n_total:.1f}%)")
    return inputs


def build_pruned_sequence_inputs(inputs, visual_keep_mask, video_token_id: int = VIDEO_PAD_TOKEN_ID):
    input_ids = get_input_value(inputs, "input_ids")
    attention_mask = get_input_value(inputs, "attention_mask")
    mm_token_type_ids = get_input_value(inputs, "mm_token_type_ids")
    visual_pos = get_visual_token_positions(input_ids, video_token_id)

    if visual_pos.numel() != visual_keep_mask.numel():
        raise ValueError(
            f"visual_pos count ({visual_pos.numel()}) != mask size ({visual_keep_mask.numel()})"
        )

    seq_keep_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    seq_keep_mask[visual_pos[~visual_keep_mask]] = False

    pruned_inputs = {
        "input_ids": input_ids[:, seq_keep_mask].clone(),
        "attention_mask": attention_mask[:, seq_keep_mask].clone()
        if attention_mask is not None
        else torch.ones(1, int(seq_keep_mask.sum().item()), dtype=torch.long, device=input_ids.device),
        "mm_token_type_ids": mm_token_type_ids[:, seq_keep_mask].clone() if mm_token_type_ids is not None else None,
    }

    n_kept = int(visual_keep_mask.sum().item())
    print(f"        dropped: kept {n_kept}/{visual_keep_mask.numel()} visual tokens ({100*n_kept/visual_keep_mask.numel():.1f}%)")
    return pruned_inputs, seq_keep_mask


def compute_pruned_position_ids(model, inputs, seq_keep_mask):
    input_ids = get_input_value(inputs, "input_ids")
    attention_mask = get_input_value(inputs, "attention_mask")
    mm_token_type_ids = get_input_value(inputs, "mm_token_type_ids")
    image_grid_thw = get_input_value(inputs, "image_grid_thw")
    video_grid_thw = get_input_value(inputs, "video_grid_thw")

    if mm_token_type_ids is None:
        raise ValueError("Qwen inputs are missing `mm_token_type_ids`; cannot preserve multimodal RoPE after pruning.")

    position_ids, _ = model.model.get_rope_index(
        input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )

    pruned_position_ids = position_ids[:, :, seq_keep_mask].clone()
    pruned_seq_len = int(seq_keep_mask.sum().item())
    rope_delta = (pruned_position_ids.max() + 1 - pruned_seq_len).view(1, 1)
    return pruned_position_ids, rope_delta.to(input_ids.device)


def build_pruned_inputs_embeds(model, pruned_input_ids, pruned_visual_features):
    inputs_embeds = model.model.get_input_embeddings()(pruned_input_ids)
    video_mask = pruned_input_ids == model.config.video_token_id
    if int(video_mask.sum().item()) != pruned_visual_features.shape[0]:
        raise ValueError(
            "Number of kept <|video_pad|> placeholders does not match the number of kept visual features: "
            f"{int(video_mask.sum().item())} vs {pruned_visual_features.shape[0]}"
        )

    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[video_mask] = pruned_visual_features.to(inputs_embeds.device, inputs_embeds.dtype)
    return inputs_embeds


def get_next_decode_position_ids(attention_mask, rope_deltas):
    next_pos = attention_mask.long().sum(dim=-1, keepdim=True) - 1
    next_pos = next_pos + rope_deltas.to(attention_mask.device)
    return next_pos.unsqueeze(0).expand(3, -1, -1)


def manual_generate_with_pruned_visual(
    model,
    original_inputs,
    pruned_inputs,
    visual_keep_mask,
    position_ids,
    rope_deltas,
    max_new_tokens: int = 256,
    min_new_tokens: int = 128,
):
    with torch.inference_mode():
        visual_outputs = model.model.visual(
            get_input_value(original_inputs, "pixel_values_videos"),
            get_input_value(original_inputs, "video_grid_thw"),
        )
        pruned_visual_features = visual_outputs.pooler_output[visual_keep_mask]
        inputs_embeds = build_pruned_inputs_embeds(model, pruned_inputs["input_ids"], pruned_visual_features)

        lm_outputs = model.model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=pruned_inputs["attention_mask"],
            position_ids=position_ids.to(inputs_embeds.device),
            use_cache=True,
            return_dict=True,
        )
        logits = model.lm_head(lm_outputs.last_hidden_state[:, -1:, :])

        generated_tokens = []
        attention_mask = pruned_inputs["attention_mask"].clone()
        past_key_values = lm_outputs.past_key_values
        eos_token_id = model.generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        for step in range(max_new_tokens):
            generated_tokens.append(next_token)

            if eos_token_id is not None and step + 1 >= min_new_tokens and torch.all(next_token == eos_token_id):
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones(attention_mask.size(0), 1, dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=1,
            )
            next_position_ids = get_next_decode_position_ids(attention_mask, rope_deltas)
            next_embeds = model.model.get_input_embeddings()(next_token)

            lm_outputs = model.model.language_model(
                input_ids=None,
                inputs_embeds=next_embeds,
                attention_mask=attention_mask,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = lm_outputs.past_key_values
            logits = model.lm_head(lm_outputs.last_hidden_state[:, -1:, :])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        if generated_tokens:
            generated_tokens = torch.cat(generated_tokens, dim=1)
            return torch.cat([pruned_inputs["input_ids"], generated_tokens], dim=1)
        return pruned_inputs["input_ids"]


def prepare_pruned_generation_state(model, inputs, visual_keep_mask):
    pruned_inputs, seq_keep_mask = build_pruned_sequence_inputs(inputs, visual_keep_mask, model.config.video_token_id)
    position_ids, rope_deltas = compute_pruned_position_ids(model, inputs, seq_keep_mask)
    return {
        "pruned_inputs": pruned_inputs,
        "position_ids": position_ids,
        "rope_deltas": rope_deltas,
        "visual_keep_mask": visual_keep_mask,
    }



# ── main ──────────────────────────────────────────────────────────────────────

def main(
    enable_trace: bool = True,
    n_frames: int = 4,
    gaze_pruning = False,
    text_pruning = False,
    save_vis: bool = False,
    save_text_raters: bool = False,
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  |  torch {torch.__version__}  |  frames: {n_frames}")

    results = {}

    with timed_block("model_load", results):
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "attn_implementation": "eager",
        }
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, **model_kwargs)
        model.eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"Model loaded in {results['model_load']['time_s']:.2f}s  "
          f"({results['model_load']['mem_peak_mb']:.0f} MB MPS)")

    with timed_block("input_preprocessing", results):
        inputs, pil_frames, frame_gaze, build_inputs_logs = build_inputs(processor, n_frames)

    # ── score computation ─────────────────────────────────────────────────────
    gaze_scores = None
    if gaze_pruning:
        # 1. capture post-merger encoder features (used later for L2-norm blend)
        features = capture_visual_features(model, inputs)
        # 2. compute gaze score — already returned post-merger flat (N_visual,)
        gaze_scores = get_gaze_score(
            inputs, frame_gaze, pil_frames,
            save_vis=save_vis,
            features=features,
        )

    text_scores = None
    if text_pruning:
        with timed_block("probe_prefill", results):
            text_scores = get_text_scores_attention(
                model,
                processor,
                inputs,
                save_debug=save_text_raters,
            )

    # ── mask + visualize + apply ──────────────────────────────────────────────
    pruned_state = None
    if gaze_pruning or text_pruning:
        prune_scores = get_pruning_scores(
            gaze_pruning, text_pruning, gaze_scores, text_scores, alpha=0.5,
        )
        if text_pruning:
            pruning_mask = get_pruning_mask(prune_scores, keep_ratio=SPARSEVLM_KEEP_RATIO)
        else:
            pruning_mask = get_pruning_mask(prune_scores, threshold=GAZE_PRUNE_THRESHOLD)

        if save_vis:
            T_vis, H_vis, W_vis = inputs.video_grid_thw[0].tolist()
            visualize_pruning(
                pil_frames, pruning_mask,
                (T_vis, H_vis // 2, W_vis // 2),
                save=True,
            )

        pruned_state = prepare_pruned_generation_state(model, inputs, pruning_mask)


    # warmup — MPS compiles Metal shaders on first forward; exclude from timing
    print("\nWarming up...")
    with torch.inference_mode():
        if pruned_state is not None:
            _ = manual_generate_with_pruned_visual(
                model,
                inputs,
                pruned_state["pruned_inputs"],
                pruned_state["visual_keep_mask"],
                pruned_state["position_ids"],
                pruned_state["rope_deltas"],
                max_new_tokens=16,
                min_new_tokens=8,
            )
        else:
            _ = model.generate(**inputs, max_new_tokens=16, min_new_tokens=8)
    mps_sync()
    print("Warmup done.\n")

    profiler = ComponentProfiler(model)
    profiler.attach()

    with torch.inference_mode():
        with timed_block("end_to_end", results):
            if pruned_state is not None:
                generated_ids = manual_generate_with_pruned_visual(
                    model,
                    inputs,
                    pruned_state["pruned_inputs"],
                    pruned_state["visual_keep_mask"],
                    pruned_state["position_ids"],
                    pruned_state["rope_deltas"],
                    max_new_tokens=256,
                    min_new_tokens=128,
                )
            else:
                generated_ids = model.generate(**inputs, max_new_tokens=256, min_new_tokens=128)

    profiler.detach()
    profiler.report()

    print("\n── End-to-end ───────────────────────────────────────────────────────")
    for k in ["input_preprocessing", "probe_prefill", "end_to_end"]:
        r = results.get(k, {})
        print(f"  {k:<25}  {r.get('time_s', 0)*1000:8.1f} ms")

    if enable_trace:
        profiler2 = ComponentProfiler(model)
        profiler2.attach()
        trace_path = str(Path(__file__).parent / "vlm_trace.json")
        with torch.inference_mode():
            run_with_torch_profiler(
                (
                    lambda: manual_generate_with_pruned_visual(
                        model,
                        inputs,
                        pruned_state["pruned_inputs"],
                        pruned_state["visual_keep_mask"],
                        pruned_state["position_ids"],
                        pruned_state["rope_deltas"],
                        max_new_tokens=128,
                        min_new_tokens=0,
                    )
                ) if pruned_state is not None else
                (lambda: model.generate(**inputs, max_new_tokens=128))
                ,
                trace_path=trace_path,
            )
        profiler2.detach()

    input_ids_for_trim = pruned_state["pruned_inputs"]["input_ids"] if pruned_state is not None else inputs.input_ids
    trimmed = [out[len(inp):] for inp, out in zip(input_ids_for_trim, generated_ids)]
    output  = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    print(f"\nModel output: {output[0][:200]}")

    write_log(
        device=device,
        n_frames=n_frames,
        results=results,
        timings=profiler._timings,
        inputs=pruned_state["pruned_inputs"] if pruned_state is not None else inputs,
        output_text=output[0],
        build_inputs_logs=build_inputs_logs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--gaze-prune", action="store_true")
    parser.add_argument("--text-prune", action="store_true")
    parser.add_argument("--frames", type=int, default=4,
                        help="Number of frames to sample from the clip (default 4)")
    parser.add_argument("--save-vis", action="store_true",
                        help="Save gaze visualization to gaze_vis.png instead of showing")
    parser.add_argument("--save-text-raters", action="store_true",
                        help="Save SparseVLM text-token rater scores to text_raters.json")
    args = parser.parse_args()
    main(
        enable_trace=not args.no_trace,
        n_frames=args.frames,
        gaze_pruning=args.gaze_prune,
        text_pruning=args.text_prune,
        save_vis=args.save_vis,
        save_text_raters=args.save_text_raters,
    )
