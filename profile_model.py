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
        "Describe in detail what is happening in this egocentric video. "
        "What objects are visible? What is the person doing with their hands? "
        "Where are they looking and why? What activity are they performing?"
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

    pixel_shape = (
        str(tuple(inputs.pixel_values_videos.shape))
        if hasattr(inputs, "pixel_values_videos") and inputs.pixel_values_videos is not None
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
**Input shape:** input_ids {tuple(inputs.input_ids.shape)} | pixel_values {pixel_shape}
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


# ── get gaze score ───────────────────────────────────────────────────────────

def get_gaze_score(inputs, frame_gaze, pil_frames, save_vis = False):
    T, H, W = inputs.video_grid_thw[0].tolist()
    hs = torch.arange(H)
    ws = torch.arange(W)
    grid_h, grid_w = torch.meshgrid(hs, ws, indexing="ij")  # both (H, W)

    token_x = (grid_w + 0.5) / W  # (H, W)  normalized col centers
    token_y = (grid_h + 0.5) / H  # (H, W)  normalized row centers
    gaze_points = []
    for g in frame_gaze:
        if g is not None:
            gaze_points.append((g["gaze_x"], g["gaze_y"]))
        else:
            gaze_points.append((0.5, 0.5))  # center fallback

    gaze_tensor = torch.tensor(gaze_points)  # (T, 2)

    # compute scores: one (H, W) heatmap per temporal slice
    sigma = 0.1
    scores = torch.zeros(T, H, W)
    for t in range(T):
        gx, gy = gaze_tensor[t]
        dist_sq = (token_x - gx) ** 2 + (token_y - gy) ** 2
        scores[t] = torch.exp(-dist_sq / (2 * sigma ** 2))

    visualize_gaze_scores(pil_frames, frame_gaze, scores, sigma=sigma, save=save_vis)
    return scores


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

def get_pruning_mask(prune_scores, threshold):
    pruning_mask = prune_scores > threshold
    return pruning_mask


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


# def get_text_scores_attention(model, inputs):
#     generated_ids, attn_scores = model.generate(**inputs, max_new_tokens=1, output_attentions=True, return_dict_in_generate=True)


# def get_text_scores_mini_crossattn(model, inputs):



# ── main ──────────────────────────────────────────────────────────────────────

def main(enable_trace: bool = True, n_frames: int = 4, gaze_pruning = False, text_pruning = False, save_vis: bool = False):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  |  torch {torch.__version__}  |  frames: {n_frames}")

    results = {}

    with timed_block("model_load", results):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"Model loaded in {results['model_load']['time_s']:.2f}s  "
          f"({results['model_load']['mem_peak_mb']:.0f} MB MPS)")

    with timed_block("input_preprocessing", results):
        inputs, pil_frames, frame_gaze, build_inputs_logs = build_inputs(processor, n_frames)
        
    gaze_scores = None
    if gaze_pruning:
        import torch.nn.functional as F
        gaze_scores = get_gaze_score(inputs, frame_gaze, pil_frames, save_vis)
        # video_grid_thw is pre-merger (T, H, W); input_ids has post-merger tokens (T, H//2, W//2)
        # pool 2x2 spatially to align with the token merger inside the model
        T, H, W = gaze_scores.shape
        gaze_scores = F.avg_pool2d(
            gaze_scores.unsqueeze(1).float(),  # (T, 1, H, W)
            kernel_size=2, stride=2
        ).squeeze(1)                            # (T, H//2, W//2)
        gaze_scores = gaze_scores.reshape(-1)   # (T * H//2 * W//2,) matches visual_pos

    text_scores = None
    # if text_pruning:
    #     text_scores = get_text_scores()
    #     text_scores = text_scores.float()


    if gaze_pruning or text_pruning:
        prune_scores = get_pruning_scores(gaze_pruning, text_pruning, gaze_scores, text_scores, alpha=0.5)
        pruning_mask = get_pruning_mask(prune_scores, threshold=0.3)
        inputs = apply_pruning_mask(inputs, pruning_mask)
        # video_grid_thw is pre-merger (T, H, W); pruning operates post-merger (T, H//2, W//2)
        T_vis, H_vis, W_vis = inputs.video_grid_thw[0].tolist()
        if save_vis:
            visualize_pruning(pil_frames, pruning_mask, (T_vis, H_vis // 2, W_vis // 2), save=True)


    # warmup — MPS compiles Metal shaders on first forward; exclude from timing
    print("\nWarming up...")
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=16, min_new_tokens=8)
    mps_sync()
    print("Warmup done.\n")

    profiler = ComponentProfiler(model)
    profiler.attach()

    with torch.inference_mode():
        with timed_block("end_to_end", results):
            generated_ids = model.generate(**inputs, max_new_tokens=256, min_new_tokens=128)

    profiler.detach()
    profiler.report()

    print("\n── End-to-end ───────────────────────────────────────────────────────")
    for k in ["input_preprocessing", "end_to_end"]:
        r = results.get(k, {})
        print(f"  {k:<25}  {r.get('time_s', 0)*1000:8.1f} ms")

    if enable_trace:
        profiler2 = ComponentProfiler(model)
        profiler2.attach()
        trace_path = str(Path(__file__).parent / "vlm_trace.json")
        with torch.inference_mode():
            run_with_torch_profiler(
                lambda: model.generate(**inputs, max_new_tokens=128),
                trace_path=trace_path,
            )
        profiler2.detach()

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output  = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    print(f"\nModel output: {output[0][:200]}")

    write_log(
        device=device,
        n_frames=n_frames,
        results=results,
        timings=profiler._timings,
        inputs=inputs,
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
    args = parser.parse_args()
    main(enable_trace=not args.no_trace, n_frames=args.frames, gaze_pruning=args.gaze_prune, text_pruning=args.text_prune, save_vis=args.save_vis)
