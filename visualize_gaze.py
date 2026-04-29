"""
visualize_gaze.py

Visualize gaze-based token scores overlaid on video frames.

Usage:
  python visualize_gaze.py --frames 4 --sigma 0.1
  python visualize_gaze.py --frames 4 --sigma 0.05 --save
"""

import argparse
import json
import csv
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

DATA_DIR   = Path(__file__).parent / "data" / "EgoGazeVQA_full"
FRAMES_DIR = DATA_DIR / "frames"
METADATA_CSV = DATA_DIR / "metadata.csv"
MODEL_ID   = "Qwen/Qwen2-VL-2B-Instruct"


def load_sample_with_frames(n_frames: int):
    with open(METADATA_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        local_path = DATA_DIR / row["file_name"]
        if not local_path.exists():
            continue

        parts = Path(row["file_name"])
        frames_dir = FRAMES_DIR / parts.parts[0] / parts.parts[1] / parts.stem
        if not frames_dir.exists():
            continue

        jpgs = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        if not jpgs:
            continue

        n = len(jpgs)
        indices = [int(i * (n - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
        selected = [jpgs[min(i, n - 1)] for i in indices]

        gaze_data = {}
        gaze_json = frames_dir / "gaze.json"
        if gaze_json.exists():
            with open(gaze_json) as f:
                gaze_data = json.load(f)

        pil_frames = [Image.open(p).convert("RGB") for p in selected]
        frame_gaze = [gaze_data.get(p.stem) for p in selected]

        print(f"Clip: {row['file_name']}")
        return pil_frames, frame_gaze, row

    raise FileNotFoundError("No preprocessed clip found.")


def compute_gaze_scores(frame_gaze: list, video_grid_thw, sigma: float) -> torch.Tensor:
    T, H, W = video_grid_thw

    hs = torch.arange(H)
    ws = torch.arange(W)
    grid_h, grid_w = torch.meshgrid(hs, ws, indexing="ij")
    token_x = (grid_w + 0.5) / W  # (H, W)
    token_y = (grid_h + 0.5) / H  # (H, W)

    gaze_points = []
    for g in frame_gaze:
        if g is not None and g.get("gaze_x") is not None:
            gaze_points.append((g["gaze_x"], g["gaze_y"]))
        else:
            gaze_points.append((0.5, 0.5))

    scores = torch.zeros(T, H, W)
    for t in range(T):
        gx, gy = gaze_points[t]
        dist_sq = (token_x - gx) ** 2 + (token_y - gy) ** 2
        scores[t] = torch.exp(-dist_sq / (2 * sigma ** 2))

    return scores  # (T, H, W)


def visualize(pil_frames: list, frame_gaze: list, scores: torch.Tensor, sigma: float, save: bool):
    T = len(pil_frames)
    fig, axes = plt.subplots(1, T, figsize=(5 * T, 5))
    if T == 1:
        axes = [axes]

    fig.suptitle(f"Gaze token scores  (σ={sigma})", fontsize=13)

    colormap = cm.get_cmap("hot")

    for t, (frame, gaze, ax) in enumerate(zip(pil_frames, frame_gaze, axes)):
        img = np.array(frame)
        h_img, w_img = img.shape[:2]

        # upsample score heatmap to image resolution
        score_np = scores[t].numpy()  # (H, W)
        heatmap = cv2.resize(score_np, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        colored = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3)

        # blend: 45% heatmap, 55% original image
        overlay = (0.45 * colored + 0.55 * img).astype(np.uint8)

        ax.imshow(overlay)

        # gaze ground-truth dot
        if gaze is not None and gaze.get("gaze_x") is not None:
            gx_px = gaze["gaze_x"] * w_img
            gy_px = gaze["gaze_y"] * h_img
            ax.plot(gx_px, gy_px, "c+", markersize=18, markeredgewidth=2.5, label="gaze GT")
            ax.plot(gx_px, gy_px, "co", markersize=8, alpha=0.7)
            narr = gaze.get("narration_text", "")
            ax.set_xlabel(narr[:60], fontsize=7)
        else:
            ax.set_xlabel("(no gaze)", fontsize=7)

        conf = gaze.get("confidence") if gaze else None
        conf_str = f"  conf={conf:.2f}" if conf is not None else ""
        ax.set_title(f"frame {t}{conf_str}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    if save:
        out = Path(__file__).parent / "gaze_vis.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--save", action="store_true", help="Save to gaze_vis.png instead of showing")
    args = parser.parse_args()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    pil_frames, frame_gaze, row = load_sample_with_frames(args.frames)

    # run through processor just to get video_grid_thw
    messages = [{"role": "user", "content": [{"type": "video", "video": pil_frames}, {"type": "text", "text": "Describe."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    thw = inputs.video_grid_thw[0].tolist()  # [T, H, W]
    T, H, W = thw
    print(f"video_grid_thw: T={T}  H={H}  W={W}  → {T*H*W} tokens")

    scores = compute_gaze_scores(frame_gaze, (T, H, W), args.sigma)

    print(f"Score stats: min={scores.min():.3f}  max={scores.max():.3f}  mean={scores.mean():.3f}")
    for thresh in [0.9, 0.7, 0.5, 0.3]:
        kept = (scores > thresh).sum().item()
        print(f"  tokens > {thresh}: {kept}/{T*H*W}  ({100*kept/(T*H*W):.1f}%)")

    visualize(pil_frames, frame_gaze, scores, args.sigma, args.save)


if __name__ == "__main__":
    main()
