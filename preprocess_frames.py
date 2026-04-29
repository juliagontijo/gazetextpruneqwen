"""
preprocess_frames.py

One-time preprocessing: extracts frames at narration timestamps from video clips
and saves them as JPEGs alongside a companion JSON with gaze metadata.

Output layout:
  data/EgoGazeVQA_full/frames/{dataset}/{scene_id}/{start}_{end}/{frame}.jpg
  data/EgoGazeVQA_full/frames/{dataset}/{scene_id}/{start}_{end}/gaze.json

Usage:
  python preprocess_frames.py --dataset ego4d
  python preprocess_frames.py --dataset egoexo
  python preprocess_frames.py --dataset ego4d --force   # re-extract even if already done
"""

import argparse
import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm

BASE_DIR = Path(__file__).parent / "data" / "EgoGazeVQA_full"
FRAMES_DIR = BASE_DIR / "frames"


def parse_clip_name(mp4_name: str) -> tuple[int, int]:
    stem = Path(mp4_name).stem
    start, end = stem.split("_")
    return int(start), int(end)


def extract_narration_frames(
    video_path: Path,
    target_frames: list[int],
    out_dir: Path,
    force: bool,
) -> list[int]:
    """
    Extract specific frame numbers from a video file.
    Returns list of frame numbers successfully written.
    """
    missing = [f for f in target_frames if not (out_dir / f"{f}.jpg").exists()]
    if not missing and not force:
        return target_frames

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_start, clip_end = parse_clip_name(video_path.name)
    written = []

    for abs_frame in sorted(target_frames):
        out_path = out_dir / f"{abs_frame}.jpg"
        if out_path.exists() and not force:
            written.append(abs_frame)
            continue

        # abs_frame is the global frame index in the original video;
        # within this clip it's at position (abs_frame - clip_start)
        clip_offset = abs_frame - clip_start
        if clip_offset < 0:
            continue
        # end_frame in clip name is exclusive; clamp to last readable frame
        if clip_offset >= total_frames:
            clip_offset = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_offset)
        ret, frame = cap.read()
        if not ret:
            # One more try at the penultimate frame (decoder sometimes can't seek to last)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, clip_offset - 1))
            ret, frame = cap.read()
        if not ret:
            print(f"  [WARN] Could not read frame {abs_frame} from {video_path.name}")
            continue

        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(abs_frame)

    cap.release()
    return written


def process_dataset(dataset: str, force: bool) -> None:
    json_path = BASE_DIR / f"{dataset}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Narration JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    videos_dir = BASE_DIR / dataset
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    total_clips = 0
    total_frames = 0
    skipped_clips = 0

    for scene_id, scene_data in tqdm(data.items(), desc=f"Scenes ({dataset})"):
        narrations = scene_data.get("narrations", [])
        if not narrations:
            continue

        # Index narrations by frame for fast lookup
        narr_by_frame: dict[int, dict] = {}
        for n in narrations:
            frame = n.get("timestamp_frame")
            if frame is not None:
                narr_by_frame[int(frame)] = n

        scene_video_dir = videos_dir / scene_id
        if not scene_video_dir.exists():
            continue

        clips = sorted(scene_video_dir.glob("*.mp4"))
        for clip_path in clips:
            try:
                clip_start, clip_end = parse_clip_name(clip_path.name)
            except ValueError:
                print(f"  [WARN] Unexpected clip name: {clip_path.name}")
                continue

            # Find narrations whose timestamp falls within this clip
            clip_narrations = {
                frame: narr
                for frame, narr in narr_by_frame.items()
                if clip_start <= frame <= clip_end
            }

            if not clip_narrations:
                skipped_clips += 1
                continue

            out_dir = FRAMES_DIR / dataset / scene_id / clip_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)

            gaze_json_path = out_dir / "gaze.json"
            if gaze_json_path.exists() and not force:
                # Already processed — just count
                total_clips += 1
                total_frames += len(list(out_dir.glob("*.jpg")))
                continue

            written = extract_narration_frames(
                clip_path,
                list(clip_narrations.keys()),
                out_dir,
                force,
            )

            # Build companion gaze.json: {frame_number: {gaze_x, gaze_y, confidence, narration_text}}
            gaze_data = {}
            for frame in written:
                narr = clip_narrations[frame]
                gaze_info = narr.get("gaze_info", {})
                gaze_data[str(frame)] = {
                    "gaze_x": gaze_info.get("gaze_x"),
                    "gaze_y": gaze_info.get("gaze_y"),
                    "confidence": gaze_info.get("confidence"),
                    "narration_text": narr.get("narration_text"),
                    "timestamp_sec": narr.get("timestamp_sec"),
                }

            with open(gaze_json_path, "w") as f:
                json.dump(gaze_data, f, indent=2)

            total_clips += 1
            total_frames += len(written)

    print(f"\nDone: {total_clips} clips, {total_frames} frames written ({skipped_clips} clips had no matching narrations)")
    print(f"Output: {FRAMES_DIR / dataset}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess EgoGazeVQA frames at narration timestamps")
    parser.add_argument("--dataset", required=True, choices=["ego4d", "egoexo", "egtea"], help="Dataset to preprocess")
    parser.add_argument("--force", action="store_true", help="Re-extract frames even if output already exists")
    args = parser.parse_args()

    process_dataset(args.dataset, args.force)


if __name__ == "__main__":
    main()
