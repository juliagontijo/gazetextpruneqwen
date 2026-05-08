"""
Precompute caption ground-truth strings from per-frame Ego4D narrations.

For every metadata row that has local frame data, read its `gaze.json`,
concatenate the per-frame `narration_text` fields into a single ground-truth
string, and write the result to a JSON cache.

Output:
    data/EgoGazeVQA_full/caption_ground_truth.json
    {
      "ego4d/<scene>/<clip>.mp4": "The camera wearer serves food. Another person ...",
      ...
    }

Usage:
    python precompute_captions.py
    python precompute_captions.py --frames 4              # match the slot count used at inference
    python precompute_captions.py --output custom.json    # write elsewhere
"""
import argparse
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "EgoGazeVQA_full"
METADATA_CSV = DATA_DIR / "metadata.csv"
FRAMES_DIR = DATA_DIR / "frames"
DEFAULT_OUT = DATA_DIR / "caption_ground_truth.json"


def clean_narration(text: str) -> str:
    """Strip Ego4D speaker codes (#C, #O) and tidy whitespace."""
    text = text.strip()
    text = text.replace("#C C", "The camera wearer")
    text = text.replace("#C", "The camera wearer")
    text = text.replace("#O", "Another person")
    return " ".join(text.split())


def build_ground_truth(file_name: str, n_frames: int | None) -> str | None:
    """
    Read gaze.json for the given clip, sample n_frames evenly (matching
    profile_forked_model.load_preprocessed_frames), concatenate the
    cleaned narration_text fields, dedupe consecutive duplicates.
    Returns None if the clip has no frames or no narrations.
    """
    parts = Path(file_name)
    dataset = parts.parts[0]
    scene_id = parts.parts[1]
    clip_stem = parts.stem

    frames_dir = FRAMES_DIR / dataset / scene_id / clip_stem
    if not frames_dir.exists():
        return None

    jpgs = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not jpgs:
        return None

    gaze_json = frames_dir / "gaze.json"
    if not gaze_json.exists():
        return None
    with open(gaze_json) as f:
        gaze_data = json.load(f)

    # Sample the SAME n_frames slots that load_preprocessed_frames will pick.
    # If n_frames is None, use ALL frames in the directory (richest GT).
    if n_frames is None:
        selected = jpgs
    else:
        n = len(jpgs)
        indices = [int(i * (n - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
        selected = [jpgs[min(i, n - 1)] for i in indices]

    narrations: list[str] = []
    for p in selected:
        entry = gaze_data.get(p.stem)
        if not entry:
            continue
        raw = entry.get("narration_text", "")
        if not raw:
            continue
        cleaned = clean_narration(raw)
        if not cleaned:
            continue
        if not narrations or narrations[-1] != cleaned:
            narrations.append(cleaned)

    if not narrations:
        return None
    return " ".join(narrations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4,
                    help="Sample-frame count to match at inference time. "
                         "Use 0 for ALL frames (richer GT, but mismatched if "
                         "you change inference n_frames later).")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT,
                    help=f"Output JSON path (default: {DEFAULT_OUT}).")
    args = ap.parse_args()

    n_frames = None if args.frames == 0 else args.frames

    with open(METADATA_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    seen_files = set()
    cache: dict[str, str] = {}
    skipped_no_frames = 0
    skipped_no_narration = 0

    for row in rows:
        fn = row["file_name"]
        if fn in seen_files:
            continue
        seen_files.add(fn)

        gt = build_ground_truth(fn, n_frames)
        if gt is None:
            # distinguish missing frames from missing narrations for diagnostics
            parts = Path(fn)
            dataset, scene_id, clip_stem = parts.parts[0], parts.parts[1], parts.stem
            frames_dir = FRAMES_DIR / dataset / scene_id / clip_stem
            if not frames_dir.exists() or not list(frames_dir.glob("*.jpg")):
                skipped_no_frames += 1
            else:
                skipped_no_narration += 1
            continue

        cache[fn] = gt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"Processed {len(seen_files)} unique clips.")
    print(f"  Wrote ground truths:        {len(cache)}")
    print(f"  Skipped (no local frames):  {skipped_no_frames}")
    print(f"  Skipped (no narrations):    {skipped_no_narration}")
    print(f"  → {args.output}")

    # Length stats for sanity
    if cache:
        lens = [len(v.split()) for v in cache.values()]
        lens.sort()
        print(f"\nGround-truth length (words):")
        print(f"  min/median/max = {lens[0]} / {lens[len(lens)//2]} / {lens[-1]}")


if __name__ == "__main__":
    main()
