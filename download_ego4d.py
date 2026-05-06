"""
Download Ego4D video clips for specific video_ids from taiyi09/EgoGazeVQA on HuggingFace.

Usage:
  conda run -n gazeprune python download_ego4d.py
  conda run -n gazeprune python download_ego4d.py --video-ids dafc891e-05f0-4734-88c6-1f818ac67a23 566ad4e5-1ce4-4679-9d19-ef63072c848c

Downloads to: data/EgoGazeVQA_full/ego4d/<video_id>/<clip>.mp4
"""

import argparse
import csv
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "taiyi09/EgoGazeVQA"
BASE_DIR = Path(__file__).parent / "data" / "EgoGazeVQA_full"
METADATA_CSV = BASE_DIR / "metadata.csv"

# Video IDs from metadata that aren't the one already partially downloaded
DEFAULT_VIDEO_IDS = [
    "dafc891e-05f0-4734-88c6-1f818ac67a23",  # 30 QA rows
    "566ad4e5-1ce4-4679-9d19-ef63072c848c",  # 30 QA rows
    "2bb31b69-fcda-4f54-8338-f590944df999",  # 30 QA rows (already have raw mp4s)
]


def get_clips_for_video_ids(video_ids: list[str]) -> list[str]:
    """Return unique HF repo file paths for all clips belonging to the given video_ids."""
    seen = set()
    paths = []
    with open(METADATA_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["video_id"] in video_ids and row["dataset"] == "ego4d":
                p = row["file_name"]
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
    return paths


def download_clips(video_ids: list[str], dry_run: bool = False) -> None:
    clips = get_clips_for_video_ids(video_ids)
    if not clips:
        print("No clips found for the given video IDs.")
        return

    print(f"Found {len(clips)} clips across {len(video_ids)} video(s).\n")

    for i, hf_path in enumerate(clips, 1):
        local_path = BASE_DIR / hf_path
        if local_path.exists():
            print(f"  [{i:02d}/{len(clips)}] skip (exists): {hf_path}")
            continue

        if dry_run:
            print(f"  [{i:02d}/{len(clips)}] would download: {hf_path}")
            continue

        print(f"  [{i:02d}/{len(clips)}] downloading: {hf_path} ...", end=" ", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            local_dir=str(BASE_DIR),
        )
        print("done")

    print(f"\nAll done. Run next:\n  conda run -n gazeprune python preprocess_frames.py --dataset ego4d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-ids", nargs="+", default=DEFAULT_VIDEO_IDS,
        help="Ego4D video IDs to download (default: 3 pre-selected IDs = 90 QA rows).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded without downloading.")
    args = parser.parse_args()

    download_clips(args.video_ids, dry_run=args.dry_run)
