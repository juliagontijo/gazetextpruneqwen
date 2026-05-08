"""
profile_frames.py — Iso-cost frames experiment.

Hypothesis: 8 frames + text pruning r=0.5 ≈ 4 frames no-prune in active token
count, but the 8-frame model sees richer temporal context → better CoT reasoning.

Conditions:
  4f_no_prune          — 4 frames, no pruning  (current baseline)
  8f_no_prune          — 8 frames, no pruning  (more context, ~2× slower)
  8f_text_l10_r0.5     — 8 frames, text-guided pruning at layer 10, keep 50%
  8f_gaze_l10_r0.5     — 8 frames, gaze-guided pruning at layer 10, keep 50%

Usage (from frames-experiments/):
    python profile_frames.py --condition 4f_no_prune      --num-samples 30
    python profile_frames.py --condition 8f_no_prune      --num-samples 30
    python profile_frames.py --condition 8f_text_l10_r0.5 --num-samples 30
    python profile_frames.py --condition 8f_gaze_l10_r0.5 --num-samples 30

Or run everything via:
    bash run_frames_experiments.sh
"""

import argparse
import importlib.util
import sys
from pathlib import Path

# ── Locate sara-experiments/profile_sara.py ──────────────────────────────────
HERE = Path(__file__).resolve().parent
SARA_DIR = HERE.parent / "sara-experiments"
sys.path.insert(0, str(SARA_DIR))

import profile_sara  # noqa: E402  (intentional late import after path patch)

DEFAULT_CSV = HERE / "results" / "frames-experiments.csv"

CONDITIONS = {
    "4f_no_prune": dict(
        n_frames=4, prune_text=False, prune_gaze=False, prune_random=False,
        prune_layers=[10], prune_ratio=0.5, prune_alpha=0.5,
        config_tag="4f_no_prune",
    ),
    "8f_no_prune": dict(
        n_frames=8, prune_text=False, prune_gaze=False, prune_random=False,
        prune_layers=[10], prune_ratio=0.5, prune_alpha=0.5,
        config_tag="8f_no_prune",
    ),
    "8f_text_l10_r0.5": dict(
        n_frames=8, prune_text=True, prune_gaze=False, prune_random=False,
        prune_layers=[10], prune_ratio=0.5, prune_alpha=0.5,
        config_tag="8f_text_l10_r0.5",
    ),
    "8f_gaze_l10_r0.5": dict(
        n_frames=8, prune_text=False, prune_gaze=True, prune_random=False,
        prune_layers=[10], prune_ratio=0.5, prune_alpha=0.5,
        config_tag="8f_gaze_l10_r0.5",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition",   choices=list(CONDITIONS), required=True,
                    help="Which experimental condition to run.")
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--results-csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--reset",       action="store_true",
                    help="Remove existing rows for this condition before running.")
    args = ap.parse_args()

    cond = CONDITIONS[args.condition]
    profile_sara.main(
        n_frames=cond["n_frames"],
        num_samples=args.num_samples,
        seed=args.seed,
        qa_types=None,
        task="mcq",
        prune_text=cond["prune_text"],
        prune_gaze=cond["prune_gaze"],
        prune_random=cond["prune_random"],
        prune_layers=cond["prune_layers"],
        prune_ratio=cond["prune_ratio"],
        prune_alpha=cond["prune_alpha"],
        config_tag=cond["config_tag"],
        results_csv=args.results_csv,
        save_viz=False,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
