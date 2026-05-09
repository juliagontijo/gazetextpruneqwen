"""
batch_speedup_test.py — Quick test to verify batching shows real speedup.

Runs the same sample at batch_size = 1, 2, 4, 8 with and without pruning.
Reports throughput (tokens/sec) and speedup ratios.

If we see speedup here, batched experiments are worth running overnight.

Usage:
    cd batched-experiments
    python batch_speedup_test.py
    python batch_speedup_test.py --batch-sizes 1 2 4 8 16
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import csv

# ── Repo root ────────────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / "modeling_qwen2_vl.py").exists():
            return current
        parent = current.parent
        if parent == current:
            raise FileNotFoundError("Cannot find repo root.")
        current = parent

REPO_ROOT    = _find_repo_root()
DATA_DIR     = REPO_ROOT / "data" / "EgoGazeVQA_full"
METADATA_CSV = DATA_DIR / "metadata.csv"
FRAMES_DIR   = DATA_DIR / "frames"
MODEL_ID     = "Qwen/Qwen2-VL-2B-Instruct"


def load_forked_model():
    path = REPO_ROOT / "modeling_qwen2_vl.py"
    spec = importlib.util.spec_from_file_location(
        "transformers.models.qwen2_vl.modeling_qwen2_vl", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "transformers.models.qwen2_vl"
    sys.modules["transformers.models.qwen2_vl.modeling_qwen2_vl"] = module
    spec.loader.exec_module(module)
    return module.Qwen2VLForConditionalGeneration


def pick_sample():
    """Return first available sample from metadata."""
    with open(METADATA_CSV, newline="") as f:
        for row in csv.DictReader(f):
            fn = row["file_name"].replace(".mp4", "")
            folder = FRAMES_DIR / fn
            if folder.exists() and list(folder.glob("*.jpg")):
                return row
    raise RuntimeError("No samples found.")


def load_frames(file_name: str, n_frames: int = 4):
    fn = file_name.replace(".mp4", "")
    folder = FRAMES_DIR / fn
    jpgs = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem))
    n = len(jpgs)
    indices = [int(i * (n - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
    return [Image.open(jpgs[min(i, n-1)]).convert("RGB") for i in indices]


def build_batch_inputs(processor, frames, question, options_str, batch_size, device):
    """Duplicate the same sample batch_size times."""
    prompt = (
        f"Watch the video carefully and answer the following question.\n"
        f"Think step by step, then end with 'Answer: X'.\n\n"
        f"Question: {question}\n\nOptions:\n{options_str}"
    )
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text",  "text": prompt},
        ]}
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text] * batch_size,
        images=image_inputs * batch_size if image_inputs else None,
        videos=video_inputs * batch_size if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def run_timed(model, inputs, max_new_tokens, extra_kwargs=None):
    extra_kwargs = extra_kwargs or {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            min_new_tokens=50,
            **extra_kwargs,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    # tokens generated across the whole batch
    new_tokens = (out.shape[1] - inputs.input_ids.shape[1]) * inputs.input_ids.shape[0]
    return elapsed, new_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes",  type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--prune-ratio",    type=float, default=0.5)
    ap.add_argument("--prune-layer",    type=int,   default=10)
    ap.add_argument("--n-frames",       type=int,   default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading model {MODEL_ID}...")
    Qwen2VL = load_forked_model()
    model = Qwen2VL.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="eager")
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model loaded.\n")

    sample = pick_sample()
    frames = load_frames(sample["file_name"], args.n_frames)
    options_str = "\n".join(o.strip() for o in sample["answer_options"].split("|"))

    pruning_kwargs = {
        "pruning_layers": [args.prune_layer],
        "pruning_ratios": {args.prune_layer: args.prune_ratio},
        "prune_text": True,
    }

    print(f"{'─'*65}")
    print(f"  Sample: {sample['file_name']}")
    print(f"  Question: {sample['question'][:60]}...")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Prune ratio: {args.prune_ratio}  Prune layer: {args.prune_layer}")
    print(f"{'─'*65}\n")

    results = []

    for bs in args.batch_sizes:
        print(f"  Batch size = {bs}")
        try:
            inputs = build_batch_inputs(
                processor, frames, sample["question"], options_str, bs, device)

            # Warmup
            with torch.inference_mode():
                _ = model.generate(**inputs, use_cache=True,
                                   max_new_tokens=8)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # No pruning
            t_base, tok_base = run_timed(model, inputs, args.max_new_tokens)
            tps_base = tok_base / t_base

            # Text pruning
            t_prune, tok_prune = run_timed(model, inputs, args.max_new_tokens,
                                            extra_kwargs=pruning_kwargs)
            tps_prune = tok_prune / t_prune

            speedup = tps_prune / tps_base

            print(f"    no_prune : {t_base:.2f}s  {tok_base} toks  {tps_base:.1f} tok/s")
            print(f"    pruned   : {t_prune:.2f}s  {tok_prune} toks  {tps_prune:.1f} tok/s")
            print(f"    speedup  : {speedup:.2f}×\n")

            results.append({
                "batch_size": bs,
                "no_prune_tps": round(tps_base, 2),
                "pruned_tps": round(tps_prune, 2),
                "speedup": round(speedup, 3),
                "no_prune_s": round(t_base, 3),
                "pruned_s": round(t_prune, 3),
            })

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at batch_size={bs} — stopping here.\n")
            break
        except Exception as e:
            print(f"    ERROR: {e}\n")
            break

    print(f"\n{'═'*55}")
    print("  SUMMARY")
    print(f"{'═'*55}")
    print(f"  {'BS':>4}  {'No-prune tok/s':>15}  {'Pruned tok/s':>13}  {'Speedup':>8}")
    print(f"  {'-'*53}")
    for r in results:
        print(f"  {r['batch_size']:>4}  {r['no_prune_tps']:>15.1f}  "
              f"{r['pruned_tps']:>13.1f}  {r['speedup']:>7.2f}×")

    # Save
    out = Path(__file__).parent / "results" / "batch_speedup_test.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {out}")

    if results:
        max_speedup = max(r["speedup"] for r in results)
        if max_speedup >= 1.15:
            print(f"\n✓ Speedup confirmed ({max_speedup:.2f}× at best batch size).")
            print("  → Batched experiments are worth running overnight.")
        else:
            print(f"\n✗ No meaningful speedup seen (max {max_speedup:.2f}×).")
            print("  → Pruning ratio too conservative or model too small.")
            print("  → Try --prune-ratio 0.1 for more aggressive pruning.")


if __name__ == "__main__":
    main()
