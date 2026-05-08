"""
train_grpo.py — GRPO fine-tuning for egocentric VQA reasoning.

Qwen2-VL-2B-Instruct + LoRA (LM attention only, vision encoder frozen).
Reward: 1.0 correct answer, 0.0 otherwise.
No KL penalty (clean GRPO for short training runs).

Install deps first:
    pip install peft

Run from sara-experiments/:
    python train_grpo.py

Expected time on A100: ~1.5 hours (150 samples × 2 epochs, G=4, 4 frames).
"""

import argparse
import csv
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from qwen_vl_utils import process_vision_info


# ── Paths ─────────────────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / "modeling_qwen2_vl.py").exists():
            return current
        parent = current.parent
        if parent == current:
            raise FileNotFoundError("modeling_qwen2_vl.py not found")
        current = parent


REPO_ROOT    = _find_repo_root()
DATA_DIR     = REPO_ROOT / "data" / "EgoGazeVQA_full"
FRAMES_DIR   = DATA_DIR / "frames"
METADATA_CSV = DATA_DIR / "metadata.csv"
MODEL_ID     = "Qwen/Qwen2-VL-2B-Instruct"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_samples(num_samples: int, seed: int, split: str = "train") -> list[dict]:
    with open(METADATA_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))

    available = []
    for row in all_rows:
        fn = row["file_name"].replace(".mp4", "")
        parts = Path(fn)
        frames_dir = FRAMES_DIR / parts.parts[0] / parts.parts[1] / parts.stem
        if frames_dir.exists() and list(frames_dir.glob("*.jpg")):
            available.append(row)

    rng = random.Random(seed)
    rng.shuffle(available)

    # stratified by qa_type
    buckets: dict[str, list] = {"causal": [], "spatial": [], "temporal": []}
    for r in available:
        qt = r.get("qa_type", "")
        if qt in buckets:
            buckets[qt].append(r)

    per_type = num_samples // 3
    samples = []
    for qt in ["causal", "spatial", "temporal"]:
        samples.extend(buckets[qt][:per_type])
    rng.shuffle(samples)
    return samples[:num_samples]


def load_frames(file_name: str, n_frames: int):
    fn = file_name.replace(".mp4", "")
    parts = Path(fn)
    frames_dir = FRAMES_DIR / parts.parts[0] / parts.parts[1] / parts.stem

    jpgs = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    n = len(jpgs)
    indices = [int(i * (n - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
    selected = [jpgs[min(i, n - 1)] for i in indices]

    frames = [Image.open(p).convert("RGB") for p in selected]
    return frames


def build_messages(row: dict, frames: list) -> list[dict]:
    options = row["answer_options"].split("|")
    options_str = "\n".join(o.strip() for o in options)
    text = (
        "You are watching a first-person (egocentric) video. "
        "The camera wearer is performing everyday activities.\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{options_str}\n\n"
        "First describe what you observe in the video. "
        "Then reason step by step about which option best matches what you saw. "
        "Finish with 'Answer: X' where X is the letter of your chosen option."
    )
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


# ── Reward ────────────────────────────────────────────────────────────────────
def compute_reward(output: str, correct_answer: str) -> float:
    correct = correct_answer.strip()[0].upper()
    m = re.search(r"Answer:\s*([A-E])", output, re.IGNORECASE)
    if m:
        return 1.0 if m.group(1).upper() == correct else 0.0
    # fallback: last standalone letter
    m = re.search(r"\b([A-E])\b", output[::-1], re.IGNORECASE)
    if m:
        return 0.5 if m.group(1).upper() == correct else 0.0
    return 0.0


# ── Core GRPO ────────────────────────────────────────────────────────────────
def grpo_step(
    model,
    processor,
    row: dict,
    n_frames: int,
    G: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, float]:
    """
    Single-sample GRPO update tensor.
    Returns (loss, mean_reward) — loss is None if no gradient signal.
    """
    frames = load_frames(row["file_name"], n_frames)
    messages = build_messages(row, frames)

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]

    # ── Generate G rollouts (no grad) ─────────────────────────────────────
    rollout_ids = []
    rollout_texts = []
    for _ in range(G):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=20,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
            )
        rollout_ids.append(out[0])
        rollout_texts.append(
            processor.decode(out[0, prompt_len:], skip_special_tokens=True)
        )

    rewards = torch.tensor(
        [compute_reward(t, row["correct_answer"]) for t in rollout_texts],
        dtype=torch.float32, device=device,
    )
    mean_reward = rewards.mean().item()

    # No gradient signal if all rewards identical
    if rewards.std() < 1e-6:
        return None, mean_reward

    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # ── Log-prob pass under current policy (with grad) ────────────────────
    pixel_values   = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    rollout_losses = []
    for i in range(G):
        full_ids = rollout_ids[i].unsqueeze(0)          # (1, full_len)
        response_ids = full_ids[:, prompt_len:]          # (1, response_len)
        if response_ids.shape[1] == 0:
            continue

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            ).logits                                     # (1, full_len, vocab)

        # logit at position t predicts token t+1
        resp_logits = logits[:, prompt_len - 1:-1, :]    # (1, response_len, vocab)
        log_probs   = F.log_softmax(resp_logits, dim=-1)
        token_lps   = log_probs.gather(
            2, response_ids.unsqueeze(-1)
        ).squeeze(-1).mean()                             # scalar

        rollout_losses.append(-advantages[i] * token_lps)

    if not rollout_losses:
        return None, mean_reward

    loss = torch.stack(rollout_losses).mean()
    return loss, mean_reward


# ── Quick eval ───────────────────────────────────────────────────────────────
def quick_eval(model, processor, samples: list[dict], n_frames: int, device) -> float:
    correct = 0
    model.eval()
    for row in samples:
        try:
            frames = load_frames(row["file_name"], n_frames)
            messages = build_messages(row, frames)
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_prompt], images=image_inputs,
                videos=video_inputs, padding=True, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            text = processor.decode(
                out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            if compute_reward(text, row["correct_answer"]) == 1.0:
                correct += 1
        except Exception:
            pass
    model.train()
    return correct / len(samples) if samples else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples",    type=int,   default=150,
                    help="Training samples (50 per qa_type).")
    ap.add_argument("--eval-samples",   type=int,   default=30,
                    help="Held-out samples for quick eval.")
    ap.add_argument("--epochs",         type=int,   default=2)
    ap.add_argument("--frames",         type=int,   default=4)
    ap.add_argument("--G",              type=int,   default=4,
                    help="Rollouts per sample.")
    ap.add_argument("--grad-accum",     type=int,   default=4,
                    help="Gradient accumulation steps.")
    ap.add_argument("--lr",             type=float, default=5e-5)
    ap.add_argument("--max-new-tokens", type=int,   default=100)
    ap.add_argument("--lora-r",         type=int,   default=8)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--output-dir",     type=Path,  default=Path("grpo_checkpoints"))
    ap.add_argument("--resume-from",    type=Path,  default=None,
                    help="LoRA checkpoint dir to resume from (e.g. grpo_checkpoints/epoch_2).")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load forked model ─────────────────────────────────────────────────
    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    module_name = "transformers.models.qwen2_vl.modeling_qwen2_vl"
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / "modeling_qwen2_vl.py"
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "transformers.models.qwen2_vl"
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    Qwen2VLModel = mod.Qwen2VLForConditionalGeneration

    model = Qwen2VLModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # ── LoRA on LM attention only ─────────────────────────────────────────
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    if args.resume_from is not None:
        from peft import set_peft_model_state_dict
        import safetensors.torch as st
        weights = st.load_file(args.resume_from / "adapter_model.safetensors")
        set_peft_model_state_dict(model, weights)
        print(f"Resumed from {args.resume_from}")

    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # ── Split train / eval ────────────────────────────────────────────────
    all_samples = load_samples(args.num_samples + args.eval_samples, args.seed)
    train_samples = all_samples[:args.num_samples]
    eval_samples  = all_samples[args.num_samples:args.num_samples + args.eval_samples]
    print(f"Train: {len(train_samples)}  Eval: {len(eval_samples)}")

    # Baseline eval
    print("Baseline eval ...")
    baseline_acc = quick_eval(model, processor, eval_samples, args.frames, device)
    print(f"  Baseline accuracy: {baseline_acc:.3f}")

    log_path = args.output_dir / "grpo_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,step,loss,mean_reward,elapsed_s\n")

    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(train_samples)
        optimizer.zero_grad()
        accum_loss = torch.tensor(0.0, device=device)
        accum_rewards = []
        n_accum = 0
        t_epoch = time.time()

        for i, row in enumerate(train_samples):
            t0 = time.time()
            try:
                loss, mean_reward = grpo_step(
                    model, processor, row,
                    args.frames, args.G, args.max_new_tokens, device,
                )
            except Exception as e:
                print(f"  [skip] {row['file_name']}: {e}")
                continue

            accum_rewards.append(mean_reward)

            if loss is not None:
                (loss / args.grad_accum).backward()
                accum_loss = accum_loss + loss.detach()
                n_accum += 1

            if (i + 1) % args.grad_accum == 0:
                if n_accum > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    global_step += 1

                avg_loss   = (accum_loss / n_accum).item() if n_accum > 0 else 0.0
                avg_reward = sum(accum_rewards) / len(accum_rewards)
                elapsed    = time.time() - t0

                print(
                    f"  e{epoch+1} step={global_step:3d} "
                    f"loss={avg_loss:.4f}  reward={avg_reward:.3f}  "
                    f"[{i+1}/{len(train_samples)}]  {elapsed:.1f}s"
                )
                with open(log_path, "a") as f:
                    f.write(f"{epoch+1},{global_step},{avg_loss:.4f},{avg_reward:.3f},{elapsed:.1f}\n")

                optimizer.zero_grad()
                accum_loss = torch.tensor(0.0, device=device)
                accum_rewards = []
                n_accum = 0

        epoch_time = time.time() - t_epoch
        print(f"Epoch {epoch+1} done in {epoch_time/60:.1f} min")

        # Per-epoch eval
        acc = quick_eval(model, processor, eval_samples, args.frames, device)
        print(f"  Eval accuracy: {acc:.3f}  (baseline: {baseline_acc:.3f})")

        ckpt = args.output_dir / f"epoch_{epoch+1}"
        model.save_pretrained(ckpt)
        processor.save_pretrained(ckpt)
        print(f"  Saved → {ckpt}")

    print(f"\nDone. Run profile_sara.py with --lora-path grpo_checkpoints/epoch_{args.epochs} to evaluate.")


if __name__ == "__main__":
    main()
