"""
llm_judge.py — LLM-as-judge scorer using a local Qwen model (free, no API key).

Reads rows produced by profile_sara.py (where cot_output contains the model's
full chain-of-thought) and scores each MCQ response using a locally loaded
Qwen text model.

The scoring rubric is specialised per qa_type:
  spatial  — quality of spatial / positional reasoning
  causal   — quality of causal / intent reasoning (gaze as attention proxy)
  temporal — quality of temporal sequencing reasoning

Scores are 0–5 integers.  Fills the llm_judge_score and llm_judge_reasoning
columns and writes back to the CSV.

Usage:
    python llm_judge.py --csv results/sara_experiments.csv
    python llm_judge.py --csv results/sara_experiments.csv --config-tag no_prune
    python llm_judge.py --csv results/sara_experiments.csv --judge-model Qwen/Qwen2.5-1.5B-Instruct
    python llm_judge.py --csv results/sara_experiments.csv --dry-run
"""

import argparse
import csv
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CSV_COLUMNS = [
    "config_tag", "sample_idx", "video_id", "qa_type",
    "prune_text", "prune_gaze", "prune_random", "prune_alpha", "prune_ratio", "prune_layer",
    "input_preprocessing_s", "vision_encoder_s", "decode_s", "tokens_generated",
    "decode_ms_per_token", "correct_answer", "predicted_answer", "correct",
    "cot_output", "llm_judge_score", "llm_judge_reasoning",
]

# ── Rubrics per question type ────────────────────────────────────────────────

_RUBRIC_SPATIAL = """You are evaluating an egocentric video QA system on a SPATIAL question.
Spatial questions ask about the location, position, or arrangement of objects in the scene.

Question context — correct answer: {correct_answer}
Model's full response (including reasoning):
{cot_output}

Score the quality of the model's spatial reasoning on a scale of 0–5:
  0 — No reasoning provided, or completely wrong / irrelevant reasoning
  1 — Attempts reasoning but makes fundamental spatial errors (wrong location, wrong objects)
  2 — Partial understanding; identifies some objects/positions but misses key spatial relationships
  3 — Reasonable spatial reasoning with minor errors; most relationships correct
  4 — Good spatial reasoning; correctly identifies positions and relationships, minor gaps
  5 — Excellent reasoning; precise spatial analysis, well-justified, leads to correct answer

Provide one sentence of evaluation, then output exactly:
Score: X"""

_RUBRIC_CAUSAL = """You are evaluating an egocentric video QA system on a CAUSAL question.
Causal questions ask why something happened, what motivated an action, or what caused an observed event.
The model has access to the person's gaze data as a proxy for their attention and intent.

Question context — correct answer: {correct_answer}
Model's full response (including reasoning):
{cot_output}

Score the quality of the model's causal reasoning on a scale of 0–5:
  0 — No reasoning, or completely wrong / irrelevant reasoning
  1 — Attempts reasoning but misidentifies the cause or confuses correlation with causation
  2 — Partially correct; identifies some causal factors but misses key motivations or intent
  3 — Reasonable causal chain with minor gaps; connects actions to plausible motivations
  4 — Good causal reasoning; correctly identifies motivation/intent and supports it with evidence
  5 — Excellent reasoning; precise causal chain, uses gaze/attention cues effectively, well-justified

Provide one sentence of evaluation, then output exactly:
Score: X"""

_RUBRIC_TEMPORAL = """You are evaluating an egocentric video QA system on a TEMPORAL question.
Temporal questions ask about the sequence of events, what happened before/after something, or the order of actions.

Question context — correct answer: {correct_answer}
Model's full response (including reasoning):
{cot_output}

Score the quality of the model's temporal reasoning on a scale of 0–5:
  0 — No reasoning, or completely ignores temporal order
  1 — Attempts temporal reasoning but gets the sequence fundamentally wrong
  2 — Partial temporal understanding; correctly orders some events but misses key transitions
  3 — Reasonable temporal reasoning; mostly correct sequence with minor ordering errors
  4 — Good temporal reasoning; correctly sequences events and uses temporal language appropriately
  5 — Excellent reasoning; precise sequencing, explicitly references before/after/during, well-justified

Provide one sentence of evaluation, then output exactly:
Score: X"""

RUBRICS = {
    "spatial":  _RUBRIC_SPATIAL,
    "causal":   _RUBRIC_CAUSAL,
    "temporal": _RUBRIC_TEMPORAL,
}


def is_judgeable_row(row: dict) -> bool:
    """MCQ rows with cot_output that haven't been judged yet."""
    return (
        row.get("cot_output", "").strip() != ""
        and row.get("llm_judge_score", "").strip() == ""
        and row.get("qa_type", "") in RUBRICS
    )


def build_prompt(row: dict) -> str:
    rubric = RUBRICS[row["qa_type"]]
    return rubric.format(
        correct_answer=row["correct_answer"],
        cot_output=row["cot_output"][:1500],  # keep under context limit
    )


def parse_score(response_text: str) -> tuple[int | None, str]:
    """Extract score integer; return (score, reasoning_text)."""
    m = re.search(r"Score:\s*([0-5])", response_text)
    score = int(m.group(1)) if m else None
    score_idx = response_text.lower().rfind("score:")
    reasoning = response_text[:score_idx].strip() if score_idx >= 0 else response_text.strip()
    return score, reasoning


def load_judge_model(model_id: str, device: str):
    """Load a local Qwen text model and tokenizer."""
    print(f"Loading judge model: {model_id}  (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    print("Judge model loaded.\n")
    return model, tokenizer


def call_judge(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 400) -> str:
    """Run a single judge inference and return the decoded response."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def judge_rows(targets: list[dict], model, tokenizer, device: str, max_new_tokens: int = 400) -> None:
    """Judge each row; mutates rows in-place."""
    n = len(targets)
    for i, row in enumerate(targets):
        qa_type = row["qa_type"]
        prompt = build_prompt(row)

        print(f"  [{i+1}/{n}] sample={row['sample_idx']} qa_type={qa_type} config={row['config_tag'][:35]}")

        try:
            response = call_judge(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
            score, reasoning = parse_score(response)

            if score is None:
                print(f"    WARNING: could not parse score from: {response[:100]!r}")
                row["llm_judge_score"] = ""
                row["llm_judge_reasoning"] = response[:500]
            else:
                print(f"    → score={score}  reasoning={reasoning[:80]!r}")
                row["llm_judge_score"] = str(score)
                row["llm_judge_reasoning"] = reasoning[:500].replace("\n", " ")

        except Exception as e:
            print(f"    ERROR: {e}")
            row["llm_judge_score"] = ""
            row["llm_judge_reasoning"] = f"ERROR: {e}"


def print_summary(targets: list[dict]) -> None:
    scored = [
        r for r in targets
        if r.get("llm_judge_score", "").strip() not in ("", ) and
        r["llm_judge_score"].lstrip("-").isdigit()
    ]
    if not scored:
        print("No rows were successfully scored.")
        return

    print(f"\nLLM judge summary ({len(scored)} rows scored):")

    by_type: dict[str, list[int]] = {}
    for r in scored:
        by_type.setdefault(r["qa_type"], []).append(int(r["llm_judge_score"]))

    print("By qa_type:")
    for qt, scores in sorted(by_type.items()):
        print(f"  {qt:10s}  mean={sum(scores)/len(scores):.2f}  n={len(scores)}")

    by_tag: dict[str, list[int]] = {}
    for r in scored:
        by_tag.setdefault(r["config_tag"], []).append(int(r["llm_judge_score"]))

    print("\nBy config_tag:")
    for tag, scores in sorted(by_tag.items()):
        print(f"  {tag:45s}  mean={sum(scores)/len(scores):.2f}  n={len(scores)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",         type=Path, required=True,
                    help="Path to sara_experiments.csv produced by profile_sara.py.")
    ap.add_argument("--config-tag",  type=str,  default=None,
                    help="Only judge rows whose config_tag contains this substring.")
    ap.add_argument("--judge-model", type=str,
                    default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="HuggingFace model ID for the judge (default: Qwen2.5-1.5B-Instruct).")
    ap.add_argument("--max-new-tokens", type=int, default=400,
                    help="Max tokens for judge response (default: 400).")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Print which rows would be judged without running inference.")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    targets = []
    for r in rows:
        if not is_judgeable_row(r):
            continue
        if args.config_tag and args.config_tag not in r["config_tag"]:
            continue
        targets.append(r)

    if not targets:
        print("No rows to judge (cot_output empty, already judged, or unsupported qa_type).")
        return

    print(f"Rows to judge: {len(targets)}  (judge model={args.judge_model})")

    if args.dry_run:
        for r in targets:
            print(f"  would judge: sample={r['sample_idx']} qa_type={r['qa_type']} config={r['config_tag']}")
        return

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    judge_model, tokenizer = load_judge_model(args.judge_model, device)

    judge_rows(targets, judge_model, tokenizer, device, max_new_tokens=args.max_new_tokens)

    # Free GPU/MPS memory before writing
    del judge_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    fieldnames = [c for c in CSV_COLUMNS if c in rows[0]]
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nUpdated → {args.csv}")
    print_summary(targets)


if __name__ == "__main__":
    main()
