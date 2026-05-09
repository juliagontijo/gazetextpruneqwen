"""
Offline LLM-as-Judge reasoning scorer for MCQ rows in ablations.csv.

Uses Qwen2.5-7B-Instruct locally (no API key needed) to evaluate each row's
chain-of-thought against the ground-truth narration using a per-QA-type rubric
with 5 binary criteria (0 or 1 each).

Two columns are written:
  reasoning_score       — mean of the 5 binary scores (0.0 – 1.0)
  reasoning_explanation — compact JSON with per-criterion {"score": 0|1, "reason": "..."}

Rubrics
-------
causal   — action_observation · context_identification · causal_inference ·
           logical_consistency · visual_grounding
spatial  — object_identification · spatial_description · spatial_inference ·
           logical_consistency · visual_grounding
temporal — event_identification · sequence_description · temporal_inference ·
           logical_consistency · visual_grounding

Usage:
    python score_reasoning.py --csv results/ablations.csv
    python score_reasoning.py --csv results/ablations.csv --config-tag text_l10_r0.5
    python score_reasoning.py --csv results/ablations.csv --dry-run
    python score_reasoning.py --csv results/ablations.csv --model Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Column definitions ──────────────────────────────────────────────────────
CSV_COLUMNS = [
    "config_tag", "sample_idx", "video_id", "qa_type",
    "prune_text", "prune_gaze", "prune_random", "prune_alpha", "prune_ratio", "prune_layer",
    "input_preprocessing_s", "vision_encoder_s", "decode_s", "tokens_generated",
    "decode_ms_per_token", "correct_answer", "predicted_answer", "correct",
    "cot_text", "narration_gt", "cot_coverage", "reasoning_score", "reasoning_explanation",
]

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ── Rubrics ──────────────────────────────────────────────────────────────────
RUBRICS = {
    "causal": {
        "criteria": [
            ("action_observation",    "Does the CoT correctly identify what action was performed?"),
            ("context_identification","Does the CoT identify relevant environmental context (objects, setting, prior actions)?"),
            ("causal_inference",      "Does the CoT draw a plausible causal link between context and the action/outcome?"),
            ("logical_consistency",   "Is the reasoning internally consistent (no contradictions)?"),
            ("visual_grounding",      "Is the reasoning grounded in visual scene details mentioned in the narration?"),
        ],
    },
    "spatial": {
        "criteria": [
            ("object_identification", "Does the CoT correctly identify the objects being asked about?"),
            ("spatial_description",   "Does the CoT describe the spatial relationship or location accurately?"),
            ("spatial_inference",     "Does the CoT reach the correct spatial conclusion given the evidence?"),
            ("logical_consistency",   "Is the reasoning internally consistent (no contradictions)?"),
            ("visual_grounding",      "Is the reasoning grounded in visual scene details mentioned in the narration?"),
        ],
    },
    "temporal": {
        "criteria": [
            ("event_identification",  "Does the CoT correctly identify the events relevant to the question?"),
            ("sequence_description",  "Does the CoT describe the temporal order of events accurately?"),
            ("temporal_inference",    "Does the CoT correctly infer what happened before/after the reference event?"),
            ("logical_consistency",   "Is the reasoning internally consistent (no contradictions)?"),
            ("visual_grounding",      "Is the reasoning grounded in visual scene details mentioned in the narration?"),
        ],
    },
}

SYSTEM_PROMPT = """You are an expert evaluator of video-QA reasoning chains for egocentric video understanding.

You will be given:
- A QUESTION about a first-person egocentric video clip
- The CORRECT ANSWER (the right multiple-choice option)
- A GROUND-TRUTH NARRATION describing what was actually happening in the clip
- The MODEL'S CHAIN-OF-THOUGHT reasoning (produced before the model chose its answer)

Your task is to evaluate the quality of the chain-of-thought reasoning using a set of binary criteria specific to the question type.

Rules:
- Base your judgement on whether the CoT reasoning demonstrates understanding of the scene, not whether the final answer was correct
- A CoT that arrives at the wrong answer but shows good reasoning should still score well on reasoning criteria
- A CoT that gets the right answer by guessing or by superficial pattern-matching should score low
- Be strict: score 1 only when the criterion is clearly met
- For each criterion output a "score" (0 or 1) AND a brief "reason" (one sentence explaining the score)
- Output ONLY a JSON object. Each key is a criterion name; each value is {"score": 0|1, "reason": "..."}. No other text."""


def build_user_prompt(qa_type: str, question: str, narration_gt: str, cot_text: str) -> str:
    rubric = RUBRICS[qa_type]
    criteria_lines = "\n".join(
        f"- {name}: {desc}"
        for name, desc in rubric["criteria"]
    )
    criterion_keys = ", ".join(f'"{name}"' for name, _ in rubric["criteria"])
    return f"""QUESTION TYPE: {qa_type}

QUESTION / CORRECT ANSWER:
{question}

GROUND-TRUTH NARRATION (what actually happened):
{narration_gt}

MODEL'S CHAIN-OF-THOUGHT:
{cot_text}

EVALUATION CRITERIA ({qa_type}):
{criteria_lines}

Respond with ONLY a JSON object using these exact keys: {{{criterion_keys}}}
Each value must be an object: {{"score": 0 or 1, "reason": "one-sentence explanation"}}"""


def is_scorable(row: dict) -> bool:
    return (
        row.get("reasoning_score", "") == ""
        and len(row.get("cot_text", "").strip()) > 10
        and len(row.get("narration_gt", "").strip()) > 10
        and row.get("qa_type", "") in RUBRICS
    )


def parse_response(response_text: str, qa_type: str) -> tuple[float, str] | tuple[None, None]:
    """Parse JSON from model response; return (mean_score, explanation_json) or (None, None)."""
    try:
        text = response_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        # Find the first { ... } block in case of preamble
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None, None
        data = json.loads(text[start:end])

        criteria_names = [name for name, _ in RUBRICS[qa_type]["criteria"]]
        scores = []
        explanation: dict[str, dict] = {}
        for name in criteria_names:
            if name not in data:
                continue
            entry = data[name]
            if isinstance(entry, dict):
                s = int(entry.get("score", 0))
                reason = str(entry.get("reason", "")).strip()
            else:
                s = int(entry)
                reason = ""
            scores.append(s)
            explanation[name] = {"score": s, "reason": reason}

        if not scores:
            return None, None

        mean_score = sum(scores) / len(scores)
        explanation_str = json.dumps(explanation, ensure_ascii=False, separators=(",", ":"))
        return mean_score, explanation_str
    except Exception:
        return None, None


def load_model(model_id: str):
    print(f"Loading judge model: {model_id} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    print(f"  Loaded on {device}.\n")
    return model, tokenizer


def score_row(model, tokenizer, row: dict) -> tuple[float, str] | tuple[None, None]:
    qa_type = row["qa_type"]
    user_prompt = build_user_prompt(
        qa_type=qa_type,
        question=row.get("correct_answer", ""),
        narration_gt=row["narration_gt"],
        cot_text=row["cot_text"],
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_response(response_text, qa_type)


def summarise(targets: list[dict], scores: list[float]) -> None:
    by_tag: dict[str, list[float]] = {}
    by_tag_qa: dict[tuple, list[float]] = {}
    for r, s in zip(targets, scores):
        by_tag.setdefault(r["config_tag"], []).append(s)
        by_tag_qa.setdefault((r["config_tag"], r["qa_type"]), []).append(s)

    print("\nReasoning score (LLM-as-Judge) per config:")
    print(f"  {'config_tag':<45} {'mean score':>10}  {'n':>4}")
    print(f"  {'-'*45} {'-'*10}  {'-'*4}")
    for tag, vals in sorted(by_tag.items()):
        m = sum(vals) / len(vals)
        print(f"  {tag:<45} {m:>10.4f}  {len(vals):>4}")

    qa_types = sorted({r["qa_type"] for r in targets})
    if len(qa_types) > 1:
        print()
        for tag in sorted(by_tag):
            for qt in qa_types:
                vals = by_tag_qa.get((tag, qt), [])
                if vals:
                    print(f"  {tag:<35} {qt:<10} {sum(vals)/len(vals):.4f}  (n={len(vals)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help=f"HuggingFace judge model (default: {DEFAULT_MODEL})")
    ap.add_argument("--config-tag", type=str, default=None,
                    help="Only score rows whose config_tag contains this substring.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be scored without running inference.")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r.setdefault("reasoning_score", "")
        r.setdefault("reasoning_explanation", "")

    targets = []
    for r in rows:
        if not is_scorable(r):
            continue
        if args.config_tag and args.config_tag not in r["config_tag"]:
            continue
        targets.append(r)

    if not targets:
        print("No scorable rows found (already scored, missing CoT/narration, or unknown qa_type).")
        return

    print(f"Scoring {len(targets)} rows with LLM-as-Judge ({args.model})...")

    if args.dry_run:
        for r in targets:
            print(f"  [dry-run] {r['config_tag']}  qa_type={r['qa_type']}  cot_len={len(r['cot_text'])}")
        return

    model, tokenizer = load_model(args.model)

    completed_scores: list[float] = []
    failed = 0

    for i, r in enumerate(targets):
        tag = r.get("config_tag", "?")
        qt = r.get("qa_type", "?")
        print(f"  [{i+1}/{len(targets)}] {tag}  qa_type={qt} ...", end=" ", flush=True)
        try:
            score, explanation = score_row(model, tokenizer, r)
            if score is None:
                print("PARSE ERROR — skipping")
                failed += 1
                completed_scores.append(float("nan"))
            else:
                r["reasoning_score"] = f"{score:.4f}"
                r["reasoning_explanation"] = explanation or ""
                completed_scores.append(score)
                print(f"{score:.4f}")
        except Exception as e:
            print(f"ERROR: {e} — skipping")
            failed += 1
            completed_scores.append(float("nan"))

        # Write incrementally so progress survives interruption
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})

    valid_scores = [s for s in completed_scores if s == s]  # filter NaN
    if failed:
        print(f"\n{failed} rows failed (parse error or model error).")

    if valid_scores:
        summarise(
            [r for r, s in zip(targets, completed_scores) if s == s],
            valid_scores,
        )

    print(f"\nUpdated → {args.csv}")


if __name__ == "__main__":
    main()
