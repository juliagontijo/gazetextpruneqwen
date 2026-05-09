"""
evaluation.py — Comprehensive LLM-based evaluation of CoT reasoning.

Reads sara_experiments.csv (produced by profile_sara.py) and adds three
evaluation columns using a local judge model:

  --mode reasoning   (default)
      Judge sees: scene description + question + correct answer + model CoT
      Fills: llm_judge_score (0–5), llm_judge_reasoning

  --mode coverage
      Judge scores three dimensions of CoT quality:
        scene_grounding  — does CoT reference actual objects/actions in scene?
        question_relevance — does CoT address what the question asks?
        logical_chain    — is the reasoning coherent and well-structured?
      Fills: llm_coverage_scene, llm_coverage_relevance, llm_coverage_logic,
             llm_coverage_final, llm_coverage_breakdown

  --mode preference
      For each pruned config, judge compares its CoT to the no_prune baseline
      side-by-side on the same sample and picks which is better.
      Fills: llm_pref_winner (A=no_prune / B=this config / Tie),
             llm_pref_reasoning

All modes look up scene description and question context from the source
dataset files — no need to re-run profile_sara.py.

Usage:
    python evaluation.py --csv results/sara_experiments.csv --mode reasoning
    python evaluation.py --csv results/sara_experiments.csv --mode coverage
    python evaluation.py --csv results/sara_experiments.csv --mode preference
    python evaluation.py --csv results/sara_experiments.csv --mode reasoning --judge-model Qwen/Qwen2.5-7B-Instruct
    python evaluation.py --csv results/sara_experiments.csv --mode reasoning --config-tag no_prune
    python evaluation.py --csv results/sara_experiments.csv --mode reasoning --dry-run
"""

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

REPO_ROOT      = _find_repo_root()
DATA_DIR       = REPO_ROOT / "data" / "EgoGazeVQA_full"
METADATA_CSV   = DATA_DIR / "metadata.csv"
EGO4D_JSON     = DATA_DIR / "ego4d.json"
EGOEXO_JSON    = DATA_DIR / "egoexo.json"
FRAMES_DIR     = DATA_DIR / "frames"

# ── New CSV columns added by this script ────────────────────────────────────
REASONING_COLS  = ["llm_judge_score", "llm_judge_reasoning"]
COVERAGE_COLS   = ["llm_coverage_scene", "llm_coverage_relevance",
                   "llm_coverage_logic", "llm_coverage_final",
                   "llm_coverage_breakdown"]
PREFERENCE_COLS = ["llm_pref_winner", "llm_pref_reasoning"]
ALL_EVAL_COLS   = REASONING_COLS + COVERAGE_COLS + PREFERENCE_COLS


# ── Dataset helpers ──────────────────────────────────────────────────────────

def _load_metadata() -> list[dict]:
    with open(METADATA_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _load_narration_db() -> dict[str, list[dict]]:
    """Load ego4d + egoexo narrations keyed by video_id."""
    db: dict[str, list[dict]] = {}
    for path in [EGO4D_JSON, EGOEXO_JSON]:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for vid_id, val in data.items():
            db[vid_id] = val.get("narrations", [])
    return db


def _build_scene_description(video_id: str, file_name: str,
                               narration_db: dict) -> str:
    """Return concatenated narration text for the clip's frame range."""
    narrations = narration_db.get(video_id, [])
    if not narrations:
        return ""

    # Parse frame range from file_name like "ego4d/uuid/123_1205.mp4"
    stem = Path(file_name).stem          # "123_1205"
    parts = stem.split("_")
    try:
        start_frame = int(parts[0])
        end_frame   = int(parts[-1])
    except ValueError:
        start_frame, end_frame = 0, 10**9

    texts = []
    for n in sorted(narrations, key=lambda x: x.get("timestamp_frame", 0)):
        tf = n.get("timestamp_frame", 0)
        if start_frame <= tf <= end_frame:
            t = n.get("narration_text", "").strip()
            t = t.replace("#C C", "The camera wearer").replace("#C", "The camera wearer").replace("#O", "Another person")
            if t and (not texts or texts[-1] != t):
                texts.append(t)
    return " ".join(texts)


def _build_sample_lookup(seed: int = 42, num_samples: int = 30) -> dict[int, dict]:
    """
    Re-run load_samples to get the same 30 clips used during profiling.
    Returns {sample_idx: metadata_row}.
    """
    all_meta = _load_metadata()
    available = []
    for row in all_meta:
        fn = row["file_name"].replace(".mp4", "")
        folder = FRAMES_DIR / fn
        if folder.exists() and list(folder.glob("*.jpg")):
            available.append(row)

    import random
    rng = random.Random(seed)
    rng.shuffle(available)
    selected = available[:num_samples]
    return {i: row for i, row in enumerate(selected)}


# ── Prompt templates ─────────────────────────────────────────────────────────

_PROMPT_REASONING = """\
You are evaluating the reasoning quality of a vision-language model on an \
egocentric video question-answering task.

Scene description (ground-truth narration of what happened):
{scene_description}

Question: {question}
Answer options: {answer_options}
Correct answer: {correct_answer}

Model's full response (reasoning + predicted answer):
{cot_output}

Rate the quality of the model's reasoning on a scale of 0–5:
  0 — No reasoning, or completely wrong / irrelevant
  1 — Attempts reasoning but makes fundamental errors
  2 — Partial understanding; misses key aspects of the scene or question
  3 — Reasonable reasoning with minor errors; mostly on track
  4 — Good reasoning; well-grounded in scene, addresses the question
  5 — Excellent; precise, well-justified, uses scene evidence effectively

Write 2–3 sentences evaluating the reasoning, then output exactly:
Score: X"""

_PROMPT_COVERAGE = """\
You are evaluating how thoroughly a vision-language model's reasoning covers \
key aspects of an egocentric video question.

Scene description (ground-truth narration):
{scene_description}

Question: {question}
Correct answer: {correct_answer}

Model's response:
{cot_output}

Score each dimension 0–5 and give one sentence of explanation per dimension.

Dimensions:
  Scene grounding   — Does the reasoning reference actual objects or actions \
from the scene description?
  Question relevance — Does the reasoning directly address what the question asks?
  Logical chain     — Is the reasoning coherent, structured, and well-justified?

Then give a final overall score (0–5).

Output format (follow exactly):
Scene grounding: X — [one sentence]
Question relevance: X — [one sentence]
Logical chain: X — [one sentence]
Final score: X"""

_PROMPT_PREFERENCE = """\
You are comparing two vision-language model responses to the same egocentric \
video question. One is from a baseline (no token pruning) and one uses \
visual token pruning ({config_b}).

Scene description (ground-truth narration):
{scene_description}

Question: {question}
Correct answer: {correct_answer}

Response A — baseline (no pruning):
{cot_a}

Response B — {config_b}:
{cot_b}

Which response shows better reasoning? Consider: scene grounding, logical \
coherence, and how well the reasoning supports the answer.

Output format (follow exactly):
Preference: A / B / Tie
Reason: [one sentence explaining your choice]"""


# ── Output parsers ────────────────────────────────────────────────────────────

def parse_reasoning(text: str):
    m = re.search(r"Score:\s*([0-5])", text)
    score = int(m.group(1)) if m else None
    idx = text.lower().rfind("score:")
    reasoning = text[:idx].strip() if idx >= 0 else text.strip()
    return score, reasoning.replace("\n", " ")


def parse_coverage(text: str):
    def extract(label):
        m = re.search(rf"{label}:\s*([0-5])\s*[—–-]\s*(.+)", text, re.IGNORECASE)
        if m:
            return int(m.group(1)), m.group(2).strip()
        return None, ""

    scene_score,   scene_reason   = extract("Scene grounding")
    rel_score,     rel_reason     = extract("Question relevance")
    logic_score,   logic_reason   = extract("Logical chain")
    m_final = re.search(r"Final score:\s*([0-5])", text, re.IGNORECASE)
    final = int(m_final.group(1)) if m_final else None

    breakdown = (
        f"Scene grounding: {scene_score} — {scene_reason} | "
        f"Question relevance: {rel_score} — {rel_reason} | "
        f"Logical chain: {logic_score} — {logic_reason}"
    )
    return scene_score, rel_score, logic_score, final, breakdown


def parse_preference(text: str):
    m = re.search(r"Preference:\s*(A|B|Tie)", text, re.IGNORECASE)
    winner = m.group(1).capitalize() if m else None
    m2 = re.search(r"Reason:\s*(.+)", text, re.IGNORECASE)
    reason = m2.group(1).strip() if m2 else text.strip()
    return winner, reason.replace("\n", " ")


# ── Judge model ───────────────────────────────────────────────────────────────

def load_judge(model_id: str, device: str):
    print(f"Loading judge: {model_id}  (device={device})")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    mdl.eval()
    print("Judge loaded.\n")
    return mdl, tok


def call_judge(model, tokenizer, prompt: str, device: str,
               max_new_tokens: int = 400) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_toks = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_toks, skip_special_tokens=True).strip()


# ── Per-mode evaluation ───────────────────────────────────────────────────────

def eval_reasoning(rows, sample_lookup, narration_db, model, tokenizer,
                   device, max_new_tokens, config_tag_filter, dry_run,
                   out_path=None, fieldnames=None, done_keys=None):
    targets = [
        r for r in rows
        if r.get("cot_output", "").strip()
        and r.get("qa_type", "") in ("causal", "spatial", "temporal")
        and (not config_tag_filter or config_tag_filter in r["config_tag"])
        and (done_keys is None or (r["config_tag"], r["sample_idx"]) not in done_keys)
    ]
    print(f"[reasoning] rows to evaluate: {len(targets)}")
    if dry_run:
        for r in targets:
            print(f"  would judge: sample={r['sample_idx']} config={r['config_tag']}")
        return

    for i, row in enumerate(targets):
        idx  = int(row["sample_idx"])
        meta = sample_lookup.get(idx, {})
        scene = _build_scene_description(
            row["video_id"], meta.get("file_name", ""), narration_db
        )
        prompt = _PROMPT_REASONING.format(
            scene_description=scene or "(not available)",
            question=meta.get("question", row.get("question", "")),
            answer_options=meta.get("answer_options", row.get("answer_options", "")),
            correct_answer=row["correct_answer"],
            cot_output=row["cot_output"][:1500],
        )
        print(f"  [{i+1}/{len(targets)}] sample={idx} qa={row['qa_type']} config={row['config_tag'][:35]}")
        resp = call_judge(model, tokenizer, prompt, device, max_new_tokens)
        score, reasoning = parse_reasoning(resp)
        if score is None:
            print(f"    WARNING: could not parse score — {resp[:80]!r}")
            row["llm_judge_score"]    = ""
            row["llm_judge_reasoning"] = resp[:500]
        else:
            print(f"    → score={score}  {reasoning[:80]!r}")
            row["llm_judge_score"]    = str(score)
            row["llm_judge_reasoning"] = reasoning[:500]
        if out_path and fieldnames:
            append_output_row(out_path, row, fieldnames)


def eval_coverage(rows, sample_lookup, narration_db, model, tokenizer,
                  device, max_new_tokens, config_tag_filter, dry_run,
                  out_path=None, fieldnames=None, done_keys=None):
    targets = [
        r for r in rows
        if r.get("cot_output", "").strip()
        and r.get("qa_type", "") in ("causal", "spatial", "temporal")
        and (not config_tag_filter or config_tag_filter in r["config_tag"])
        and (done_keys is None or (r["config_tag"], r["sample_idx"]) not in done_keys)
    ]
    print(f"[coverage] rows to evaluate: {len(targets)}")
    if dry_run:
        for r in targets:
            print(f"  would score: sample={r['sample_idx']} config={r['config_tag']}")
        return

    for i, row in enumerate(targets):
        idx  = int(row["sample_idx"])
        meta = sample_lookup.get(idx, {})
        scene = _build_scene_description(
            row["video_id"], meta.get("file_name", ""), narration_db
        )
        prompt = _PROMPT_COVERAGE.format(
            scene_description=scene or "(not available)",
            question=meta.get("question", row.get("question", "")),
            correct_answer=row["correct_answer"],
            cot_output=row["cot_output"][:1500],
        )
        print(f"  [{i+1}/{len(targets)}] sample={idx} qa={row['qa_type']} config={row['config_tag'][:35]}")
        resp = call_judge(model, tokenizer, prompt, device, max_new_tokens)
        s_scene, s_rel, s_logic, s_final, breakdown = parse_coverage(resp)
        print(f"    → scene={s_scene} rel={s_rel} logic={s_logic} final={s_final}")
        row["llm_coverage_scene"]     = str(s_scene) if s_scene is not None else ""
        row["llm_coverage_relevance"] = str(s_rel)   if s_rel   is not None else ""
        row["llm_coverage_logic"]     = str(s_logic) if s_logic is not None else ""
        row["llm_coverage_final"]     = str(s_final) if s_final is not None else ""
        row["llm_coverage_breakdown"] = breakdown[:600]
        if out_path and fieldnames:
            append_output_row(out_path, row, fieldnames)


def eval_preference(rows, sample_lookup, narration_db, model, tokenizer,
                    device, max_new_tokens, config_tag_filter, dry_run):
    # Build no_prune lookup: sample_idx → row
    baseline = {
        int(r["sample_idx"]): r
        for r in rows
        if r["config_tag"] == "no_prune"
    }
    if not baseline:
        print("[preference] ERROR: no 'no_prune' rows found in CSV.")
        return

    targets = [
        r for r in rows
        if r["config_tag"] != "no_prune"
        and r.get("cot_output", "").strip()
        and r.get("llm_pref_winner", "").strip() == ""
        and r.get("qa_type", "") in ("causal", "spatial", "temporal")
        and (not config_tag_filter or config_tag_filter in r["config_tag"])
    ]
    print(f"[preference] rows to evaluate: {len(targets)}")
    if dry_run:
        for r in targets:
            print(f"  would compare: sample={r['sample_idx']} config={r['config_tag']} vs no_prune")
        return

    for i, row in enumerate(targets):
        idx  = int(row["sample_idx"])
        base = baseline.get(idx)
        if not base or not base.get("cot_output", "").strip():
            print(f"  [{i+1}/{len(targets)}] sample={idx} SKIP — no baseline CoT")
            continue

        meta = sample_lookup.get(idx, {})
        scene = _build_scene_description(
            row["video_id"], meta.get("file_name", ""), narration_db
        )
        prompt = _PROMPT_PREFERENCE.format(
            scene_description=scene or "(not available)",
            question=meta.get("question", row.get("question", "")),
            correct_answer=row["correct_answer"],
            cot_a=base["cot_output"][:800],
            cot_b=row["cot_output"][:800],
            config_b=row["config_tag"],
        )
        print(f"  [{i+1}/{len(targets)}] sample={idx} config={row['config_tag'][:35]} vs no_prune")
        resp = call_judge(model, tokenizer, prompt, device, max_new_tokens)
        winner, reason = parse_preference(resp)
        print(f"    → winner={winner}  {reason[:80]!r}")
        row["llm_pref_winner"]    = winner or ""
        row["llm_pref_reasoning"] = reason[:500]


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(rows, mode):
    print(f"\n{'═'*60}")
    print(f"  Summary — mode: {mode}")

    if mode == "reasoning":
        scored = [r for r in rows if r.get("llm_judge_score", "").strip().lstrip("-").isdigit()]
        by_tag: dict = {}
        for r in scored:
            by_tag.setdefault(r["config_tag"], []).append(int(r["llm_judge_score"]))
        print(f"  Scored rows: {len(scored)}")
        for tag, vals in sorted(by_tag.items()):
            print(f"  {tag:45s}  mean={sum(vals)/len(vals):.2f}  n={len(vals)}")

    elif mode == "coverage":
        scored = [r for r in rows if r.get("llm_coverage_final", "").strip().isdigit()]
        by_tag: dict = {}
        for r in scored:
            by_tag.setdefault(r["config_tag"], []).append(int(r["llm_coverage_final"]))
        print(f"  Scored rows: {len(scored)}")
        for tag, vals in sorted(by_tag.items()):
            print(f"  {tag:45s}  mean_final={sum(vals)/len(vals):.2f}  n={len(vals)}")

    elif mode == "preference":
        judged = [r for r in rows if r.get("llm_pref_winner", "").strip()]
        by_tag: dict = {}
        for r in judged:
            by_tag.setdefault(r["config_tag"], {"A": 0, "B": 0, "Tie": 0})
            w = r["llm_pref_winner"]
            by_tag[r["config_tag"]][w] = by_tag[r["config_tag"]].get(w, 0) + 1
        print(f"  Judged rows: {len(judged)}  (A=no_prune wins, B=pruned wins)")
        for tag, counts in sorted(by_tag.items()):
            total = sum(counts.values())
            print(f"  {tag:45s}  A={counts.get('A',0)}  B={counts.get('B',0)}  Tie={counts.get('Tie',0)}  n={total}")

    print(f"{'═'*60}\n")


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def _output_fieldnames(rows: list[dict]) -> list[str]:
    cols = list(rows[0].keys()) if rows else []
    for col in ALL_EVAL_COLS:
        if col not in cols:
            cols.append(col)
    return cols


def load_done_keys(out_path: Path, mode: str) -> set[tuple]:
    """Return set of (config_tag, sample_idx) already scored in the output CSV."""
    if not out_path.exists():
        return set()
    check_col = {
        "reasoning":  "llm_judge_score",
        "coverage":   "llm_coverage_final",
        "preference": "llm_pref_winner",
    }[mode]
    done = set()
    with open(out_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get(check_col, "").strip():
                done.add((r["config_tag"], r["sample_idx"]))
    return done


def append_output_row(out_path: Path, row: dict, fieldnames: list[str]):
    is_new = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_csv(csv_path: Path, rows: list[dict]):
    """Final full rewrite — called at end as a safety net."""
    if not rows:
        return
    fieldnames = _output_fieldnames(rows)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Written → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",            type=Path, required=True)
    ap.add_argument("--output-csv",     type=Path, default=None,
                    help="Write results to a new file instead of overwriting --csv.")
    ap.add_argument("--mode",           type=str,  default="reasoning",
                    choices=["reasoning", "coverage", "preference"],
                    help="Which evaluation to run (default: reasoning).")
    ap.add_argument("--judge-model",    type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int,  default=400)
    ap.add_argument("--num-samples",    type=int,  default=30,
                    help="Must match --num-samples used in profile_sara.py.")
    ap.add_argument("--seed",           type=int,  default=42,
                    help="Must match --seed used in profile_sara.py.")
    ap.add_argument("--config-tag",     type=str,  default=None,
                    help="Only evaluate rows whose config_tag contains this.")
    ap.add_argument("--dry-run",        action="store_true")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {args.csv}")

    print("Building sample lookup...")
    sample_lookup = _build_sample_lookup(seed=args.seed, num_samples=args.num_samples)
    print(f"  {len(sample_lookup)} samples recovered.")

    print("Loading narration database...")
    narration_db = _load_narration_db()
    print(f"  {len(narration_db)} videos indexed.")

    if args.dry_run:
        device = "cpu"
        model = tokenizer = None
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        model, tokenizer = load_judge(args.judge_model, device)

    if args.mode == "reasoning":
        eval_reasoning(rows, sample_lookup, narration_db, model, tokenizer,
                       device, args.max_new_tokens, args.config_tag, args.dry_run)
    elif args.mode == "coverage":
        eval_coverage(rows, sample_lookup, narration_db, model, tokenizer,
                      device, args.max_new_tokens, args.config_tag, args.dry_run)
    elif args.mode == "preference":
        eval_preference(rows, sample_lookup, narration_db, model, tokenizer,
                        device, args.max_new_tokens, args.config_tag, args.dry_run)

    if not args.dry_run:
        out_path = args.output_csv or args.csv
        write_csv(out_path, rows)
        print_summary(rows, args.mode)


if __name__ == "__main__":
    main()
