"""Task loading and answer verification for the Week-1 baselines.

Datasets
--------
GSM8K
    Hugging Face ``openai/gsm8k``, config ``main``, test split
    (1,319 problems). Gold answers are the numeric value after the
    ``####`` marker in the reference solution.

ProsQA
    ProsQA ships with the Coconut release (Hao et al., 2024) --
    the anchor paper for this study -- as JSON files in the
    official repository ``facebookresearch/coconut``
    (``data/prosqa_{train,valid,test}.json``; the test split has
    500 items). That repository is the canonical source used here.
    (Third-party Hugging Face re-uploads under the name "prosqa"
    exist but are not the authors' distribution, so they are not
    used.) Each item is a graph-reasoning question whose gold
    ``answer`` is a full statement such as ``"Sally is a
    sterpus."``; ``target`` / ``neg_target`` index the correct and
    the distractor concept in ``idx_to_symbol``.

    If the canonical files cannot be fetched, the loader raises --
    it never silently substitutes a different dataset.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .netenv import sanitize_proxy_env

sanitize_proxy_env()

CACHE_DIR = Path.home() / ".cache" / "latent-agent-cost"

PROSQA_URLS = {
    split: ("https://raw.githubusercontent.com/facebookresearch/"
            f"coconut/main/data/prosqa_{split}.json")
    for split in ("train", "valid", "test")
}


@dataclass
class Task:
    task_id: str
    dataset: str
    question: str
    # GSM8K: float gold. ProsQA: the gold answer statement.
    gold: object
    # ProsQA only: correct / distractor concept strings.
    target_concept: str | None = None
    neg_concept: str | None = None


# ---------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------
def load_gsm8k(split: str = "test", limit: int | None = None) -> list[Task]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    tasks = []
    for i, row in enumerate(ds):
        gold = float(row["answer"].split("####")[-1].strip()
                     .replace(",", ""))
        tasks.append(Task(task_id=f"gsm8k-{split}-{i}",
                          dataset="gsm8k",
                          question=row["question"].strip(),
                          gold=gold))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


# ---------------------------------------------------------------------
# ProsQA (canonical: Coconut authors' GitHub distribution)
# ---------------------------------------------------------------------
def _fetch_prosqa(split: str) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"prosqa_{split}.json"
    if not path.exists():
        urllib.request.urlretrieve(PROSQA_URLS[split], path)
    return json.loads(path.read_text())


def load_prosqa(split: str = "test", limit: int | None = None) -> list[Task]:
    rows = _fetch_prosqa(split)
    tasks = []
    for i, row in enumerate(rows):
        idx_to_symbol = row.get("idx_to_symbol") or []

        def _symbol(idx):
            if idx is None:
                return None
            if isinstance(idx_to_symbol, dict):
                return idx_to_symbol.get(str(idx)) or idx_to_symbol.get(idx)
            try:
                return idx_to_symbol[int(idx)]
            except (ValueError, IndexError):
                return None

        target = _symbol(row.get("target"))
        neg = _symbol(row.get("neg_target"))
        tasks.append(Task(task_id=f"prosqa-{split}-{i}",
                          dataset="prosqa",
                          question=row["question"].strip(),
                          gold=row["answer"].strip(),
                          target_concept=target,
                          neg_concept=neg))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------
def extract_final_answer(text: str) -> str:
    """The text after the last ``####`` marker, or the last line."""
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _parse_number(s: str) -> float | None:
    m = re.findall(r"-?\$?\d[\d,]*\.?\d*", s)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").replace("$", ""))
    except ValueError:
        return None


def verify_gsm8k(completion: str, gold: float) -> bool:
    """Final-answer numeric match (the GSM8K convention)."""
    pred = _parse_number(extract_final_answer(completion))
    if pred is None:
        # Fall back to the last number anywhere in the completion.
        pred = _parse_number(completion)
    return pred is not None and abs(pred - gold) < 1e-4


def _concept_of(statement: str) -> str:
    """'Sally is a sterpus.' -> 'sterpus'."""
    s = statement.strip().rstrip(".").lower()
    m = re.search(r"\bis an?\s+(.+)$", s)
    return (m.group(1) if m else s).strip()


def verify_prosqa(completion: str, task: Task) -> bool:
    """Solved iff the final answer names the target concept and
    does not name the distractor concept."""
    final = extract_final_answer(completion).lower()
    target = (task.target_concept or _concept_of(str(task.gold))).lower()
    neg = (task.neg_concept or "").lower()
    if not target:
        return False
    if neg and neg in final:
        return False
    return target in final


def verify(task: Task, completion: str) -> bool:
    if task.dataset == "gsm8k":
        return verify_gsm8k(completion, float(task.gold))
    if task.dataset == "prosqa":
        return verify_prosqa(completion, task)
    raise ValueError(f"no verifier for dataset {task.dataset!r}")
