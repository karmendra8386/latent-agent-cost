#!/usr/bin/env python3
"""Build Coconut-curriculum training data for ProsQA.

Source (the only one used): the canonical ProsQA distribution --
the Coconut authors' repository ``facebookresearch/coconut``,
``data/prosqa_{train,valid,test}.json``, fetched through
:func:`latent_cost.data._fetch_prosqa` (the same loader the
evaluation harness uses, same cache).

Why this source is legitimate training data: every ProsQA item
ships with the dataset's OWN reasoning chain in its ``steps``
field -- the proof path through the item's logic graph (``edges``
/ ``root`` / ``target``), written out as statements, e.g. ::

    steps:  ["Sally is a scrompus.",
             "Every scrompus is a rempus.",
             "Every rempus is a sterpus."]
    answer: "Sally is a sterpus."

These chains are the dataset's annotations, generated with the
items from the graph -- not model output, not hand-written for
this project. (This is also the data Coconut itself trains on.)

Validation: a row is kept only if its chain is mechanically
coherent --

* ``question`` / ``answer`` are non-empty strings and ``steps``
  is a non-empty list of non-empty strings;
* the first step is a fact about the queried individual
  (``idx_to_symbol[root]``);
* consecutive steps chain: the object concept of step *i* is the
  subject concept of step *i+1* ("Sally is a **scrompus**" ->
  "Every **scrompus** is a rempus");
* the last step's object concept is the answer's concept, and the
  answer is about the same individual as the first step.

Rows failing any check are dropped and counted by reason -- never
repaired, never rewritten. Output is one JSON object per line,
``{"question": ..., "steps": [...], "answer": ...}`` -- exactly
the schema :mod:`latent_cost.coconut` consumes. Answers are kept
in the dataset's own statement form; the ``#### <answer>`` line
is added by the trainer when it renders an example, matching the
harness's completion format.

Usage:
    .venv/bin/python scripts/prepare_prosqa_train.py \
        --split train --out data/prosqa_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_cost.coconut import TrainExample
from latent_cost.data import _fetch_prosqa

_RULE_RE = re.compile(r"^Every\s+(.+?)\s+is an?\s+(.+?)\.?$", re.IGNORECASE)
_FACT_RE = re.compile(r"^(.+?)\s+is an?\s+(.+?)\.?$", re.IGNORECASE)


def _subject_object(statement: str) -> tuple[str, str] | None:
    """(subject, object concept) of one chain statement, lowered.

    Rules ("Every scrompus is a rempus.") and facts ("Sally is a
    scrompus.") share the shape ``<subject> is a <concept>`` once
    the leading "Every" is stripped.
    """
    s = statement.strip()
    m = _RULE_RE.match(s) or _FACT_RE.match(s)
    if not m:
        return None
    return m.group(1).strip().lower(), m.group(2).strip().lower()


def validate_row(row: dict) -> TrainExample:
    """One canonical ProsQA row -> TrainExample, or ValueError
    with a short machine-readable reason."""
    question = row.get("question")
    answer = row.get("answer")
    steps = row.get("steps")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing-question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("missing-answer")
    if (not isinstance(steps, list) or not steps
            or any(not isinstance(s, str) or not s.strip()
                   for s in steps)):
        raise ValueError("missing-steps")
    steps = [s.strip() for s in steps]
    answer = answer.strip()

    parsed = [_subject_object(s) for s in steps]
    if any(p is None for p in parsed):
        raise ValueError("unparseable-step")
    ans = _subject_object(answer)
    if ans is None:
        raise ValueError("unparseable-answer")

    # The chain must start from the queried individual (the
    # graph root), when the row exposes it.
    idx_to_symbol = row.get("idx_to_symbol") or []
    root = row.get("root")
    if root is not None and idx_to_symbol:
        try:
            root_name = str(idx_to_symbol[int(root)]).strip().lower()
        except (ValueError, IndexError, TypeError):
            root_name = None
        if root_name and parsed[0][0] != root_name:
            raise ValueError("chain-not-from-root")

    # The answer must be about the individual the chain starts
    # from, and consecutive steps must chain concept -> concept.
    if ans[0] != parsed[0][0]:
        raise ValueError("answer-individual-mismatch")
    for (_, obj), (next_subj, _) in zip(parsed, parsed[1:]):
        if obj != next_subj:
            raise ValueError("chain-break")
    if parsed[-1][1] != ans[1]:
        raise ValueError("chain-does-not-reach-answer")

    return TrainExample(question=question.strip(), steps=steps,
                        answer=answer)


def build_examples(rows: list[dict]) -> tuple[list[TrainExample],
                                              Counter]:
    """Validate canonical rows; return (kept, drop-reason counts)."""
    kept: list[TrainExample] = []
    dropped: Counter = Counter()
    for row in rows:
        try:
            kept.append(validate_row(row))
        except ValueError as e:
            dropped[str(e)] += 1
    return kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    choices=["train", "valid", "test"],
                    help="canonical ProsQA split to convert")
    ap.add_argument("--out", required=True,
                    help="output JSONL path")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on rows read (before validation)")
    args = ap.parse_args()

    rows = _fetch_prosqa(args.split)
    if args.limit is not None:
        rows = rows[:args.limit]
    kept, dropped = build_examples(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for ex in kept:
            fh.write(json.dumps(ex.to_dict()) + "\n")

    print(f"split: {args.split}  rows read: {len(rows)}")
    print(f"kept: {len(kept)}  dropped: {sum(dropped.values())}")
    for reason, n in sorted(dropped.items()):
        print(f"  dropped[{reason}]: {n}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
