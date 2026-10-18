"""Tests for the Coconut curriculum trainer and the ProsQA
training-data preparation.

Data-prep tests run on fabricated canonical-format rows (tiny
fixtures with invented names -- the only fabricated chains in
the project) plus, when the canonical files are in the local
cache, the real rows. Segmentation / schedule tests use a
character-level stub tokenizer so they need no downloads. The
end-to-end training test uses a tiny from-config GPT-2 with the
real GPT-2 tokenizer and skips if that tokenizer is unavailable.
"""

import json
import math
from pathlib import Path

import pytest

from latent_cost.coconut import (TrainExample, completion_segments,
                                 completion_text, count_positions,
                                 latent_positions, load_examples_jsonl,
                                 replaced_steps, tokenize_completion)
from scripts.prepare_prosqa_train import build_examples, validate_row

FIXTURE = (Path(__file__).parent / "fixtures" / "prosqa_tiny.jsonl")


# ---------------------------------------------------------------------
# Data preparation: validation invariants
# ---------------------------------------------------------------------
def canonical_row(**overrides):
    row = {
        "question": "Every gorpus is a wumpus. Tom is a gorpus. "
                    "Is Tom a wumpus or a lorpus?",
        "answer": "Tom is a wumpus.",
        "steps": ["Tom is a gorpus.", "Every gorpus is a wumpus."],
        "idx_to_symbol": ["Tom", "gorpus", "wumpus", "lorpus"],
        "root": 0, "target": 2, "neg_target": 3,
    }
    row.update(overrides)
    return row


def test_validate_row_accepts_coherent_chain():
    ex = validate_row(canonical_row())
    assert ex.answer == "Tom is a wumpus."
    assert ex.steps[0] == "Tom is a gorpus."


def test_build_examples_drops_each_broken_kind():
    rows = [
        canonical_row(),  # the one good row
        canonical_row(steps=[]),                       # no chain
        canonical_row(steps=["Tom is a gorpus.",
                             "Every lorpus is a wumpus."]),  # break
        canonical_row(answer="Tom is a lorpus."),      # not reached
        canonical_row(steps=["Sally is a gorpus.",
                             "Every gorpus is a wumpus."]),  # not root
        canonical_row(answer=""),                      # no answer
        canonical_row(question=""),                    # no question
    ]
    kept, dropped = build_examples(rows)
    assert len(kept) == 1
    assert dropped["missing-steps"] == 1
    assert dropped["chain-break"] == 1
    assert dropped["chain-does-not-reach-answer"] == 1
    assert dropped["chain-not-from-root"] == 1
    assert dropped["missing-answer"] == 1
    assert dropped["missing-question"] == 1


def test_prepared_schema_is_exactly_the_trainer_schema():
    kept, _ = build_examples([canonical_row()])
    d = kept[0].to_dict()
    assert set(d) == {"question", "steps", "answer"}
    assert isinstance(d["steps"], list)


def test_real_canonical_rows_validate_when_cached():
    """The dataset's own chains must pass the mechanical checks
    (guarded on the loader's cache; no network in tests)."""
    cache = Path.home() / ".cache" / "latent-agent-cost"
    path = cache / "prosqa_test.json"
    if not path.exists():
        pytest.skip("canonical ProsQA files not in the local cache")
    rows = json.loads(path.read_text())[:50]
    kept, dropped = build_examples(rows)
    assert len(kept) == 50, f"unexpected drops: {dict(dropped)}"


def test_fixture_file_loads_with_trainer_schema():
    examples = load_examples_jsonl(FIXTURE)
    assert len(examples) == 4
    for ex in examples:
        assert ex.question and ex.steps and ex.answer
        # The fixture obeys the same invariants as prepared data.
        validate_row({"question": ex.question, "answer": ex.answer,
                      "steps": ex.steps,
                      "idx_to_symbol": [], "root": None})


# ---------------------------------------------------------------------
# Curriculum schedule
# ---------------------------------------------------------------------
def test_replaced_steps_caps_at_chain_length():
    assert replaced_steps(0, 4) == 0
    assert replaced_steps(2, 4) == 2
    assert replaced_steps(9, 4) == 4  # short chain: fully replaced
    with pytest.raises(ValueError):
        replaced_steps(-1, 4)


def test_latent_positions_scale_with_thoughts_per_step():
    assert latent_positions(0, 5) == 0
    assert latent_positions(2, 5) == 2
    assert latent_positions(2, 5, thoughts_per_step=3) == 6
    assert latent_positions(9, 3, thoughts_per_step=2) == 6
    with pytest.raises(ValueError):
        latent_positions(1, 3, thoughts_per_step=0)


# ---------------------------------------------------------------------
# Completion segmentation (stub tokenizer, no downloads)
# ---------------------------------------------------------------------
class CharTokenizer:
    """One token per character, with exact offset mapping --
    enough to pin the segment-assignment logic deterministically."""

    eos_token_id = 0
    chat_template = None

    def __call__(self, text, return_offsets_mapping=False,
                 add_special_tokens=True):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def test_segments_tile_the_completion_text():
    ex = TrainExample(question="q", steps=["A is a b.", "Every b is a c."],
                      answer="A is a c.")
    text = completion_text(ex)
    assert text == "A is a b.\nEvery b is a c.\n#### A is a c."
    segs = completion_segments(ex)
    assert [k for k, _, _ in segs] == ["step", "step", "answer"]
    assert segs[0][1] == 0 and segs[-1][2] == len(text)
    for (_, s0, e0), (_, s1, _e1) in zip(segs, segs[1:]):
        assert e0 == s1  # no gaps, no overlaps


def test_tokenize_completion_partitions_all_tokens():
    ex = TrainExample(question="q",
                      steps=["A is a b.", "Every b is a c.",
                             "Every c is a d."],
                      answer="A is a d.")
    step_ids, answer_ids = tokenize_completion(CharTokenizer(), ex)
    full_ids = [ord(c) for c in completion_text(ex)]
    flat = [t for s in step_ids for t in s] + answer_ids
    assert flat == full_ids  # a partition: nothing lost or duplicated
    assert len(step_ids) == 3 and all(step_ids)
    # Step boundaries land on the segment texts exactly.
    assert bytes(step_ids[0]).decode() == "A is a b."
    assert bytes(step_ids[1]).decode() == "\nEvery b is a c."
    assert bytes(answer_ids).decode() == "\n#### A is a d."


# ---------------------------------------------------------------------
# End-to-end: tiny GPT-2, fixture data, harness reload
# ---------------------------------------------------------------------
def _gpt2_tokenizer_or_skip():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("openai-community/gpt2")
    except Exception:
        pytest.skip("GPT-2 tokenizer unavailable locally")


def test_tiny_end_to_end_train_save_and_harness_load(tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import AutoConfig, AutoModelForCausalLM

    from latent_cost.coconut import (example_loss, save_checkpoint,
                                     train_curriculum)

    tokenizer = _gpt2_tokenizer_or_skip()
    examples = load_examples_jsonl(FIXTURE)

    config = AutoConfig.from_pretrained("openai-community/gpt2")
    config.n_layer, config.n_embd, config.n_head = 2, 64, 2
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)

    # The loss is a finite scalar at both ends of the curriculum,
    # including the stage that replaces whole short chains.
    for stage in (0, 3, 9):
        loss = example_loss(model, tokenizer, examples[0], stage=stage)
        assert loss.dim() == 0 and math.isfinite(float(loss))

    history = train_curriculum(
        model, tokenizer, examples, stages=[0, 1], epochs_per_stage=1,
        lr=1e-3, grad_accum=2, seed=0, log=lambda *a: None)
    assert [h["stage"] for h in history] == [0, 1]
    assert all(h["examples"] == len(examples) for h in history)
    assert all(math.isfinite(h["loss"]) for h in history)

    ckpt = save_checkpoint(model, tokenizer, tmp_path / "ckpt",
                           meta={"smoke": True})
    assert (ckpt / "config.json").exists()
    assert (ckpt / "coconut_training.json").exists()

    # The harness loads the checkpoint directory unchanged and
    # both GPT-2 conditions run through it (tiny token budgets:
    # pipeline validation, not a result).
    from latent_cost.data import Task
    from latent_cost.harness import build_messages
    from latent_cost.models import HFRunner
    runner = HFRunner(str(ckpt), dtype="float32", device="cpu")
    task = Task(task_id="prosqa-test-0", dataset="prosqa",
                question=examples[0].question, gold=examples[0].answer,
                target_concept="lorpus", neg_concept="sterpus")
    msgs = build_messages(task)
    g_cot = runner.generate(msgs, seed=0, max_new_tokens=4,
                            mode="explicit")
    assert isinstance(g_cot.text, str) and g_cot.latent_steps == 0
    g_lat = runner.generate(msgs, seed=0, max_new_tokens=4,
                            mode="latent", latent_steps=3)
    assert g_lat.latent_steps == 3


def test_count_positions_grows_with_stage():
    tokenizer = CharTokenizer()
    ex = load_examples_jsonl(FIXTURE)[0]
    base = count_positions(tokenizer, ex, stage=0)
    later = count_positions(tokenizer, ex, stage=2)
    # Replacing two steps removes their tokens but adds two
    # latent positions; net effect is shorter, never longer.
    assert later < base
