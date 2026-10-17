"""Unit tests for the Week-2 conditions: latent, zero-latent,
perturbation, pause-token control, and the GPT-2 control pair.

Pipeline/accounting tests run on the MockRunner (deterministic,
no model download). The latent-feedback transform is tested on
real tensors. Two guarded integration tests exercise the real
HFRunner paths (cached Qwen2.5-0.5B; a from-config random GPT-2)
and skip when the artifacts are not available locally.
"""

import json
from pathlib import Path

import pytest

from latent_cost.data import Task
from latent_cost.harness import CONDITION_MODES, build_messages, run_condition
from latent_cost.metrics import (AttemptLog, ModelSpec, PriceCard,
                                 StepLog, compute_metrics)
from latent_cost.models import (LATENT_VARIANTS, MockRunner,
                                transform_latent_state)

torch = pytest.importorskip("torch")

CARD = PriceCard(input_usd_per_token=0.01, output_usd_per_token=0.02,
                 label="test-card")
SPEC = ModelSpec(name="synthetic", n_params=1_000)
K = 8


def prosqa_task(i=0):
    return Task(task_id=f"prosqa-test-{i}", dataset="prosqa",
                question="Tom is a gorpus. Every gorpus is a wumpus. "
                         "Is Tom a wumpus or a lorpus?",
                gold="Tom is a wumpus.",
                target_concept="wumpus", neg_concept="lorpus")


def mock_attempts(condition, tasks=None, seeds=(0,)):
    tasks = tasks if tasks is not None else [prosqa_task()]
    return run_condition(tasks, runner=MockRunner(), condition=condition,
                         seeds=seeds, latent_steps=K)


# ---------------------------------------------------------------------
# Latent-feedback transform
# ---------------------------------------------------------------------
def test_transform_identity_returns_state():
    h = torch.randn(1, 1, 64)
    out = transform_latent_state(h, "identity")
    assert torch.equal(out, h)


def test_transform_zero_actually_zeroes():
    h = torch.randn(1, 1, 64)
    assert not torch.equal(h, torch.zeros_like(h))  # sanity: h nonzero
    out = transform_latent_state(h, "zero")
    assert out.shape == h.shape and out.dtype == h.dtype
    assert torch.equal(out, torch.zeros_like(h))


def test_transform_perturb_adds_gaussian_noise_at_scale():
    torch.manual_seed(1234)
    h = torch.zeros(1, 1, 100_000)
    out = transform_latent_state(h, "perturb", sigma=0.5,
                                 torch_module=torch)
    noise = out - h
    assert noise.std().item() == pytest.approx(0.5, rel=0.02)
    assert noise.abs().max() > 0  # it actually moved


def test_transform_perturb_zero_sigma_is_identity():
    h = torch.randn(2, 1, 32)
    out = transform_latent_state(h, "perturb", sigma=0.0,
                                 torch_module=torch)
    assert torch.equal(out, h)


def test_transform_unknown_variant_rejected():
    with pytest.raises(ValueError):
        transform_latent_state(torch.zeros(1, 1, 4), "sideways")


# ---------------------------------------------------------------------
# MockRunner mode accounting
# ---------------------------------------------------------------------
def test_mock_latent_records_budget_and_no_reasoning_text():
    msgs = build_messages(prosqa_task())
    g = MockRunner().generate(msgs, seed=0, mode="latent", latent_steps=K)
    assert g.latent_steps == K and g.pause_tokens == 0
    # The completion is only the decoded answer segment: no
    # reasoning lines exist to audit.
    assert g.text.startswith("####")
    kinds = [(s.kind, s.trace_text) for s in g.steps]
    assert ("latent_reasoning", None) in kinds
    assert any(k == "final_answer" and t for k, t in kinds)


@pytest.mark.parametrize("condition", ["latent_zero", "latent_perturb"])
def test_mock_ablations_carry_variant_meta(condition):
    msgs = build_messages(prosqa_task())
    mode = CONDITION_MODES[condition]
    g = MockRunner().generate(msgs, seed=0, mode=mode, latent_steps=K,
                              perturb_scale=0.25)
    assert g.latent_steps == K
    assert g.meta["latent_variant"] == LATENT_VARIANTS[mode]
    if condition == "latent_perturb":
        assert g.meta["perturb_scale"] == 0.25


def test_mock_pause_budget_matches_latent_budget():
    """The outline's key control property: the pause condition
    spends exactly the latent condition's step budget, on filler
    tokens instead of latent states."""
    msgs = build_messages(prosqa_task())
    runner = MockRunner()
    g_lat = runner.generate(msgs, seed=0, mode="latent", latent_steps=K)
    g_pause = runner.generate(msgs, seed=0, mode="pause", latent_steps=K)
    assert g_pause.pause_tokens == K == g_lat.latent_steps
    # Pause positions are visible output positions: they are
    # counted in completion_tokens, unlike latent steps.
    assert g_pause.completion_tokens == (
        len(g_pause.text.split()) + K)
    assert g_lat.completion_tokens == len(g_lat.text.split())


def test_mock_unknown_mode_rejected():
    with pytest.raises(ValueError):
        MockRunner().generate(build_messages(prosqa_task()), seed=0,
                              mode="telepathy")


# ---------------------------------------------------------------------
# run_condition integration (all Week-2 conditions, mock path)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("condition,expected_latent", [
    ("explicit_cot", 0), ("latent", K), ("latent_zero", K),
    ("latent_perturb", K), ("pause_token", 0),
    ("gpt2_cot", 0), ("gpt2_latent", K),
])
def test_run_condition_accounting(condition, expected_latent):
    attempts = mock_attempts(condition, seeds=(0, 1))
    assert len(attempts) == 2
    for a in attempts:
        assert a.condition == condition
        assert a.latent_steps == expected_latent
    r = compute_metrics(attempts, price_card=CARD, model=SPEC)
    assert r.condition == condition and r.attempts == 2
    # FLOPs charge latent steps like token positions.
    if expected_latent:
        exp = 2 * SPEC.n_params * (
            attempts[0].prompt_tokens + attempts[0].completion_tokens
            + K)
        assert attempts[0].flops(SPEC) == exp


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        run_condition([prosqa_task()], runner=MockRunner(),
                      condition="hybrid")


def test_latent_auditability_reflects_traceless_reasoning():
    """Latent reasoning carries no explicit trace (recorded in the
    step log); the auditability rubric scores action-bearing
    steps, so a latent attempt whose decoded answer line exists
    still scores its final answer -- and one without an answer
    line scores 0."""
    attempts = mock_attempts("latent")
    a = attempts[0]
    latent_steps = [s for s in a.steps if s.kind == "latent_reasoning"]
    assert latent_steps and all(
        s.trace_text is None for s in latent_steps)
    r = compute_metrics(attempts, price_card=CARD, model=SPEC)
    assert r.auditability == pytest.approx(1.0)

    bare = AttemptLog(
        task_id="t", dataset="prosqa", condition="latent", seed=0,
        prompt_tokens=10, completion_tokens=0, latency_s=1.0,
        solved=False, latent_steps=K,
        steps=[StepLog(kind="latent_reasoning", trace_text=None),
               StepLog(kind="final_answer", trace_text="")])
    r2 = compute_metrics([bare], price_card=CARD, model=SPEC)
    assert r2.auditability == pytest.approx(0.0)


def test_attempt_log_carries_variant_meta(tmp_path):
    log = tmp_path / "att.jsonl"
    run_condition([prosqa_task()], runner=MockRunner(),
                  condition="latent_perturb", seeds=(0,),
                  latent_steps=K, perturb_scale=0.3, log_path=log)
    rec = json.loads(log.read_text().strip())
    assert rec["latent_variant"] == "perturb"
    assert rec["perturb_scale"] == 0.3
    assert rec["latent_steps"] == K
    step_kinds = {s["kind"]: s["trace_text"] for s in rec["steps"]}
    assert step_kinds["latent_reasoning"] is None


# ---------------------------------------------------------------------
# Real-runner integration (guarded; skip when artifacts absent)
# ---------------------------------------------------------------------
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


@pytest.mark.skipif(
    not (HF_CACHE / "models--Qwen--Qwen2.5-0.5B-Instruct").exists(),
    reason="Qwen2.5-0.5B-Instruct not in the local HF cache")
def test_hf_latent_and_zero_paths_on_small_model():
    from latent_cost.models import HFRunner
    runner = HFRunner("Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
    msgs = build_messages(prosqa_task())
    for mode in ("latent", "latent_zero", "latent_perturb", "pause"):
        g = runner.generate(msgs, seed=0, max_new_tokens=8, mode=mode,
                            latent_steps=4, perturb_scale=0.1)
        assert isinstance(g.text, str)
        if mode == "pause":
            assert g.pause_tokens == 4
            assert g.completion_tokens >= 4
        else:
            assert g.latent_steps == 4
            fin = [s for s in g.steps if s.kind == "final_answer"]
            assert fin and fin[0].trace_text is not None


@pytest.mark.skipif(
    not (HF_CACHE / "models--openai-community--gpt2").exists(),
    reason="gpt2 not in the local HF cache")
def test_gpt2_control_paths_from_config():
    """GPT-2 control: random-weight model built from the cached
    config (pipeline validation of the no-chat-template prompt
    path and the GPT-2 backbone adapter -- not a result)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from latent_cost.models import HFRunner
    runner = object.__new__(HFRunner)
    runner._torch = torch
    runner.name = "gpt2 (from-config, random weights)"
    runner.device = "cpu"
    from transformers import AutoTokenizer
    runner.tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/gpt2")
    cfg = AutoConfig.from_pretrained("openai-community/gpt2")
    runner.model = AutoModelForCausalLM.from_config(
        cfg, dtype=torch.float32)
    runner.model.eval()
    runner._n_params = sum(p.numel()
                           for p in runner.model.parameters())
    # GPT-2 predates chat templates: the prompt must render
    # without one (or, if a template exists, still render).
    text = runner._prompt_text(build_messages(prosqa_task()))
    assert "wumpus" in text and "####" in text
    msgs = build_messages(prosqa_task())
    g = runner.generate(msgs, seed=0, max_new_tokens=6,
                        mode="explicit")
    assert isinstance(g.text, str) and g.latent_steps == 0
    g = runner.generate(msgs, seed=0, max_new_tokens=6,
                        mode="latent", latent_steps=3)
    assert g.latent_steps == 3
