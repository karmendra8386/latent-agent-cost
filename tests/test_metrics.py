"""Unit tests for the five-metric math, on synthetic run logs.

Every expectation below is hand-computed from the definitions in
latent_cost/metrics.py (which restate Section 3 of the outline).
"""

import math

import pytest

from latent_cost.metrics import (
    AttemptLog, ModelSpec, PriceCard, StepLog,
    attempt_from_tokenecon_record, compute_metrics,
)

MODEL = ModelSpec(name="synthetic-1k", n_params=1_000)
CARD = PriceCard(input_usd_per_token=0.01, output_usd_per_token=0.02,
                 label="test-card")


def att(task="t", solved=True, pt=10, ct=5, lat=1.0, seed=0,
        latent=0, cost=None, steps=None, dataset="gsm8k"):
    return AttemptLog(task_id=task, dataset=dataset,
                      condition="explicit_cot", seed=seed,
                      prompt_tokens=pt, completion_tokens=ct,
                      latency_s=lat, solved=solved, latent_steps=latent,
                      cost_usd=cost, steps=steps or [])


def test_usd_per_solve_counts_failures_in_numerator():
    # costs: solved att = 10*.01 + 5*.02 = 0.20 ; failed att = 0.20
    # total 0.40 over 1 solve -> 0.40 (the failure is NOT free)
    r = compute_metrics([att(solved=True), att(solved=False)],
                        price_card=CARD, model=MODEL)
    assert r.solves == 1 and r.attempts == 2
    assert r.usd_per_solve == pytest.approx(0.40)


def test_explicit_cost_overrides_price_card():
    r = compute_metrics([att(solved=True, cost=3.0),
                         att(solved=False, cost=1.0)],
                        price_card=CARD, model=MODEL)
    assert r.usd_per_solve == pytest.approx(4.0)


def test_gpu_time_costing_when_no_token_rates():
    card = PriceCard(usd_per_second=0.5, label="time-card")
    r = compute_metrics([att(solved=True, lat=2.0),
                         att(solved=False, lat=2.0)],
                        price_card=card, model=MODEL)
    assert r.usd_per_solve == pytest.approx(2.0)  # (2+2)*0.5 / 1


def test_latency_median_and_iqr():
    atts = [att(solved=True, lat=x) for x in (1.0, 2.0, 3.0, 4.0)]
    r = compute_metrics(atts, price_card=CARD, model=MODEL)
    assert r.latency_per_solve_median_s == pytest.approx(2.5)
    # inclusive quartiles of [1,2,3,4]: q1=1.75, q3=3.25 -> IQR 1.5
    assert r.latency_per_solve_iqr_s == pytest.approx(1.5)
    assert r.latency_total_per_solve_s == pytest.approx(10.0 / 4)


def test_flops_estimator_charges_latent_steps():
    # explicit: 2 * 1000 * (10 + 5 + 0) = 30_000
    a = att(solved=True)
    assert a.flops(MODEL) == 30_000
    # latent: same tokens + 7 latent steps = 2*1000*22 = 44_000
    b = att(solved=True, latent=7)
    assert b.flops(MODEL) == 44_000
    r = compute_metrics([a, b], price_card=CARD, model=MODEL)
    assert r.flops_per_solve == pytest.approx((30_000 + 44_000) / 2)


def test_tokens_per_solve_visible_and_total():
    r = compute_metrics([att(solved=True, pt=10, ct=5),
                         att(solved=False, pt=30, ct=15)],
                        price_card=CARD, model=MODEL)
    assert r.tokens_per_solve == pytest.approx(20.0)        # (5+15)/1
    assert r.total_tokens_per_solve == pytest.approx(60.0)  # (15+45)/1


def test_auditability_rubric():
    traced = StepLog(kind="final_answer", trace_text="#### 42")
    untraced = StepLog(kind="tool_call", trace_text=None)
    reasoning = StepLog(kind="reasoning", trace_text=None)  # not AB
    r = compute_metrics(
        [att(solved=True, steps=[traced, reasoning]),
         att(solved=False, steps=[untraced])],
        price_card=CARD, model=MODEL)
    # 2 action-bearing steps, 1 traced -> 0.5 ; reasoning ignored
    assert r.auditability == pytest.approx(0.5)


def test_auditability_none_without_action_bearing_steps():
    r = compute_metrics([att(solved=True,
                             steps=[StepLog(kind="reasoning",
                                            trace_text="x")])],
                        price_card=CARD, model=MODEL)
    assert r.auditability is None


def test_zero_solves_gives_infinite_per_solve_costs():
    r = compute_metrics([att(solved=False), att(solved=False)],
                        price_card=CARD, model=MODEL)
    assert r.solves == 0 and r.accuracy == 0.0
    assert math.isinf(r.usd_per_solve)
    assert math.isinf(r.flops_per_solve)


def test_empty_attempts_rejected():
    with pytest.raises(ValueError):
        compute_metrics([], price_card=CARD, model=MODEL)


def test_tokenecon_record_adapter():
    rec = {"task_id": "gsm8k-test-0", "prompt_tokens": 100,
           "completion_tokens": 50, "cost_usd": 0.25}
    a = attempt_from_tokenecon_record(rec, dataset="gsm8k",
                                      condition="explicit_cot", seed=3,
                                      latency_s=1.5, solved=True)
    assert a.prompt_tokens == 100 and a.completion_tokens == 50
    assert a.cost_usd == 0.25 and a.seed == 3
    r = compute_metrics([a], price_card=CARD, model=MODEL)
    assert r.usd_per_solve == pytest.approx(0.25)


def test_tokenecon_callrecord_object_adapter():
    """Interop with the real published tokenecon package."""
    tokenecon = pytest.importorskip("tokenecon")
    tier = tokenecon.ModelTier(
        name="test-tier", input_per_mtok=10.0, output_per_mtok=20.0,
        capability=0.5)
    rec = tokenecon.CallRecord(tier=tier, prompt_tokens=1_000_000,
                               completion_tokens=500_000)
    # hand-computed: 1e6 * 10/1e6 + 5e5 * 20/1e6 = 10 + 10 = 20 USD
    assert rec.cost() == pytest.approx(20.0)
    a = attempt_from_tokenecon_record(rec, dataset="gsm8k",
                                      condition="explicit_cot", seed=0,
                                      latency_s=1.0, solved=True)
    assert a.prompt_tokens == 1_000_000
    assert a.completion_tokens == 500_000
    assert a.cost_usd == pytest.approx(20.0)


def test_attempt_jsonl_round_trip():
    a = att(solved=True, steps=[StepLog(kind="final_answer",
                                        trace_text="#### 5")])
    b = AttemptLog.from_dict(a.to_dict())
    assert b == a
