"""Five-metric costing for reasoning-mode comparisons, per solved task.

This module extends the published ``tokenecon`` package (Pandey, 2026;
PyPI ``tokenecon==0.2.0``) with the five-metric cost model defined in
Section 3 of the outline for *Tokens Are Not Dollars: Latent vs Explicit
Reasoning Inside LLM Agents, Measured in Cost per Solved Task*.

Every metric is defined **per solved task**, not per token and not per
attempt: a cheap run that fails is not cheap. One "attempt" is one
(task, seed, condition) trial; one "solve" is an attempt the verifier
marks correct. Failed attempts count in every numerator.

The five metrics
----------------
1. ``$/solve``
   Total inference cost over ALL attempts divided by the number of
   solves. Cost comes from ``AttemptLog.cost_usd`` when the runner
   recorded it; otherwise it is computed from token counts with a
   stated :class:`PriceCard` (input/output per-token rates), or, when
   the card carries only ``usd_per_second``, from wall-clock latency
   (the GPU-time costing the outline allows for local runs). The rate
   actually used is always reported alongside the number.

2. ``latency/solve`` (median + spread)
   Distribution of per-attempt wall-clock seconds (end to end,
   including any tool-call waits the runner folded into the attempt
   latency). Reported as the median with the inter-quartile range as
   the spread, per the outline's "report median and spread, not only
   the mean". ``latency_total_per_solve_s`` (total seconds / solves)
   is reported as a secondary figure.

3. ``FLOPs/solve``
   Estimated floating-point operations over all attempts divided by
   solves. Estimator (stated here so runs can be re-priced and
   re-estimated): each processed token position costs one forward
   pass, and one forward pass over one token position costs
   approximately ``2 * n_params`` FLOPs for a dense transformer
   (the standard Kaplan et al. approximation; the attention
   quadratic term and embedding lookups are omitted, which is
   conservative for the short sequences used here). Latent steps
   are charged exactly like token positions -- a latent step still
   runs a forward pass, which is the paper's central accounting
   point::

       flops(attempt) = 2 * n_params
                        * (prompt_tokens + completion_tokens
                           + latent_steps)

4. ``tokens/solve``
   Visible (generated) tokens over all attempts divided by solves.
   ``total_tokens_per_solve`` (prompt + generated) is reported as a
   secondary figure for comparability with agent-cost work that
   prices the whole trace.

5. ``auditability score``
   Share of action-bearing steps that carry a human-readable
   explicit trace, scored against this rubric: a step is
   *action-bearing* iff its ``kind`` is one of ``tool_call``,
   ``external_action``, ``money_step``, or ``final_answer`` (the
   step that commits the agent's answer to the outside world).
   A step *carries an explicit trace* iff ``trace_text`` is a
   non-empty human-readable string. The score is the ratio over
   all steps of all attempts (it is a proportion, so -- unlike the
   other four -- it is not divided by solves). ``None`` when no
   action-bearing steps were logged.

If a run set has zero solves, the per-solve quotients are ``inf``
and ``solves == 0`` is flagged on the result: a condition that
solves nothing has no finite cost per solve.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional

#: Step kinds that act on the world (or commit an answer to it) and
#: therefore fall under the auditability rubric.
ACTION_BEARING_KINDS = frozenset(
    {"tool_call", "external_action", "money_step", "final_answer"}
)


@dataclass
class StepLog:
    """One step inside an attempt.

    ``kind`` is a free-form label; membership in
    :data:`ACTION_BEARING_KINDS` decides whether the auditability
    rubric looks at the step. ``trace_text`` is the human-readable
    explicit trace for the step, or ``None`` when the step produced
    none (e.g. a latent step).
    """

    kind: str
    trace_text: Optional[str] = None

    @property
    def is_action_bearing(self) -> bool:
        return self.kind in ACTION_BEARING_KINDS

    @property
    def has_explicit_trace(self) -> bool:
        return bool(self.trace_text and self.trace_text.strip())


@dataclass
class PriceCard:
    """A stated price card. Exactly the rates used are reported.

    Rates are USD per token (not per million) for input/output, or
    USD per wall-clock second for local GPU/CPU-time costing.
    """

    input_usd_per_token: float = 0.0
    output_usd_per_token: float = 0.0
    usd_per_second: float = 0.0
    label: str = "unstated"

    def cost_of(self, prompt_tokens: int, completion_tokens: int,
                latency_s: float) -> float:
        token_cost = (prompt_tokens * self.input_usd_per_token
                      + completion_tokens * self.output_usd_per_token)
        if token_cost > 0.0:
            return token_cost
        return latency_s * self.usd_per_second


@dataclass
class ModelSpec:
    """The model a run used; ``n_params`` feeds the FLOPs estimator."""

    name: str
    n_params: int


@dataclass
class AttemptLog:
    """One (task, seed, condition) trial, as written to the JSONL log."""

    task_id: str
    dataset: str
    condition: str
    seed: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    solved: bool
    latent_steps: int = 0
    cost_usd: Optional[float] = None
    steps: list[StepLog] = field(default_factory=list)

    # -- costing -------------------------------------------------
    def resolved_cost_usd(self, price_card: PriceCard) -> float:
        if self.cost_usd is not None:
            return float(self.cost_usd)
        return price_card.cost_of(
            self.prompt_tokens, self.completion_tokens, self.latency_s)

    def flops(self, model: ModelSpec) -> float:
        """Estimated FLOPs for this attempt (see module docstring)."""
        positions = (self.prompt_tokens + self.completion_tokens
                     + self.latent_steps)
        return 2.0 * model.n_params * positions

    # -- serialisation -------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AttemptLog":
        steps = [StepLog(**s) for s in d.pop("steps", [])]
        return cls(steps=steps, **d)


@dataclass
class MetricsResult:
    """The five metrics for one (dataset, condition) cell."""

    dataset: str
    condition: str
    model: str
    attempts: int
    solves: int
    accuracy: float
    usd_per_solve: float
    latency_per_solve_median_s: float
    latency_per_solve_iqr_s: float
    latency_total_per_solve_s: float
    flops_per_solve: float
    tokens_per_solve: float
    total_tokens_per_solve: float
    auditability: Optional[float]
    price_card_label: str

    def to_dict(self) -> dict:
        return asdict(self)


def _per_solve(total: float, solves: int) -> float:
    return total / solves if solves > 0 else float("inf")


def compute_metrics(attempts: Iterable[AttemptLog], *,
                    price_card: PriceCard,
                    model: ModelSpec) -> MetricsResult:
    """Aggregate attempt logs into the five per-solved-task metrics.

    All attempts in ``attempts`` must belong to one
    (dataset, condition) cell; the cell identity is taken from the
    first attempt. Raises ``ValueError`` on an empty iterable.
    """
    attempts = list(attempts)
    if not attempts:
        raise ValueError("compute_metrics needs at least one attempt")

    solves = sum(1 for a in attempts if a.solved)
    total_cost = sum(a.resolved_cost_usd(price_card) for a in attempts)
    total_flops = sum(a.flops(model) for a in attempts)
    total_visible_tokens = sum(a.completion_tokens for a in attempts)
    total_tokens = sum(a.prompt_tokens + a.completion_tokens
                       for a in attempts)
    latencies = [a.latency_s for a in attempts]
    total_latency = sum(latencies)

    if len(latencies) >= 4:
        q1, _q2, q3 = statistics.quantiles(latencies, n=4, method="inclusive")
        iqr = q3 - q1
    elif len(latencies) >= 2:
        iqr = max(latencies) - min(latencies)
    else:
        iqr = 0.0

    action_steps = [s for a in attempts for s in a.steps
                    if s.is_action_bearing]
    if action_steps:
        auditability = (sum(1 for s in action_steps
                            if s.has_explicit_trace) / len(action_steps))
    else:
        auditability = None

    first = attempts[0]
    return MetricsResult(
        dataset=first.dataset,
        condition=first.condition,
        model=model.name,
        attempts=len(attempts),
        solves=solves,
        accuracy=solves / len(attempts),
        usd_per_solve=_per_solve(total_cost, solves),
        latency_per_solve_median_s=statistics.median(latencies),
        latency_per_solve_iqr_s=iqr,
        latency_total_per_solve_s=_per_solve(total_latency, solves),
        flops_per_solve=_per_solve(total_flops, solves),
        tokens_per_solve=_per_solve(float(total_visible_tokens), solves),
        total_tokens_per_solve=_per_solve(float(total_tokens), solves),
        auditability=auditability,
        price_card_label=price_card.label,
    )


# ---------------------------------------------------------------------
# tokenecon interop
# ---------------------------------------------------------------------
def attempt_from_tokenecon_record(record: dict, *,
                                  dataset: str,
                                  condition: str,
                                  seed: int,
                                  latency_s: float,
                                  solved: bool,
                                  steps: Optional[list[StepLog]] = None,
                                  ) -> AttemptLog:
    """Build an :class:`AttemptLog` from a tokenecon-style record.

    ``tokenecon`` (the published toolkit this costing layer extends)
    accounts a run in token streams -- prompt, completion, tool-call
    and retrieval tokens -- with a priced total. This adapter accepts
    either a ``tokenecon.CallRecord`` (or any object with the same
    attributes) or a plain ``dict`` in that shape (``prompt_tokens``
    / ``input_tokens``, ``completion_tokens`` / ``output_tokens``,
    and optionally ``cost_usd`` / ``total_cost`` / ``cost``), so
    logs priced by tokenecon can be re-aggregated under the
    five-metric model without re-pricing.
    """
    if not isinstance(record, dict):
        cost_attr = getattr(record, "cost", None)
        if callable(cost_attr):  # tokenecon.CallRecord.cost is a method
            cost_attr = cost_attr()
        record = {
            "task_id": getattr(record, "task_id", "unknown"),
            "prompt_tokens": getattr(record, "prompt_tokens", 0),
            "completion_tokens": getattr(record, "completion_tokens", 0),
            "cost_usd": cost_attr,
            "latent_steps": getattr(record, "latent_steps", 0),
        }
    prompt = record.get("prompt_tokens", record.get("input_tokens", 0))
    completion = record.get("completion_tokens",
                            record.get("output_tokens", 0))
    cost = record.get("cost_usd",
                      record.get("total_cost", record.get("cost")))
    return AttemptLog(
        task_id=str(record.get("task_id", record.get("id", "unknown"))),
        dataset=dataset,
        condition=condition,
        seed=seed,
        prompt_tokens=int(prompt),
        completion_tokens=int(completion),
        latency_s=float(latency_s),
        solved=bool(solved),
        latent_steps=int(record.get("latent_steps", 0)),
        cost_usd=None if cost is None else float(cost),
        steps=list(steps or []),
    )
