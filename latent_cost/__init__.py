"""latent_cost: five-metric costing extension for tokenecon.

Implements the per-solved-task cost model (Section 3 of the outline
for *Tokens Are Not Dollars*) on top of the published tokenecon
package: $/solve, latency/solve (median + spread), FLOPs/solve,
tokens/solve, and an auditability score.
"""

from .metrics import (
    ACTION_BEARING_KINDS,
    AttemptLog,
    MetricsResult,
    ModelSpec,
    PriceCard,
    StepLog,
    attempt_from_tokenecon_record,
    compute_metrics,
)

__all__ = [
    "ACTION_BEARING_KINDS",
    "AttemptLog",
    "MetricsResult",
    "ModelSpec",
    "PriceCard",
    "StepLog",
    "attempt_from_tokenecon_record",
    "compute_metrics",
]

__version__ = "0.1.0"
