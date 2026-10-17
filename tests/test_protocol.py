"""Protocol tests for the corrected Week-1 harness protocol.

The Week-1 Kaggle ProsQA baseline (accuracy 0.144) was diagnosed as
a harness artifact: 75.7% of failures truncated at the 256-token
generation cap and 22.1% missed answer extraction. The 2026-10-17
protocol fix raised the cap to 768, made the prompt close with an
explicit ``#### <answer>`` final-line instruction, and repaired two
label bugs (run headings and the price-card label). These tests pin
that protocol so it cannot silently regress.
"""

import inspect

from latent_cost.data import Task
from latent_cost.harness import (USER_SUFFIX, build_messages,
                                 run_condition, run_kind_note)
from latent_cost.models import HFRunner, MockRunner


def test_generation_cap_default_is_768_everywhere():
    assert inspect.signature(
        run_condition).parameters["max_new_tokens"].default == 768
    assert inspect.signature(
        HFRunner.generate).parameters["max_new_tokens"].default == 768
    assert inspect.signature(
        MockRunner.generate).parameters["max_new_tokens"].default == 768


def test_prompts_close_with_final_line_instruction():
    for dataset in ("gsm8k", "prosqa"):
        task = Task(task_id="t", dataset=dataset, question="Q?", gold=0)
        user = build_messages(task)[-1]["content"]
        assert user.endswith(USER_SUFFIX[dataset].strip())
        assert "Finish with a final line in the form: ####" in user
    assert "#### <number>" in USER_SUFFIX["gsm8k"]
    # ProsQA expects the full statement, with the worked example.
    assert "#### Sally is a sterpus" in USER_SUFFIX["prosqa"]


def test_run_kind_note_labels():
    assert run_kind_note("SMOKE") == "(pipeline validation, not a result)"
    assert run_kind_note("smoke") == "(pipeline validation, not a result)"
    assert run_kind_note("WEEK1") == "(baseline run)"
