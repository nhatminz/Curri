import numpy as np
import pytest

from racprompt.recoverability import (
    aggregate_prompt,
    geometric_mean_bridgeability,
    state_recoverability,
)


def test_recoverability_formula_ranges_and_values():
    state = state_recoverability(
        dplus=0.5,
        positive_mass_on_student_topk=0.25,
        alpha=np.array([0.2, 0.1]),
        next_compatibility=np.array([0.8, 0.2]),
    )
    assert state.valid
    assert 0 <= state.accessibility <= 1
    assert 0 <= state.future_compatibility <= 1
    assert 0 <= state.bridgeability <= 1
    prompt = aggregate_prompt([state])
    assert prompt.need == pytest.approx(0.5)
    assert prompt.recoverability == pytest.approx(state.bridgeability)
    assert prompt.teachability == pytest.approx(prompt.need * prompt.recoverability)


def test_no_positive_branch_marks_state_invalid():
    state = state_recoverability(0.4, 0.0, np.array([]), np.array([]))
    assert not state.valid
    assert np.isnan(state.future_compatibility)
    assert aggregate_prompt([state]).valid_states == 0


def test_geometric_mean_is_bottleneck_sensitive_and_safe():
    value = geometric_mean_bridgeability([1.0, 1e-30])
    assert np.isfinite(value)
    assert value == pytest.approx(1e-3)
    assert value < np.mean([1.0, 1e-30])


def test_branch_probe_is_no_grad_and_does_not_mutate_inputs():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from racprompt.scoring import DiagnosticScorer

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(11, 5)
            self.head = torch.nn.Linear(5, 11)

        def forward(
            self, input_ids, attention_mask=None, use_cache=False, past_key_values=None
        ):
            logits = self.head(self.embedding(input_ids))
            batch, length = input_ids.shape
            cache = (
                (torch.zeros(batch, 1, length, 1), torch.zeros(batch, 1, length, 1)),
            )
            return SimpleNamespace(logits=logits, past_key_values=cache)

    student, teacher = TinyLM(), TinyLM()
    scorer = DiagnosticScorer(student, teacher, torch.device("cpu"), stats_top_k=4)
    prefix = (1, 2, 3)
    candidates = [4, 5]
    values, used_cache = scorer.probe_next_compatibility(prefix, candidates)
    assert values.shape == (2,)
    assert used_cache
    assert prefix == (1, 2, 3) and candidates == [4, 5]
    assert all(parameter.grad is None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
