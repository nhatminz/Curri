import numpy as np

from racprompt.curriculum import CurriculumState, PromptScore


def make_state() -> CurriculumState:
    return CurriculumState([f"p{i}" for i in range(100)], seed=7)


def test_probabilities_normalize_and_have_nonzero_exploration():
    state = make_state()
    state.scores[:] = np.linspace(0, 1, len(state.scores))
    probabilities = state.probabilities(step=10)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-12)
    assert np.all(probabilities > 0)
    assert probabilities.min() >= state.eps_explore / len(probabilities)


def test_sampling_is_with_replacement_and_exact_size():
    state = make_state()
    indices, _ = state.sample(step=0, global_batch_size=1000)
    assert len(indices) == 1000
    assert len(np.unique(indices)) < len(indices)


def test_mastered_prompt_memory_can_decrease_after_low_teachability():
    state = make_state()
    state.scores[3] = 0.95
    before = state.scores[3]
    state.update(PromptScore(3, "p3", 0.01, 0.01, 1.0, 20, 10), step=5)
    assert state.scores[3] < before
    assert state.usage_counts[3] == 1


def test_stale_score_ages_toward_prior():
    state = make_state()
    state.scores[0] = 1.0
    state.last_seen[0] = 0
    assert abs(state.effective_scores(10000)[0] - state.initial_score) < 1e-6
