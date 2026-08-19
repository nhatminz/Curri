import random

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from racprompt.checkpoint import capture_rng_state, load_training_state, save_checkpoint  # noqa: E402
from racprompt.config import RACConfig  # noqa: E402
from racprompt.curriculum import CurriculumState, PromptScore  # noqa: E402


class SaveableLinear(torch.nn.Linear):
    def save_pretrained(self, path, safe_serialization=True):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")


class DummyTokenizer:
    def save_pretrained(self, path):
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_checkpoint_restores_curriculum_optimizer_step_and_rng(tmp_path):
    model = SaveableLinear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    curriculum = CurriculumState(["a", "b"], seed=5)
    curriculum.update(PromptScore(0, "a", 0.8, 0.9, 0.7, 10, 4), step=7)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    rng = capture_rng_state()
    expected = (random.random(), np.random.random(), float(torch.rand(())))

    checkpoint = save_checkpoint(
        tmp_path,
        7,
        model,
        DummyTokenizer(),
        optimizer,
        scheduler,
        curriculum,
        [rng],
        {"schema": "test"},
        RACConfig(),
        keep_last_n=3,
    )
    curriculum.scores[:] = 0
    curriculum.usage_counts[:] = 0
    optimizer.param_groups[0]["lr"] = 99
    state = load_training_state(checkpoint, optimizer, scheduler, curriculum, rank=0)
    observed = (random.random(), np.random.random(), float(torch.rand(())))
    assert state["step"] == 7
    assert curriculum.usage_counts.tolist() == [1, 0]
    assert curriculum.scores[0] > 0
    assert optimizer.param_groups[0]["lr"] != 99
    assert observed == pytest.approx(expected)
