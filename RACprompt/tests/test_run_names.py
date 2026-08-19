import re

import pytest

from racprompt.config import (
    is_automatic_run_name,
    make_timestamped_run_name,
    read_latest_run_name,
    run_name_from_checkpoint,
)


def test_timestamped_run_name_format():
    name = make_timestamped_run_name()
    assert re.fullmatch(r"rac_opd_qwen3_\d{8}_\d{6}", name)
    assert is_automatic_run_name("auto")
    assert is_automatic_run_name("timestamp")
    assert not is_automatic_run_name(name)


def test_latest_run_marker(tmp_path):
    (tmp_path / "latest_run.txt").write_text(
        "rac_opd_qwen3_20260819_120000\n", encoding="utf-8"
    )
    assert read_latest_run_name(tmp_path) == "rac_opd_qwen3_20260819_120000"


def test_checkpoint_path_can_restore_original_run_name(tmp_path):
    checkpoint = (
        tmp_path / "rac_opd_qwen3_20260819_120000" / "checkpoints" / "step_000050"
    )
    assert run_name_from_checkpoint(checkpoint) == "rac_opd_qwen3_20260819_120000"
    with pytest.raises(ValueError):
        run_name_from_checkpoint(tmp_path / "step_000050")
