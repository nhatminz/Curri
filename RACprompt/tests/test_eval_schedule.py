from racprompt.evaluator import should_evaluate


def test_eval_schedule_includes_zero_interval_and_final():
    selected = [step for step in range(0, 124) if should_evaluate(step, 123, 50)]
    assert selected == [0, 50, 100, 123]


def test_resume_does_not_repeat_step_zero_evaluation():
    assert not should_evaluate(0, 100, 50, resumed=True)
