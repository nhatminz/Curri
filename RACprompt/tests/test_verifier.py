from racprompt.verifier import extract_boxed, verify_answer


def test_nested_box_and_fraction_normalization():
    response = r"Work gives \boxed{\frac{1}{2}}."
    assert extract_boxed(response) == r"\frac{1}{2}"
    assert verify_answer(response, "1/2", "math500").correct


def test_symbolic_fraction_equivalence():
    assert verify_answer(r"Final answer: \boxed{2/4}", "1/2", "math500").correct


def test_aime_integer_format_and_parse_failure():
    assert verify_answer("Therefore the answer is 007.", "7", "aime24").correct
    failed = verify_answer("I could not finish.", "7", "aime24")
    assert not failed.correct
    assert failed.failure == "prediction_parse_failure"
