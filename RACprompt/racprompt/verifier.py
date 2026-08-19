from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    extracted: str | None
    normalized_prediction: str | None
    normalized_answer: str | None
    correct: bool
    failure: str | None


def extract_boxed(text: str) -> str | None:
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    brace = text.find("{", start)
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : index]
    return None


def extract_final_answer(text: str, benchmark: str | None = None) -> str | None:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    patterns = (
        r"(?i)(?:final answer|answer)\s*(?:is|:|=)\s*([^\n]+)",
        r"(?i)therefore[, ]+([^\n.]+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return str(matches[-1]).strip().rstrip(".$")
    if benchmark and "aime" in benchmark.lower():
        integers = re.findall(r"(?<![\d.])-?\d{1,6}(?![\d.])", text)
        return integers[-1] if integers else None
    return None


def _latex_fraction(value: str) -> str:
    pattern = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    while previous != value:
        previous = value
        value = pattern.sub(r"(\1)/(\2)", value)
    return value


def normalize_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    boxed = extract_boxed(text)
    if boxed is not None:
        text = boxed
    text = text.replace("$", "").replace("\\,", "").replace("\\!", "")
    text = text.replace("−", "-").replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = _latex_fraction(text)
    text = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".,;:")
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text or None


def _symbolically_equal(left: str, right: str) -> bool:
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transforms = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )
        left_expr = parse_expr(left, transformations=transforms, evaluate=True)
        right_expr = parse_expr(right, transformations=transforms, evaluate=True)
        return bool(simplify(left_expr - right_expr) == 0)
    except Exception:
        return False


def verify_answer(
    response: str, ground_truth: Any, benchmark: str | None = None
) -> VerificationResult:
    extracted = extract_final_answer(response, benchmark)
    prediction = normalize_answer(extracted)
    answer = normalize_answer(ground_truth)
    if prediction is None:
        return VerificationResult(
            extracted, prediction, answer, False, "prediction_parse_failure"
        )
    if answer is None:
        return VerificationResult(
            extracted, prediction, answer, False, "ground_truth_parse_failure"
        )
    if prediction == answer:
        return VerificationResult(extracted, prediction, answer, True, None)
    if benchmark and "aime" in benchmark.lower():
        try:
            correct = int(prediction) == int(answer)
            return VerificationResult(
                extracted, prediction, answer, correct, None if correct else "not_equal"
            )
        except ValueError:
            return VerificationResult(
                extracted, prediction, answer, False, "aime_non_integer"
            )
    correct = _symbolically_equal(prediction, answer)
    return VerificationResult(
        extracted, prediction, answer, correct, None if correct else "not_equivalent"
    )


def answer_from_record(record: Any) -> Any:
    value = record.answer
    if isinstance(value, dict):
        for key in ("answer", "value", "solution", "target"):
            if key in value:
                return value[key]
    return value
