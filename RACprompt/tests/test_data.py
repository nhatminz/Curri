from racprompt.data import detect_schema, render_prompt


class DummyTokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, **kwargs):
        return f"CHAT:{messages[0]['content']}"


def test_schema_detection_and_overrides():
    schema = detect_schema(["problem_id", "question", "solution"])
    assert schema.id_field == "problem_id"
    assert schema.prompt_field == "question"
    assert schema.answer_field == "solution"
    overridden = detect_schema(
        ["x", "y", "z"], prompt_field="x", answer_field="y", id_field="z"
    )
    assert overridden.prompt_field == "x"


def test_auto_template_uses_model_template_but_not_twice():
    tokenizer = DummyTokenizer()
    assert render_prompt(tokenizer, "2+2?") == "CHAT:2+2?"
    preformatted = "<|im_start|>user\n2+2?<|im_end|>"
    assert render_prompt(tokenizer, preformatted) == preformatted
