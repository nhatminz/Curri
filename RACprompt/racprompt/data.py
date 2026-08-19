from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger(__name__)

PROMPT_CANDIDATES = (
    "prompt",
    "problem",
    "question",
    "query",
    "input",
    "instruction",
    "messages",
)
ANSWER_CANDIDATES = (
    "answer",
    "solution",
    "ground_truth",
    "target",
    "label",
    "response",
)
ID_CANDIDATES = ("id", "problem_id", "question_id", "uid", "uuid")


@dataclass(frozen=True)
class PromptRecord:
    index: int
    prompt_id: str
    prompt: Any
    answer: Any = None
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DetectedSchema:
    prompt_field: str
    answer_field: str | None
    id_field: str | None
    columns: tuple[str, ...]


def _choose_field(
    columns: Sequence[str],
    override: str | None,
    candidates: Sequence[str],
    required: bool,
) -> str | None:
    if override:
        if override not in columns:
            raise KeyError(
                f"Configured field {override!r} is absent; columns={list(columns)}"
            )
        return override
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    if required:
        raise KeyError(f"Could not detect required field from columns={list(columns)}")
    return None


def detect_schema(
    columns: Sequence[str],
    prompt_field: str | None = None,
    answer_field: str | None = None,
    id_field: str | None = None,
) -> DetectedSchema:
    columns = tuple(str(column) for column in columns)
    return DetectedSchema(
        prompt_field=str(_choose_field(columns, prompt_field, PROMPT_CANDIDATES, True)),
        answer_field=_choose_field(columns, answer_field, ANSWER_CANDIDATES, False),
        id_field=_choose_field(columns, id_field, ID_CANDIDATES, False),
        columns=columns,
    )


def _load_json_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "train", "test", "validation"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_with_datasets(path: Path, split: str | None) -> Any:
    try:
        from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "Install the 'datasets' package to load this data format"
        ) from exc

    if path.is_dir():
        try:
            loaded = load_from_disk(str(path))
        except (FileNotFoundError, ValueError, TypeError):
            loaded = None
        if loaded is None:
            parquet = sorted(path.rglob("*.parquet"))
            arrow = sorted(path.rglob("*.arrow"))
            json_files = sorted(path.rglob("*.json")) + sorted(path.rglob("*.jsonl"))
            if parquet:
                loaded = load_dataset(
                    "parquet", data_files=[str(item) for item in parquet]
                )
            elif json_files:
                loaded = load_dataset(
                    "json", data_files=[str(item) for item in json_files]
                )
            elif arrow:
                raise ValueError(
                    f"Found raw Arrow files under {path}, but no load_from_disk metadata. "
                    "Export them as Parquet/JSON or provide a Hugging Face save_to_disk directory."
                )
            else:
                raise FileNotFoundError(
                    f"No supported dataset files found under {path}"
                )
    elif path.suffix.lower() in (".json", ".jsonl"):
        loaded = Dataset.from_list(_load_json_file(path))
    elif path.suffix.lower() == ".parquet":
        loaded = load_dataset("parquet", data_files=str(path))
    elif path.suffix.lower() == ".arrow":
        raise ValueError("A standalone .arrow file needs Hugging Face dataset metadata")
    else:
        raise ValueError(f"Unsupported dataset path: {path}")

    if (
        isinstance(loaded, DatasetDict)
        or hasattr(loaded, "keys")
        and not hasattr(loaded, "column_names")
    ):
        keys = list(loaded.keys())
        chosen = split if split in keys else ("train" if "train" in keys else keys[0])
        LOGGER.info("Dataset splits=%s; selected split=%s", keys, chosen)
        return loaded[chosen]
    return loaded


def load_records(
    path: str | Path,
    split: str | None = None,
    prompt_field: str | None = None,
    answer_field: str | None = None,
    id_field: str | None = None,
) -> tuple[list[PromptRecord], DetectedSchema]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source}")
    dataset = _load_with_datasets(source, split)
    columns = (
        tuple(dataset.column_names)
        if hasattr(dataset, "column_names")
        else tuple(dataset[0].keys())
    )
    schema = detect_schema(columns, prompt_field, answer_field, id_field)
    LOGGER.info(
        "Detected dataset schema: columns=%s prompt=%s answer=%s id=%s rows=%d",
        list(schema.columns),
        schema.prompt_field,
        schema.answer_field,
        schema.id_field,
        len(dataset),
    )
    records: list[PromptRecord] = []
    seen_ids: set[str] = set()
    for index in range(len(dataset)):
        row = dict(dataset[index])
        prompt_id = str(row[schema.id_field]) if schema.id_field else str(index)
        if prompt_id in seen_ids:
            prompt_id = f"{prompt_id}#{index}"
        seen_ids.add(prompt_id)
        records.append(
            PromptRecord(
                index=index,
                prompt_id=prompt_id,
                prompt=row[schema.prompt_field],
                answer=row.get(schema.answer_field) if schema.answer_field else None,
                raw=row,
            )
        )
    if not records:
        raise ValueError(f"Dataset is empty: {source}")
    return records, schema


def _as_messages(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list) and all(isinstance(item, Mapping) for item in prompt):
        messages = []
        for item in prompt:
            role = str(item.get("role", "user"))
            content = item.get("content", item.get("text", ""))
            messages.append({"role": role, "content": str(content)})
        return messages
    if isinstance(prompt, Mapping) and "messages" in prompt:
        return _as_messages(prompt["messages"])
    return [{"role": "user", "content": str(prompt)}]


def render_prompt(
    tokenizer: Any,
    prompt: Any,
    template_mode: str = "auto",
    enable_thinking: bool | None = None,
) -> str:
    if template_mode not in {"auto", "chat", "raw"}:
        raise ValueError(f"Unknown prompt_template_mode={template_mode!r}")
    prompt_is_messages = (
        isinstance(prompt, list) or isinstance(prompt, Mapping) and "messages" in prompt
    )
    looks_preformatted = isinstance(prompt, str) and any(
        marker in prompt for marker in ("<|im_start|>", "<|user|>", "<|assistant|>")
    )
    use_chat = template_mode == "chat" or (
        template_mode == "auto"
        and not looks_preformatted
        and (prompt_is_messages or bool(getattr(tokenizer, "chat_template", None)))
    )
    if not use_chat:
        return str(prompt)
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(_as_messages(prompt), **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(_as_messages(prompt), **kwargs)


def tokenize_prompt(
    tokenizer: Any,
    rendered_prompt: str,
    max_prompt_tokens: int,
) -> list[int]:
    previous_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
        )["input_ids"]
    finally:
        tokenizer.truncation_side = previous_side
    if not encoded:
        fallback = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        )
        if fallback is None:
            raise ValueError(
                "Tokenizer produced an empty prompt and has no BOS/EOS fallback"
            )
        encoded = [int(fallback)]
    return [int(token) for token in encoded]
