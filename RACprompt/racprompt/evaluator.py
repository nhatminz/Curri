from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

from .config import DataConfig, EvalConfig
from .data import load_records, render_prompt
from .verifier import answer_from_record, verify_answer


LOGGER = logging.getLogger(__name__)


class VLLMMathEvaluator:
    def __init__(self, model_path: str, config: EvalConfig):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError("vLLM is mandatory for RAC-OPD evaluation") from exc
        kwargs: dict[str, Any] = {
            "model": model_path,
            "dtype": "bfloat16",
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "trust_remote_code": False,
        }
        signature = inspect.signature(LLM)
        if config.data_parallel_size > 1:
            if "data_parallel_size" in signature.parameters:
                kwargs["data_parallel_size"] = config.data_parallel_size
            else:
                LOGGER.warning(
                    "Installed vLLM does not expose offline data_parallel_size; using TP=%d on one replica",
                    config.tensor_parallel_size,
                )
        if config.max_model_len is not None:
            kwargs["max_model_len"] = config.max_model_len
        self.llm = LLM(**kwargs)
        self.sampling_params = SamplingParams(
            n=config.num_samples_per_problem,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def evaluate_dataset(
        self,
        name: str,
        dataset_path: str,
        output_dir: str | Path,
        data_config: DataConfig,
        step: int,
    ) -> dict[str, Any]:
        records, schema = load_records(
            dataset_path,
            split=data_config.eval_split,
            prompt_field=data_config.prompt_field,
            answer_field=data_config.answer_field,
            id_field=data_config.id_field,
        )
        if schema.answer_field is None:
            raise ValueError(f"No ground-truth answer field detected for {name}")
        prompts = [
            render_prompt(
                self.tokenizer,
                item.prompt,
                data_config.prompt_template_mode,
                data_config.enable_thinking,
            )
            for item in records
        ]
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=True)
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        correct = 0
        failures = 0
        total = 0
        with (target / f"{name}_samples.jsonl").open("w", encoding="utf-8") as handle:
            for record, rendered_prompt, request_output in zip(
                records, prompts, outputs
            ):
                sample_rows = []
                for sample_index, sample in enumerate(request_output.outputs):
                    verification = verify_answer(
                        sample.text, answer_from_record(record), name
                    )
                    total += 1
                    correct += int(verification.correct)
                    failures += int(
                        verification.failure
                        in {"prediction_parse_failure", "ground_truth_parse_failure"}
                    )
                    sample_rows.append(
                        {
                            "sample_index": sample_index,
                            "response": sample.text,
                            "extracted_answer": verification.extracted,
                            "normalized_answer": verification.normalized_prediction,
                            "correct": verification.correct,
                            "failure": verification.failure,
                        }
                    )
                row = {
                    "step": step,
                    "benchmark": name,
                    "prompt_index": record.index,
                    "prompt_id": record.prompt_id,
                    "prompt": rendered_prompt,
                    "ground_truth": answer_from_record(record),
                    "samples": sample_rows,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "step": int(step),
            "benchmark": name,
            "problems": len(records),
            "samples": total,
            "mean_accuracy": correct / total if total else 0.0,
            "parse_failure_count": failures,
            "parse_failure_rate": failures / total if total else 0.0,
        }
        with (target / f"{name}_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary


def append_eval_metrics(
    output_dir: str | Path, summaries: list[dict[str, Any]]
) -> None:
    import csv

    path = Path(output_dir) / "logs" / "eval_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "benchmark",
        "problems",
        "samples",
        "mean_accuracy",
        "parse_failure_count",
        "parse_failure_rate",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(summaries)


def should_evaluate(
    step: int, max_steps: int, every_steps: int, resumed: bool = False
) -> bool:
    if step == 0:
        return not resumed
    return step == max_steps or (every_steps > 0 and step % every_steps == 0)
