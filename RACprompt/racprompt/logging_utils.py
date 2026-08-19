from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable


def setup_logging(is_main: bool, level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level if is_main else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


class RunLogger:
    def __init__(self, output_dir: str | Path, tensorboard: bool = True):
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.analysis_dir = self.output_dir / "analysis"
        self.plot_dir = self.output_dir / "plots"
        for directory in (
            self.log_dir,
            self.analysis_dir,
            self.plot_dir,
            self.output_dir / "eval",
            self.output_dir / "checkpoints",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "train.jsonl"
        self.csv_path = self.log_dir / "train_metrics.csv"
        self.critical_path = self.analysis_dir / "critical_states.csv"
        self._metric_fields: list[str] | None = None
        self._critical_fields: list[str] | None = None
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(self.log_dir / "tensorboard")
            except ImportError:
                logging.getLogger(__name__).warning(
                    "TensorBoard unavailable; continuing without it"
                )

    @staticmethod
    def _append_csv(
        path: Path, row: dict[str, Any], fields: list[str] | None
    ) -> list[str]:
        if fields is None:
            fields = list(row.keys())
        missing = [key for key in row if key not in fields]
        if missing:
            raise ValueError(f"CSV schema changed for {path}: new fields={missing}")
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in fields})
        return fields

    def log_step(self, metrics: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, allow_nan=True) + "\n")
        self._metric_fields = self._append_csv(
            self.csv_path, metrics, self._metric_fields
        )
        if self.writer is not None:
            step = int(metrics["step"])
            for key, value in metrics.items():
                if key != "step" and isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, step)

    def log_critical_states(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self._critical_fields = self._append_csv(
                self.critical_path, row, self._critical_fields
            )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
