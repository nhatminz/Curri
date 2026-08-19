from __future__ import annotations

import argparse
import logging
from pathlib import Path

from racprompt.checkpoint import checkpoint_step, resolve_resume_path
from racprompt.config import load_config
from racprompt.evaluator import VLLMMathEvaluator, append_eval_metrics
from racprompt.logging_utils import setup_logging
from racprompt.plotting import plot_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full vLLM evaluation on MATH-500/AIME 2024/AIME 2025"
    )
    parser.add_argument("--config", default="configs/rac_opd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--model", default=None, help="Model directory; overrides --checkpoint"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint directory or 'latest'"
    )
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(True)
    config = load_config(args.config, args.set)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config.paths.output_root) / config.training.run_name
    )
    if args.model:
        model_path = args.model
        step = 0 if args.step is None else args.step
    elif args.checkpoint:
        checkpoint = resolve_resume_path(args.checkpoint, output_dir / "checkpoints")
        assert checkpoint is not None
        model_path = str(checkpoint / "student")
        step = checkpoint_step(checkpoint) if args.step is None else args.step
    else:
        model_path = config.paths.student_model
        step = 0 if args.step is None else args.step
    evaluator = VLLMMathEvaluator(model_path, config.evaluation)
    summaries = []
    step_dir = output_dir / "eval" / f"step_{step:06d}"
    for name, path in (
        ("math500", config.paths.math500),
        ("aime24", config.paths.aime24),
        ("aime25", config.paths.aime25),
    ):
        summaries.append(
            evaluator.evaluate_dataset(name, path, step_dir, config.data, step)
        )
    append_eval_metrics(output_dir, summaries)
    plot_evaluation(output_dir)
    logging.info("Evaluation summaries: %s", summaries)


if __name__ == "__main__":
    main()
