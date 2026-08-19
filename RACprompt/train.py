from __future__ import annotations

import argparse
from racprompt.config import load_config
from racprompt.distributed import cleanup_distributed, init_distributed
from racprompt.logging_utils import setup_logging
from racprompt.trainer import RACOPDTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RAC-OPD")
    parser.add_argument("--config", default="configs/rac_opd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    resume = args.resume_from_checkpoint or args.resume
    if resume:
        overrides.append(f"checkpoint.resume={resume}")
    if args.dry_run:
        overrides.append("training.dry_run=true")
    config = load_config(args.config, overrides)
    context = init_distributed()
    setup_logging(context.is_main)
    try:
        trainer = RACOPDTrainer(config, context)
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
