from __future__ import annotations

import argparse
from pathlib import Path

from racprompt.config import is_automatic_run_name, load_config, read_latest_run_name
from racprompt.plotting import generate_all_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate RAC-OPD plots from raw CSV artifacts"
    )
    parser.add_argument("--config", default="configs/rac_opd.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if is_automatic_run_name(config.training.run_name) and not args.output_dir:
        config.training.run_name = read_latest_run_name(config.paths.output_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config.paths.output_root) / config.training.run_name
    )
    generate_all_plots(output_dir)
    print(f"Plots written to {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
