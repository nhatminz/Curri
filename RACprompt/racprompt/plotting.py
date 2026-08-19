from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _prepare_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_evaluation(output_dir: str | Path) -> None:
    import pandas as pd

    root = Path(output_dir)
    path = root / "logs" / "eval_metrics.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    for benchmark in ("math500", "aime24", "aime25"):
        subset = frame[frame["benchmark"] == benchmark].sort_values("step")
        if not subset.empty:
            ax.plot(
                subset["step"],
                subset["mean_accuracy"],
                marker="o",
                label=benchmark.upper(),
            )
    ax.set(xlabel="Training step", ylabel="Mean accuracy over all samples", ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(root / "plots" / f"eval_curves.{suffix}", dpi=180)
    plt.close(fig)


def plot_loss(output_dir: str | Path) -> None:
    import pandas as pd

    root = Path(output_dir)
    path = root / "logs" / "train_metrics.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame["opd_loss_ema"] = frame["opd_loss"].ewm(alpha=0.1, adjust=False).mean()
    frame[["step", "opd_loss", "opd_loss_ema"]].to_csv(
        root / "analysis" / "loss_curve.csv", index=False
    )
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["step"], frame["opd_loss"], alpha=0.35, label="Raw OPD loss")
    ax.plot(frame["step"], frame["opd_loss_ema"], label="EMA (alpha=0.1)")
    ax.set(xlabel="Training step", ylabel="OPD loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(root / "plots" / f"loss_curve.{suffix}", dpi=180)
    plt.close(fig)


def save_and_plot_prompt_usage(
    output_dir: str | Path, curriculum: Any
) -> dict[str, Any]:
    import pandas as pd

    root = Path(output_dir)
    frame = pd.DataFrame(
        {
            "prompt_index": np.arange(len(curriculum.prompt_ids)),
            "prompt_id": curriculum.prompt_ids,
            "usage_count": curriculum.usage_counts,
            "memory_score": curriculum.scores,
            "last_seen_step": curriculum.last_seen,
            "latest_G": curriculum.latest_g,
            "latest_R": curriculum.latest_r,
            "latest_T": curriculum.latest_t,
            "ema_G": curriculum.ema_g,
            "ema_R": curriculum.ema_r,
            "ema_T": curriculum.ema_t,
        }
    )
    frame.to_csv(root / "analysis" / "prompt_usage.csv", index=False)
    counts = frame["usage_count"]
    never = int((counts == 0).sum())
    top = frame.nlargest(20, "usage_count")[
        ["prompt_id", "prompt_index", "usage_count"]
    ].to_dict("records")
    report = {
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "p90": float(counts.quantile(0.90)),
        "p95": float(counts.quantile(0.95)),
        "p99": float(counts.quantile(0.99)),
        "never_sampled": never,
        "never_sampled_percent": 100.0 * never / len(frame),
        "top_20": top,
    }
    with (root / "analysis" / "prompt_usage_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=min(50, max(10, int(counts.max()) + 1)))
    ax.set(xlabel="Times sampled", ylabel="Number of prompts")
    fig.tight_layout()
    fig.savefig(root / "plots" / "prompt_usage_hist.png", dpi=180)
    fig.savefig(root / "plots" / "prompt_usage_hist.pdf")
    plt.close(fig)
    return report


def plot_prompt_usage_csv(output_dir: str | Path) -> None:
    import pandas as pd

    root = Path(output_dir)
    path = root / "analysis" / "prompt_usage.csv"
    if not path.exists():
        return
    counts = pd.read_csv(path)["usage_count"]
    if counts.empty:
        return
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=min(50, max(10, int(counts.max()) + 1)))
    ax.set(xlabel="Times sampled", ylabel="Number of prompts")
    fig.tight_layout()
    fig.savefig(root / "plots" / "prompt_usage_hist.png", dpi=180)
    fig.savefig(root / "plots" / "prompt_usage_hist.pdf")
    plt.close(fig)


def plot_critical_states(output_dir: str | Path) -> None:
    import pandas as pd

    root = Path(output_dir)
    path = root / "analysis" / "critical_states.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frame["normalized_position"].dropna(), bins=40, range=(0, 1), density=True)
    ax.set(xlabel="Normalized response position", ylabel="Density", xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(root / "plots" / "critical_state_normalized_position.png", dpi=180)
    fig.savefig(root / "plots" / "critical_state_normalized_position.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frame["absolute_position"].dropna(), bins=50)
    ax.set(xlabel="Absolute response-token position", ylabel="Selected states")
    fig.tight_layout()
    fig.savefig(root / "plots" / "critical_state_absolute_position.png", dpi=180)
    fig.savefig(root / "plots" / "critical_state_absolute_position.pdf")
    plt.close(fig)

    exploded = frame.assign(
        selection_reason=frame["selection_reason"].str.split("|", regex=False)
    ).explode("selection_reason")
    reasons = [
        item
        for item in ("segment", "global_peak", "change_point", "fill")
        if item in set(exploded["selection_reason"])
    ]
    fig, axes = plt.subplots(
        max(1, len(reasons)), 1, figsize=(8, 2.5 * max(1, len(reasons))), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, reason in zip(axes, reasons):
        values = exploded.loc[
            exploded["selection_reason"] == reason, "normalized_position"
        ]
        ax.hist(values, bins=30, range=(0, 1))
        ax.set_ylabel(reason)
    axes[-1].set_xlabel("Normalized response position")
    fig.tight_layout()
    fig.savefig(root / "plots" / "critical_state_by_reason.png", dpi=180)
    fig.savefig(root / "plots" / "critical_state_by_reason.pdf")
    plt.close(fig)


def generate_all_plots(output_dir: str | Path) -> None:
    Path(output_dir, "plots").mkdir(parents=True, exist_ok=True)
    Path(output_dir, "analysis").mkdir(parents=True, exist_ok=True)
    plot_evaluation(output_dir)
    plot_loss(output_dir)
    plot_prompt_usage_csv(output_dir)
    plot_critical_states(output_dir)
