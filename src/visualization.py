from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _ensure_fig_ax(ax=None, figsize=(10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _set_log_scale_if_wealth_axis(ax: plt.Axes, axis: str = "x") -> None:
    try:
        if axis == "x":
            values = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches] if ax.patches else []
        else:
            values = [patch.get_height() for patch in ax.patches] if ax.patches else []
        if values and min(values) > 0:
            if axis == "x":
                ax.set_xscale("log")
            else:
                ax.set_yscale("log")
        else:
            if axis == "x":
                ax.set_xscale("symlog", linthresh=1.0)
            else:
                ax.set_yscale("symlog", linthresh=1.0)
    except Exception:
        pass


def _set_log_scale_for_values(ax: plt.Axes, values: np.ndarray, axis: str = "y") -> None:
    if axis == "x":
        if np.all(values > 0):
            ax.set_xscale("log")
        else:
            ax.set_xscale("symlog", linthresh=1.0)
    else:
        if np.all(values > 0):
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1.0)


def plot_terminal_wealth_histogram(
    terminal_wealths: Iterable[float],
    strategy_name: Optional[str] = None,
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of terminal wealth outcomes."""
    values = np.asarray(list(terminal_wealths), dtype=float)
    fig, ax = _ensure_fig_ax(ax, figsize)
    ax.hist(values, bins=bins, color="skyblue", edgecolor="black")
    title = "Terminal Wealth Distribution"
    if strategy_name:
        title += f" — {strategy_name}"
    ax.set_title(title)
    ax.set_xlabel("Terminal Wealth ($)")
    ax.set_ylabel("Number of Trials")
    ax.grid(True, linestyle="--", alpha=0.3)
    _set_log_scale_for_values(ax, values, axis="x")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_percentile_curve(
    terminal_wealths: Iterable[float],
    strategy_name: Optional[str] = None,
    percentiles: Optional[Iterable[float]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a percentile curve for terminal wealth outcomes."""
    values = np.asarray(list(terminal_wealths), dtype=float)
    if percentiles is None:
        percentiles = np.linspace(0, 100, 101)
    else:
        percentiles = np.asarray(list(percentiles), dtype=float)

    wealth_values = np.percentile(values, percentiles)
    fig, ax = _ensure_fig_ax(ax, figsize)
    ax.plot(percentiles, wealth_values, marker="o", linestyle="-", color="tab:blue")
    title = "Terminal Wealth Percentile Curve"
    if strategy_name:
        title += f" — {strategy_name}"
    ax.set_title(title)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Terminal Wealth ($)")
    ax.grid(True, linestyle="--", alpha=0.3)
    _set_log_scale_for_values(ax, wealth_values, axis="y")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_ruin_probability(
    strategy_wealths: dict[str, Iterable[float]],
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the probability of ruin for multiple strategies."""
    labels = []
    probabilities = []
    for label, wealths in strategy_wealths.items():
        values = np.asarray(list(wealths), dtype=float)
        ruined = np.sum(values <= 0.0)
        probability = float(ruined) / len(values) if len(values) > 0 else 0.0
        labels.append(label)
        probabilities.append(probability)

    fig, ax = _ensure_fig_ax(ax, figsize)
    ax.bar(labels, probabilities, color="tomato")
    ax.set_title("Probability of Ruin")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_summary_metrics(
    strategy_wealths: dict[str, Iterable[float]],
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot grouped summary metrics for multiple strategies."""
    metric_names = ["Mean", "Median", "5th Percentile", "95th Percentile"]
    strategy_labels = list(strategy_wealths.keys())
    metrics_by_strategy = []

    for wealths in strategy_wealths.values():
        values = np.asarray(list(wealths), dtype=float)
        if len(values) == 0:
            metrics_by_strategy.append([0.0, 0.0, 0.0, 0.0])
        else:
            metrics_by_strategy.append([
                float(np.mean(values)),
                float(np.percentile(values, 50)),
                float(np.percentile(values, 5)),
                float(np.percentile(values, 95)),
            ])

    fig, ax = _ensure_fig_ax(ax, figsize)
    x = np.arange(len(metric_names))
    width = 0.8 / max(1, len(strategy_labels))

    for index, (label, metrics) in enumerate(zip(strategy_labels, metrics_by_strategy)):
        ax.bar(
            x + index * width,
            metrics,
            width=width,
            label=label,
            alpha=0.85,
        )

    ax.set_title("Summary Wealth Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Terminal Wealth ($)")
    ax.set_xticks(x + width * (len(strategy_labels) - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.legend(title="Strategy")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Use log scale for wealth metrics if possible.
    all_values = np.concatenate([np.asarray(list(w), dtype=float) for w in strategy_wealths.values()])
    _set_log_scale_for_values(ax, np.asarray([v for v in np.concatenate(metrics_by_strategy) if not np.isnan(v)]), axis="y")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_simulation_paths(
    paths: List[List[float]],
    strategy_name: Optional[str] = None,
    num_paths: int = 25,
    starting_age: int = 25,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a selection of simulation path histories if available."""
    if not paths:
        raise ValueError("No simulation paths provided.")

    fig, ax = _ensure_fig_ax(ax, figsize)
    selected = paths[:min(len(paths), num_paths)]
    for path in selected:
        x_values = range(starting_age, starting_age + len(path))
        ax.plot(x_values, path, alpha=0.65)

    title = "Simulation Path Trajectories"
    if strategy_name:
        title += f" — {strategy_name}"
    ax.set_title(title)
    ax.set_xlabel("Age")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, linestyle="--", alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax

