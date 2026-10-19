#!/usr/bin/env python3
"""Plot whole-tissue T-cell marker panels by progression status."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_RESULT_DIR = Path("manuscript_panels_results/whole_tissue_tcell_marker_progression_updated")
DEFAULT_FIGURE_DIR = DEFAULT_RESULT_DIR / "figures"
DEFAULT_TABLE_DIR = DEFAULT_RESULT_DIR / "tables"

DEFAULT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_RESULT_DIR / "mplconfig"))
(DEFAULT_RESULT_DIR / "mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


GROUP_COL = "progression"
GROUP_ORDER = ["NonProgressor", "Progressor"]
GROUP_LABELS = {"NonProgressor": "Non-progressor", "Progressor": "Progressor"}
MARKER_ORDER = ["AR", "Ki67", "GZMB", "PD1"]
PALETTE = {"NonProgressor": "#4C78A8", "Progressor": "#F58518"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make separate no-title manuscript panels from whole-tissue T-cell marker progression tables."
    )
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--prefix", default="fig_whole_tissue_tcell_marker_progression")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 1.8,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
        }
    )
    sns.set_theme(style="white", context="talk")


def ordered_markers(df: pd.DataFrame) -> list[str]:
    present = set(df["marker"].dropna().astype(str))
    return [marker for marker in MARKER_ORDER if marker in present]


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_bottom_legend(fig: plt.Figure, font_size: float) -> None:
    handles = [
        Patch(
            facecolor=PALETTE[group],
            edgecolor="black",
            linewidth=1.8,
            label=GROUP_LABELS.get(group, group),
        )
        for group in GROUP_ORDER
    ]
    fig.legend(
        handles=handles,
        labels=[GROUP_LABELS.get(group, group) for group in GROUP_ORDER],
        frameon=False,
        loc="upper center",
        ncol=len(GROUP_ORDER),
        bbox_to_anchor=(0.5, 0.16),
        prop={"size": font_size},
        handlelength=1.8,
    )


def significance_label(p_value: object) -> str:
    p = pd.to_numeric(pd.Series([p_value]), errors="coerce").iloc[0]
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def tukey_upper_whisker(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.nan
    q1, q3 = np.percentile(clean, [25, 75])
    upper_fence = q3 + 1.5 * (q3 - q1)
    inlier = clean[clean <= upper_fence]
    return float(inlier.max()) if inlier.size else float(q3)


def marker_upper_whisker(data: pd.DataFrame, marker: str, metric: str) -> float:
    tops = []
    marker_data = data.loc[data["marker"].astype(str).eq(marker)]
    for group in GROUP_ORDER:
        top = tukey_upper_whisker(marker_data.loc[marker_data[GROUP_COL].eq(group), metric])
        if np.isfinite(top):
            tops.append(top)
    return max(tops) if tops else np.nan


def annotate_significance(ax: plt.Axes, tests: pd.DataFrame, markers: list[str], data: pd.DataFrame, metric: str) -> None:
    if tests.empty:
        return
    whisker_tops = [marker_upper_whisker(data, marker, metric) for marker in markers]
    whisker_tops = [value for value in whisker_tops if np.isfinite(value)]
    ymax = max(whisker_tops) if whisker_tops else float(pd.to_numeric(data[metric], errors="coerce").max())
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    y_step = 0.07 * ymax
    annotation_top = ymax
    for i, marker in enumerate(markers):
        row = tests.loc[tests["marker"].astype(str).eq(marker)]
        if row.empty:
            continue
        marker_top = marker_upper_whisker(data, marker, metric)
        if not np.isfinite(marker_top):
            continue
        y = marker_top + y_step
        label = significance_label(row.iloc[0]["p_value"])
        ax.text(i, y, label, ha="center", va="bottom", fontsize=36)
        annotation_top = max(annotation_top, y)
    ax.set_ylim(bottom=0, top=annotation_top + 2.0 * y_step)


def plot_whole_sample_fraction(summary: pd.DataFrame, output_base: Path) -> None:
    markers = ordered_markers(summary)
    fig, ax = plt.subplots(figsize=(10.2, 9.6))
    bar_width = 0.34
    offsets = np.array([-bar_width / 1.8, bar_width / 1.8])
    x = np.arange(len(markers))

    for offset, group in zip(offsets, GROUP_ORDER):
        values = (
            summary.loc[summary[GROUP_COL].eq(group)]
            .set_index("marker")
            .reindex(markers)["positive_fraction"]
            .to_numpy(dtype=float)
        )
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=PALETTE[group],
            edgecolor="black",
            linewidth=1.8,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(markers)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction positive", fontsize=42, labelpad=14)
    ax.set_title("")
    ax.tick_params(axis="both", labelsize=38)
    ax.grid(axis="y", color="#d8d8d8", linewidth=1.0)
    ax.set_ylim(bottom=0)
    sns.despine(fig=fig)
    add_bottom_legend(fig, font_size=38)
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    save_figure(fig, output_base)


def plot_region_metric(
    region: pd.DataFrame,
    tests: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_base: Path,
) -> None:
    data = region.copy()
    markers = ordered_markers(data)
    fig, ax = plt.subplots(figsize=(10.2, 9.8))

    sns.boxplot(
        data=data,
        x="marker",
        y=metric,
        hue=GROUP_COL,
        order=markers,
        hue_order=GROUP_ORDER,
        orient="v",
        palette=PALETTE,
        width=0.68,
        fliersize=0,
        linewidth=1.8,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=42, labelpad=14)
    ax.set_title("")
    ax.tick_params(axis="both", labelsize=38)
    ax.grid(axis="y", color="#d8d8d8", linewidth=1.0)
    ax.set_ylim(bottom=0)
    annotate_significance(ax, tests, markers, data, metric)
    sns.despine(fig=fig)
    add_bottom_legend(fig, font_size=38)
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    save_figure(fig, output_base)


def main() -> None:
    args = parse_args()
    setup_style()

    table_dir = args.table_dir
    figure_dir = args.figure_dir
    summary = pd.read_csv(table_dir / "whole_tissue_tcell_marker_progression_summary.csv")
    region = pd.read_csv(table_dir / "whole_tissue_tcell_marker_progression_region_stats.csv")
    fraction_tests = pd.read_csv(table_dir / "whole_tissue_tcell_marker_progression_region_fraction_tests.csv")
    density_tests = pd.read_csv(table_dir / "whole_tissue_tcell_marker_progression_density_tests.csv")
    region["marker_positive_density_x1e6"] = region["marker_positive_density_proxy"] * 1_000_000.0

    plot_whole_sample_fraction(
        summary,
        figure_dir / f"{args.prefix}_whole_sample_fraction_separate_panel",
    )
    plot_region_metric(
        region,
        fraction_tests,
        "positive_fraction",
        "Fraction positive",
        figure_dir / f"{args.prefix}_region_fraction_separate_panel",
    )
    plot_region_metric(
        region,
        density_tests,
        "marker_positive_density_x1e6",
        "Positive T-cell density",
        figure_dir / f"{args.prefix}_positive_tcell_density_separate_panel",
    )

    print(figure_dir / f"{args.prefix}_whole_sample_fraction_separate_panel.png")
    print(figure_dir / f"{args.prefix}_whole_sample_fraction_separate_panel.pdf")
    print(figure_dir / f"{args.prefix}_region_fraction_separate_panel.png")
    print(figure_dir / f"{args.prefix}_region_fraction_separate_panel.pdf")
    print(figure_dir / f"{args.prefix}_positive_tcell_density_separate_panel.png")
    print(figure_dir / f"{args.prefix}_positive_tcell_density_separate_panel.pdf")


if __name__ == "__main__":
    main()
