#!/usr/bin/env python3
"""Plot whole-tissue composition and immune metrics by tumor grade."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_RESULT_DIR = Path("manuscript_panels_results/whole_tissue_grade/combined")
DEFAULT_TABLE_DIR = DEFAULT_RESULT_DIR / "tables"
DEFAULT_FIGURE_DIR = DEFAULT_RESULT_DIR / "figures"

DEFAULT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIG_DIR = DEFAULT_RESULT_DIR / "mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


GROUP_COL = "condition"
GROUP_ORDER = ["Low-grade", "High-grade"]
IMMUNE_ORDER = [
    "T Cells",
    "CD4+ T Cells",
    "FOXP3+ T Cells",
    "Macrophages",
    "Tryp+",
    "CD20+",
    "CD66b+",
]
NONIMMUNE_LABEL = "All other cells"
STACKED_PALETTE = {
    "T Cells": "#1f77b4",
    "CD4+ T Cells": "#ff7f0e",
    "FOXP3+ T Cells": "#2ca02c",
    "Macrophages": "#d62728",
    "Tryp+": "#9467bd",
    "CD20+": "#8c564b",
    "CD66b+": "#e377c2",
    NONIMMUNE_LABEL: "#d9d9d9",
}
GROUP_PALETTE = {"Low-grade": "#3A78A2", "High-grade": "#B74E4E"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make separate no-title manuscript panels from whole-tissue grade tables."
    )
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--prefix", default="fig_whole_tissue_grade")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 1.9,
            "xtick.major.width": 1.9,
            "ytick.major.width": 1.9,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
        }
    )
    sns.set_theme(style="white", context="talk")


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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


def add_stacked_legend(fig: plt.Figure, font_size: float) -> None:
    labels = IMMUNE_ORDER + [NONIMMUNE_LABEL]
    handles = [
        Patch(facecolor=STACKED_PALETTE[label], edgecolor="black", linewidth=1.0, label=label)
        for label in labels
    ]
    fig.legend(
        handles=handles,
        labels=labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.19),
        ncol=2,
        prop={"size": font_size},
        handlelength=1.3,
        handletextpad=0.55,
        columnspacing=1.2,
    )


def annotate_two_group_test(ax: plt.Axes, test_row: pd.Series, data: pd.DataFrame, metric: str) -> None:
    whisker_tops = []
    for group in GROUP_ORDER:
        top = tukey_upper_whisker(data.loc[data[GROUP_COL].eq(group), metric])
        if np.isfinite(top):
            whisker_tops.append(top)
    if not whisker_tops:
        return
    ymax = max(max(whisker_tops), 0.0)
    span = max(ymax, max(whisker_tops) - min(whisker_tops), 1e-12)
    y = ymax + 0.10 * span
    tick = 0.025 * span
    ax.plot([0, 0, 1, 1], [y - tick, y, y, y - tick], color="#1f1f1f", linewidth=1.9, clip_on=False)
    ax.text(
        0.5,
        y + 0.02 * span,
        significance_label(test_row["p_value"]),
        ha="center",
        va="bottom",
        fontsize=36,
        color="#1f1f1f",
    )
    ax.set_ylim(bottom=0, top=y + 0.16 * span)


def plot_composition(summary: pd.DataFrame, immune_counts: pd.DataFrame, output_base: Path) -> None:
    pivot = (
        immune_counts.pivot(index=GROUP_COL, columns="cell_type_sync", values="fraction_of_all_cells")
        .reindex(GROUP_ORDER)
        .reindex(columns=IMMUNE_ORDER)
        .fillna(0.0)
    )
    nonimmune = summary.set_index(GROUP_COL).reindex(GROUP_ORDER)["nonimmune_fraction_of_all_cells"]

    fig, ax = plt.subplots(figsize=(8.8, 10.4))
    x = np.arange(len(GROUP_ORDER))
    bottom = np.zeros(len(GROUP_ORDER))
    for label in IMMUNE_ORDER:
        values = pivot[label].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.68,
            color=STACKED_PALETTE[label],
            edgecolor="white",
            linewidth=1.2,
            zorder=2,
        )
        bottom += values
    ax.bar(
        x,
        nonimmune.to_numpy(dtype=float),
        bottom=bottom,
        width=0.68,
        color=STACKED_PALETTE[NONIMMUNE_LABEL],
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of all cells", fontsize=42, labelpad=14)
    ax.tick_params(axis="both", labelsize=36)
    ax.grid(axis="y", color="#d8d8d8", linewidth=1.0)
    ax.set_ylim(0, 1.0)
    sns.despine(fig=fig)
    add_stacked_legend(fig, font_size=28)
    fig.tight_layout(rect=[0, 0.28, 1, 1])
    save_figure(fig, output_base)


def plot_group_metric(
    data: pd.DataFrame,
    tests: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_base: Path,
) -> None:
    plot_data = data.copy()
    fig, ax = plt.subplots(figsize=(8.8, 9.8))

    sns.boxplot(
        data=plot_data,
        x=GROUP_COL,
        y=metric,
        hue=GROUP_COL,
        order=GROUP_ORDER,
        hue_order=GROUP_ORDER,
        palette=GROUP_PALETTE,
        width=0.56,
        fliersize=0,
        linewidth=1.9,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=42, labelpad=14)
    ax.tick_params(axis="both", labelsize=36)
    ax.grid(axis="y", color="#d8d8d8", linewidth=1.0)
    if not tests.empty:
        annotate_two_group_test(ax, tests.iloc[0], plot_data, metric)
    else:
        ax.set_ylim(bottom=0)
    sns.despine(fig=fig)
    fig.tight_layout()
    save_figure(fig, output_base)


def main() -> None:
    args = parse_args()
    setup_style()

    table_dir = args.table_dir
    figure_dir = args.figure_dir
    summary = pd.read_csv(table_dir / "whole_tissue_grade_summary.csv")
    immune_counts = pd.read_csv(table_dir / "whole_tissue_grade_immune_counts.csv")
    region = pd.read_csv(table_dir / "whole_tissue_grade_region_stats.csv")
    density = pd.read_csv(table_dir / "whole_tissue_grade_density_proxy.csv")
    region_tests = pd.read_csv(table_dir / "whole_tissue_grade_region_tests.csv")
    density_tests = pd.read_csv(table_dir / "whole_tissue_grade_density_tests.csv")
    density = density.copy()
    density["immune_density_proxy_per_million_area"] = density["immune_density_proxy"] * 1_000_000.0

    composition_base = figure_dir / f"{args.prefix}_composition_separate_panel"
    region_base = figure_dir / f"{args.prefix}_region_immune_fraction_separate_panel"
    density_base = figure_dir / f"{args.prefix}_immune_density_separate_panel"

    plot_composition(summary, immune_counts, composition_base)
    plot_group_metric(
        region,
        region_tests,
        "immune_fraction",
        "Immune fraction",
        region_base,
    )
    plot_group_metric(
        density,
        density_tests,
        "immune_density_proxy_per_million_area",
        "Immune density",
        density_base,
    )

    print(composition_base.with_suffix(".png"))
    print(composition_base.with_suffix(".pdf"))
    print(region_base.with_suffix(".png"))
    print(region_base.with_suffix(".pdf"))
    print(density_base.with_suffix(".png"))
    print(density_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
