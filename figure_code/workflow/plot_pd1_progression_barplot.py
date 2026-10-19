#!/usr/bin/env python3
"""Plot PD1- or GZMB-positive T-cell fractions across axons and radii."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path("significance_axons_immune_cells/paper_ready_tcell_marker_composition")
OUTDIR = ROOT / "progression"
REGION_TABLE = ROOT / "marker_positive_fraction_region_level.csv"
STATS_TABLE = OUTDIR / "progression_marker_positive_fraction_stats.csv"

MPLCONFIG_DIR = ROOT / "mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


GROUP_ORDER = ["NonProgressor", "Progressor"]
GROUP_LABELS = {"NonProgressor": "Non-progressor", "Progressor": "Progressor"}
PALETTE = {"NonProgressor": "#4C78A8", "Progressor": "#F58518"}
AXON_ORDER = ["CXCR4+ / CD44+ / FYN+ Axon", "NRXN1+ / MAOA+ Axon"]
AXON_LABELS = {
    "CXCR4+ / CD44+ / FYN+ Axon": "CXCR4+\nCD44+ FYN+\naxon",
    "NRXN1+ / MAOA+ Axon": "NRXN1+\nMAOA+\naxon",
}
RADIUS_ORDER = [20, 50, 100]
FIGSIZE = (31.0, 13.0)
XTICK_SIZE = 50
YTICK_SIZE = 56
XLABEL_SIZE = 60
YLABEL_SIZE = 68
LEGEND_SIZE = 58
ANNOTATION_SIZE = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot one progression marker-composition barplot with manuscript-ready styling."
    )
    parser.add_argument("--marker", default="PD1", choices=["AR", "GZMB", "Ki67", "PD1"])
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 2.2,
            "xtick.major.width": 2.2,
            "ytick.major.width": 2.2,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
        }
    )
    sns.set_theme(style="whitegrid", context="talk")


def q_stars(q_value: object) -> str:
    q = pd.to_numeric(pd.Series([q_value]), errors="coerce").iloc[0]
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    marker = args.marker
    output_base = OUTDIR / f"{marker}_positive_fraction_barplot"
    setup_style()
    region = pd.read_csv(REGION_TABLE)
    stats = pd.read_csv(STATS_TABLE)

    data = region.loc[
        region["marker"].eq(marker)
        & region["progression"].isin(GROUP_ORDER)
        & region["axon"].isin(AXON_ORDER)
        & region["radius_um"].isin(RADIUS_ORDER)
    ].copy()
    tests = stats.loc[
        stats["marker"].eq(marker)
        & stats["axon"].isin(AXON_ORDER)
        & stats["radius_um"].isin(RADIUS_ORDER)
    ].copy()

    fig, axes = plt.subplots(1, len(RADIUS_ORDER), figsize=FIGSIZE, sharey=True)
    bar_width = 0.34
    offsets = {"NonProgressor": -0.19, "Progressor": 0.19}
    global_annotation_top = 0.0

    for ax, radius in zip(axes, RADIUS_ORDER):
        radius_data = data.loc[data["radius_um"].eq(radius)].copy()
        for axon_index, axon in enumerate(AXON_ORDER):
            axon_data = radius_data.loc[radius_data["axon"].eq(axon)].copy()
            errorbar_tops = []
            for group in GROUP_ORDER:
                vals = (
                    axon_data.loc[axon_data["progression"].eq(group), "pct_marker_positive_near"]
                    .dropna()
                    .sort_values()
                    .to_numpy(dtype=float)
                )
                if vals.size == 0:
                    continue
                x = axon_index + offsets[group]
                median = float(np.median(vals))
                q1, q3 = np.percentile(vals, [25, 75])
                errorbar_tops.append(float(q3))
                ax.bar(
                    x,
                    median,
                    width=bar_width,
                    color=PALETTE[group],
                    edgecolor=PALETTE[group],
                    linewidth=4.6,
                    zorder=2,
                )
                ax.errorbar(
                    x,
                    median,
                    yerr=np.array([[median - q1], [q3 - median]]),
                    fmt="none",
                    ecolor="#333333",
                    elinewidth=4.2,
                    capsize=12,
                    capthick=4.2,
                    zorder=4,
                )
            test_row = tests.loc[tests["radius_um"].eq(radius) & tests["axon"].eq(axon)]
            if not test_row.empty:
                x1 = axon_index + offsets["NonProgressor"]
                x2 = axon_index + offsets["Progressor"]
                local_top = max(errorbar_tops) if errorbar_tops else 0.0
                y = local_top + 7.0
                tick = 2.5
                ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color="#333333", linewidth=4.0)
                ax.text(
                    (x1 + x2) / 2.0,
                    y + 1.3,
                    q_stars(test_row.iloc[0]["q_two_sided_within_marker"]),
                    ha="center",
                    va="bottom",
                    fontsize=ANNOTATION_SIZE,
                    color="#333333",
                )
                global_annotation_top = max(global_annotation_top, y + 9.5)

        ax.set_xlim(-0.55, len(AXON_ORDER) - 0.45)
        ax.set_xticks(np.arange(len(AXON_ORDER)))
        ax.set_xticklabels([AXON_LABELS[axon] for axon in AXON_ORDER], fontsize=XTICK_SIZE)
        ax.set_xlabel(f"{radius} um radius", fontsize=XLABEL_SIZE, labelpad=34)
        ax.set_title("")
        ax.grid(axis="y", color="#d8d8d8", linewidth=1.8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=YTICK_SIZE, width=3.2, length=12)
        ax.tick_params(axis="x", length=0, pad=26)
        sns.despine(ax=ax)

    y_top = min(112.0, max(35.0, global_annotation_top))
    for ax in axes:
        ax.set_ylim(0, y_top)
    axes[0].set_ylabel(f"{marker}+ fraction (%)", fontsize=YLABEL_SIZE, labelpad=26)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=24,
            markerfacecolor=PALETTE[group],
            markeredgecolor="#333333",
            label=GROUP_LABELS[group],
        )
        for group in GROUP_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        prop={"size": LEGEND_SIZE},
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.35, top=0.80, wspace=0.24)
    save_all(fig, output_base)
    print(output_base.with_suffix(".pdf"))
    print(output_base.with_suffix(".png"))
    print(output_base.with_suffix(".svg"))


if __name__ == "__main__":
    main()
