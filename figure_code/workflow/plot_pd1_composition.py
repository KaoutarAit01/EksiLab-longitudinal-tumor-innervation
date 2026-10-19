#!/usr/bin/env python3
"""Plot PD1-positive and PD1-negative T-cell composition around axons."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path("significance_axons_immune_cells/paper_ready_tcell_marker_composition")
OUTDIR = ROOT / "progression"
REGION_TABLE = ROOT / "marker_positive_fraction_region_level.csv"
OUTPUT_BASE = OUTDIR / "PD1_raw_composition_donuts"

MPLCONFIG_DIR = ROOT / "mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MARKER = "PD1"
RADIUS_ORDER = [20, 50, 100]
ROW_ORDER = [
    ("CXCR4+ / CD44+ / FYN+ Axon", "NonProgressor"),
    ("CXCR4+ / CD44+ / FYN+ Axon", "Progressor"),
    ("NRXN1+ / MAOA+ Axon", "NonProgressor"),
    ("NRXN1+ / MAOA+ Axon", "Progressor"),
]
ROW_LABELS = {
    ("CXCR4+ / CD44+ / FYN+ Axon", "NonProgressor"): "CXCR4+ CD44+ FYN+ axon\nNon-progressor",
    ("CXCR4+ / CD44+ / FYN+ Axon", "Progressor"): "CXCR4+ CD44+ FYN+ axon\nProgressor",
    ("NRXN1+ / MAOA+ Axon", "NonProgressor"): "NRXN1+ MAOA+ axon\nNon-progressor",
    ("NRXN1+ / MAOA+ Axon", "Progressor"): "NRXN1+ MAOA+ axon\nProgressor",
}
POS_COLOR = "#C04A69"
NEG_COLOR = "#D8DCE1"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
        }
    )


def save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    region = pd.read_csv(REGION_TABLE)
    data = region.loc[
        region["marker"].eq(MARKER)
        & region["progression"].isin(["NonProgressor", "Progressor"])
        & region["radius_um"].isin(RADIUS_ORDER)
    ].copy()

    pooled = (
        data.groupby(["axon", "progression", "radius_um"], observed=True)
        .agg(
            n_regions=("unique_region", "nunique"),
            positive=("raw_positive_near", "sum"),
            negative=("raw_negative_near", "sum"),
            total=("raw_marker_split_near_total", "sum"),
        )
        .reset_index()
    )

    fig = plt.figure(figsize=(16.4, 18.2))
    gs = fig.add_gridspec(
        len(ROW_ORDER),
        len(RADIUS_ORDER) + 1,
        width_ratios=[1.50, 1.0, 1.0, 1.0],
        hspace=0.24,
        wspace=0.36,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=17,
            markerfacecolor=POS_COLOR,
            markeredgecolor="black",
            markeredgewidth=1.8,
            label="PD1+ T cells",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=17,
            markerfacecolor=NEG_COLOR,
            markeredgecolor="black",
            markeredgewidth=1.8,
            label="PD1- T cells",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.59, 0.975),
        ncol=2,
        prop={"size": 28},
        columnspacing=2.6,
        handletextpad=0.9,
    )

    for x_position, radius in zip([0.455, 0.682, 0.91], RADIUS_ORDER):
        fig.text(x_position, 0.905, f"{radius} um", ha="center", va="bottom", fontsize=32)

    for row_index, (axon, progression) in enumerate(ROW_ORDER):
        label_ax = fig.add_subplot(gs[row_index, 0])
        label_ax.axis("off")
        label_ax.text(
            0.98,
            0.5,
            ROW_LABELS[(axon, progression)],
            ha="right",
            va="center",
            fontsize=26,
            linespacing=1.1,
        )

        for col_index, radius in enumerate(RADIUS_ORDER, start=1):
            ax = fig.add_subplot(gs[row_index, col_index])
            ax.set_aspect("equal")
            ax.axis("off")
            row = pooled.loc[
                pooled["axon"].eq(axon)
                & pooled["progression"].eq(progression)
                & pooled["radius_um"].eq(radius)
            ]
            if row.empty:
                continue
            values = row.iloc[0]
            positive = float(values["positive"])
            negative = float(values["negative"])
            total = float(values["total"])
            pct = int(round(100.0 * positive / total)) if total > 0 else 0
            ax.pie(
                [positive, negative],
                colors=[POS_COLOR, NEG_COLOR],
                startangle=90,
                counterclock=False,
                radius=1.0,
                wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2.8},
            )
            ax.text(
                0,
                0,
                f"{pct}%\nPD1+",
                ha="center",
                va="center",
                fontsize=25,
                linespacing=1.0,
            )
            ax.text(
                0.5,
                -0.15,
                f"n={int(values['n_regions'])} regions\n{int(positive)}/{int(total)} nearby",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=18,
                color="#555555",
            )

    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.055, top=0.85)
    save_all(fig, OUTPUT_BASE)
    print(OUTPUT_BASE.with_suffix(".pdf"))
    print(OUTPUT_BASE.with_suffix(".png"))
    print(OUTPUT_BASE.with_suffix(".svg"))


if __name__ == "__main__":
    main()
