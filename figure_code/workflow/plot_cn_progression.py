#!/usr/bin/env python3
"""Plot cellular-neighborhood enrichment during AS progression."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

DEFAULT_INDIR = Path("results/cn_neighborhood_enrichment_notebook_method_progression_as")
DEFAULT_FIGDIR = DEFAULT_INDIR / "figures"
DEFAULT_COEFFICIENTS = DEFAULT_INDIR / "tables" / "notebook_method_coefficients_as.csv"
DEFAULT_PVALUES = DEFAULT_INDIR / "tables" / "notebook_method_pvalues_as.csv"

MPLCONFIG_DIR = DEFAULT_INDIR / "mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CELL_TYPE_ORDER = [
    "CXCR4+ / CD44+ / FYN+ Axon",
    "NRXN1+ / MAOA+ Axon",
    "TH+ Axon",
    "T Cells",
    "CD4+ T Cells",
    "FOXP3+ T Cells",
    "Macrophages",
    "CD20+",
    "CD66b+",
    "Tryp+",
    "AR+ Epithelia",
    "Basal Epithelia",
    "AMACR-high Epithelia",
    "aSMA+",
    "AR+aSMA+",
    "Other Stroma",
    "Mesenchymal/Endothelial",
]

CELL_LABELS = {
    "CXCR4+ / CD44+ / FYN+ Axon": "CXCR4+ / CD44+ /\nFYN+ axon",
    "NRXN1+ / MAOA+ Axon": "NRXN1+ /\nMAOA+ axon",
    "TH+ Axon": "TH+ axon",
    "T Cells": "T cells",
    "CD4+ T Cells": "CD4+ T cells",
    "FOXP3+ T Cells": "FOXP3+ T cells",
    "Macrophages": "Macrophages",
    "CD20+": "CD20+",
    "CD66b+": "CD66b+",
    "Tryp+": "Tryp+",
    "AR+ Epithelia": "AR+ epithelia",
    "Basal Epithelia": "Basal epithelia",
    "AMACR-high Epithelia": "AMACR-high\nepithelia",
    "aSMA+": "aSMA+",
    "AR+aSMA+": "AR+ aSMA+",
    "Other Stroma": "Other stroma",
    "Mesenchymal/Endothelial": "Mesenchymal /\nEndothelial",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot the AS progression CN enrichment heatmap with manuscript styling."
    )
    parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--pvalues", type=Path, default=DEFAULT_PVALUES)
    parser.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    parser.add_argument("--prefix", default="AS_progression_paper_ready")
    parser.add_argument(
        "--horizontal",
        action="store_true",
        help="Keep the original neighborhood-by-cell-type orientation.",
    )
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="white", context="paper")
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


def ordered_columns(columns: list[str]) -> list[str]:
    preferred = [label for label in CELL_TYPE_ORDER if label in columns]
    rest = sorted(label for label in columns if label not in preferred)
    return preferred + rest


def read_tables(coefficients: Path, pvalues: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coef = pd.read_csv(coefficients, index_col=0)
    pval = pd.read_csv(pvalues, index_col=0)

    coef.index = coef.index.astype(str)
    pval.index = pval.index.astype(str)
    coef = coef.apply(pd.to_numeric, errors="coerce")
    pval = pval.apply(pd.to_numeric, errors="coerce")

    columns = ordered_columns(list(coef.columns))
    coef = coef.loc[:, columns]
    pval = pval.loc[coef.index, columns]
    return coef, pval


def star_table(pvals: pd.DataFrame) -> pd.DataFrame:
    return pvals.map(lambda value: "*" if pd.notna(value) and value < 0.05 else "")


def relabel_cells(labels: list[str]) -> list[str]:
    return [CELL_LABELS.get(label, str(label).replace("_", " ")) for label in labels]


def colorbar_limit(values: pd.DataFrame) -> float:
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    max_abs = float(np.nanmax(np.abs(finite)))
    return max(1.0, math.ceil(max_abs * 10.0) / 10.0)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def plot_transposed(coef: pd.DataFrame, pval: pd.DataFrame, output_base: Path) -> None:
    heat = coef.T.copy()
    stars = star_table(pval).T
    heat.index = relabel_cells(list(heat.index))
    stars.index = heat.index
    heat.columns = [str(label) for label in heat.columns]
    stars.columns = heat.columns

    vmax = colorbar_limit(heat)
    fig, ax = plt.subplots(figsize=(12.2, 17.2))
    plot = sns.heatmap(
        heat,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=1.25,
        linecolor="white",
        annot=stars,
        fmt="",
        annot_kws={"fontsize": 38, "fontweight": "bold"},
        cbar_kws={"pad": 0.035, "shrink": 0.72},
    )

    ax.set_title("")
    ax.set_xlabel("Cellular neighborhood", fontsize=34, labelpad=20)
    ax.set_ylabel("Cell type", fontsize=34, labelpad=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="both", labelsize=28, length=0)

    for text in ax.texts:
        if not text.get_text():
            continue
        col = int(round(text.get_position()[0] - 0.5))
        row = int(round(text.get_position()[1] - 0.5))
        value = heat.iloc[row, col]
        text.set_color("white" if np.isfinite(value) and abs(value) > 0.45 * vmax else "#1f1f1f")

    cbar = plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=26, width=1.8, length=8)
    cbar.ax.set_ylabel("Progressor enrichment", fontsize=30, labelpad=18)
    sns.despine(fig=fig, left=True, bottom=True)
    fig.subplots_adjust(left=0.36, right=0.88, bottom=0.075, top=0.985)
    save_figure(fig, output_base)


def plot_horizontal(coef: pd.DataFrame, pval: pd.DataFrame, output_base: Path) -> None:
    heat = coef.copy()
    stars = star_table(pval)
    heat.columns = relabel_cells(list(heat.columns))
    stars.columns = heat.columns

    vmax = colorbar_limit(heat)
    fig, ax = plt.subplots(figsize=(22.5, 8.2))
    plot = sns.heatmap(
        heat,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=1.15,
        linecolor="white",
        annot=stars,
        fmt="",
        annot_kws={"fontsize": 32, "fontweight": "bold"},
        cbar_kws={"pad": 0.018, "shrink": 0.82},
    )

    ax.set_title("")
    ax.set_xlabel("Cell type", fontsize=32, labelpad=24)
    ax.set_ylabel("Cellular neighborhood", fontsize=32, labelpad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="both", labelsize=25, length=0)

    for text in ax.texts:
        if not text.get_text():
            continue
        col = int(round(text.get_position()[0] - 0.5))
        row = int(round(text.get_position()[1] - 0.5))
        value = heat.iloc[row, col]
        text.set_color("white" if np.isfinite(value) and abs(value) > 0.45 * vmax else "#1f1f1f")

    cbar = plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24, width=1.8, length=8)
    cbar.ax.set_ylabel("Progressor enrichment", fontsize=28, labelpad=18)
    sns.despine(fig=fig, left=True, bottom=True)
    fig.subplots_adjust(left=0.085, right=0.92, bottom=0.47, top=0.97)
    save_figure(fig, output_base)


def main() -> None:
    args = parse_args()
    setup_style()
    coef, pval = read_tables(args.coefficients, args.pvalues)

    output_base = args.figdir / args.prefix
    if args.horizontal:
        plot_horizontal(coef, pval, output_base)
    else:
        plot_transposed(coef, pval, output_base)

    print(output_base.with_suffix(".png"))
    print(output_base.with_suffix(".pdf"))
    print(output_base.with_suffix(".svg"))


if __name__ == "__main__":
    main()
