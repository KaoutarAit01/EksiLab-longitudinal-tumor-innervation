#!/usr/bin/env python3
"""Plot progression-associated local cell redistribution around axons."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_MASTER = Path("manuscript_panels_results/tables/focused_fdr_master_analysis_table.csv")
DEFAULT_OUTDIR = Path("manuscript_panels_results/figures/progression_story")

MPLCONFIG_DIR = Path("manuscript_panels_results/logs/mplconfig")
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CELL_ORDER = [
    "Macrophages",
    "T Cells",
    "CD4+ T Cells",
    "FOXP3+ T Cells",
    "CD66b+",
    "CD20+",
    "Tryp+",
]
CELL_LABELS = {
    "Macrophages": "Macrophages",
    "T Cells": "T cells",
    "CD4+ T Cells": "CD4+ T cells",
    "FOXP3+ T Cells": "FOXP3+ T cells",
    "CD66b+": "CD66b+",
    "CD20+": "CD20+",
    "Tryp+": "Tryp+",
}
ENDPOINT_ORDER = ["proximal_fraction", "log_proximal_density_ratio"]
ENDPOINT_LABELS = {
    "proximal_fraction": "Proximal fraction",
    "log_proximal_density_ratio": "Log proximal density ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a transposed manuscript-ready local redistribution heatmap."
    )
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default="fig_progression_local_redistribution")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 2.0,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
        }
    )
    sns.set_theme(style="white", context="talk")


def clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def significance_star(value: object) -> str:
    q_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(q_value):
        return ""
    return "*" if q_value < 0.05 else ""


def save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    master = pd.read_csv(args.master)

    sub = master.loc[
        master["analysis_family"].eq("progression_radial_distribution")
        & master["cohort"].eq("as")
        & master["axon_a"].eq("CXCR4+ / CD44+ / FYN+ Axon")
        & master["radius_um"].eq(100)
        & master["cell_type"].isin(CELL_ORDER)
        & master["endpoint"].isin(ENDPOINT_ORDER)
    ].copy()
    sub = clean_numeric(sub, ["effect_size", "q_value"])
    sub["cell_type"] = pd.Categorical(sub["cell_type"], categories=CELL_ORDER, ordered=True)
    sub["endpoint"] = pd.Categorical(sub["endpoint"], categories=ENDPOINT_ORDER, ordered=True)

    heat = (
        sub.pivot_table(
            index="endpoint",
            columns="cell_type",
            values="effect_size",
            aggfunc="mean",
            observed=False,
        )
        .reindex(ENDPOINT_ORDER)
        .reindex(columns=CELL_ORDER)
    )
    q_values = (
        sub.pivot_table(
            index="endpoint",
            columns="cell_type",
            values="q_value",
            aggfunc="min",
            observed=False,
        )
        .reindex(ENDPOINT_ORDER)
        .reindex(columns=CELL_ORDER)
    )
    stars = q_values.map(significance_star)
    heat.index = [ENDPOINT_LABELS[label] for label in heat.index]
    heat.columns = [CELL_LABELS[label] for label in heat.columns]
    stars.index = heat.index
    stars.columns = heat.columns

    vmax = max(0.25, float(np.nanpercentile(np.abs(heat.to_numpy(dtype=float)), 95)))
    fig, ax = plt.subplots(figsize=(18.8, 7.2))
    mask = heat.isna()
    heatmap = sns.heatmap(
        heat,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=1.4,
        linecolor="white",
        annot=stars,
        fmt="",
        annot_kws={"fontsize": 54, "fontweight": "bold"},
        mask=mask,
        cbar_kws={"pad": 0.025, "shrink": 0.86},
    )
    ax.set_xlabel("Immune cell type", fontsize=40, labelpad=24)
    ax.set_ylabel("Metric at 100 um", fontsize=40, labelpad=26)
    ax.set_title("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="both", labelsize=31, length=0)

    for text in ax.texts:
        if not text.get_text():
            continue
        col = int(round(text.get_position()[0] - 0.5))
        row = int(round(text.get_position()[1] - 0.5))
        value = heat.iloc[row, col]
        text.set_color("white" if np.isfinite(value) and abs(value) > 0.22 else "#222222")

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30, width=1.8, length=8)
    cbar.ax.set_title("Progressor\neffect", fontsize=31, pad=18)
    sns.despine(fig=fig, left=True, bottom=True)
    fig.subplots_adjust(left=0.18, right=0.91, bottom=0.36, top=0.86)

    output_base = args.outdir / args.prefix
    save_all(fig, output_base)
    print(output_base.with_suffix(".png"))
    print(output_base.with_suffix(".pdf"))
    print(output_base.with_suffix(".svg"))


if __name__ == "__main__":
    main()
