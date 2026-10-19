#!/usr/bin/env python3
"""Plot progression-associated marker effects."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_MASTER = Path("manuscript_panels_results/tables/focused_fdr_master_analysis_table.csv")
DEFAULT_OUTDIR = Path("manuscript_panels_results/figures/progression_story")

DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)
MPLCONFIG_DIR = Path("manuscript_panels_results/logs/mplconfig")
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


RADIUS_ORDER = [20, 50, 100]
ABUNDANCE_ORDER = ["AR+ T Cells", "Ki67+ T Cells", "GZMB+ T Cells"]
COMPOSITION_ORDER = ["PD1+ fraction among nearby T Cells"]
AXON_ORDER = ["CXCR4+ / CD44+ / FYN+ Axon", "NRXN1+ / MAOA+ Axon"]

ABUNDANCE_LABELS = {
    "AR+ T Cells": "AR+ T cells",
    "Ki67+ T Cells": "Ki67+ T cells",
    "GZMB+ T Cells": "GZMB+ T cells",
}
AXON_LABELS = {
    "CXCR4+ / CD44+ / FYN+ Axon": "CXCR4+ / CD44+ / FYN+",
    "NRXN1+ / MAOA+ Axon": "NRXN1+ / MAOA+",
}
PALETTE = {
    "AR+ T Cells": "#1f77b4",
    "Ki67+ T Cells": "#ff7f0e",
    "GZMB+ T Cells": "#2ca02c",
    "CXCR4+ / CD44+ / FYN+ Axon": "#7b3294",
    "NRXN1+ / MAOA+ Axon": "#c51b7d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make separate no-title manuscript panels from progression marker-specific story data."
    )
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default="fig_progression_marker_specific_changes")
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
            "lines.solid_capstyle": "round",
        }
    )
    sns.set_theme(style="white", context="talk")


def clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ensure_text(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def prepare_progression_subset(master: pd.DataFrame, analysis_family: str) -> pd.DataFrame:
    family_col = "analysis_family" if "analysis_family" in master.columns else "analysis"
    mask = master[family_col].eq(analysis_family)
    if "cohort" in master.columns:
        mask &= master["cohort"].eq("as")
    sub = master.loc[mask].copy()
    sub = clean_numeric(sub, ["radius_um", "effect_size", "p_value", "q_value"])
    sub = ensure_text(sub, ["axon_a", "cell_type", "endpoint"])
    return sub


def significance_star(value: object, threshold: float = 0.05) -> str:
    q_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(q_value):
        return ""
    return "*" if q_value < threshold else ""


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_line_legend(fig: plt.Figure, labels: list[str], font_size: float, ncol: int) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE[label],
            marker="o",
            markersize=12,
            linewidth=4.5,
            label=ABUNDANCE_LABELS.get(label, AXON_LABELS.get(label, label)),
        )
        for label in labels
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.15),
        ncol=ncol,
        prop={"size": font_size},
        handlelength=2.0,
        handletextpad=0.7,
        columnspacing=1.2,
    )


def finish_axis(ax: plt.Axes, y_values: pd.Series, star_pad: float) -> None:
    ax.axhline(0, color="#666666", linewidth=2.0, zorder=1)
    ax.set_xlim(14, 106)
    ax.set_xticks(RADIUS_ORDER)
    ax.set_xlabel("Radius (um)", fontsize=38, labelpad=12)
    ax.set_ylabel("Progressor effect", fontsize=38, labelpad=12)
    ax.tick_params(axis="both", labelsize=34)
    ax.grid(axis="y", color="#d8d8d8", linewidth=1.1)

    finite = pd.to_numeric(y_values, errors="coerce").dropna().to_numpy(dtype=float)
    if finite.size:
        lower = min(float(finite.min()), 0.0)
        upper = max(float(finite.max()), 0.0)
        span = max(upper - lower, abs(star_pad) * 4.0, 1.0)
        ax.set_ylim(lower - 0.12 * span, upper + 0.14 * span)

    sns.despine(ax=ax)


def plot_abundance_panel(master: pd.DataFrame, output_base: Path) -> None:
    sub = prepare_progression_subset(master, "progression_marker_abundance")
    abundance = sub.loc[sub["axon_a"].eq("NRXN1+ / MAOA+ Axon")].copy()
    abundance = abundance.loc[abundance["cell_type"].isin(ABUNDANCE_ORDER)].copy()
    abundance = abundance.groupby(["cell_type", "radius_um"], as_index=False).agg(
        effect_size=("effect_size", "mean"),
        q_value=("q_value", "min"),
    )
    abundance["cell_type"] = pd.Categorical(abundance["cell_type"], categories=ABUNDANCE_ORDER, ordered=True)
    abundance = abundance.sort_values(["cell_type", "radius_um"])

    fig, ax = plt.subplots(figsize=(8.4, 9.2))
    for marker in ABUNDANCE_ORDER:
        grp = abundance.loc[abundance["cell_type"].eq(marker)].sort_values("radius_um")
        if grp.empty:
            continue
        ax.plot(
            grp["radius_um"],
            grp["effect_size"],
            marker="o",
            markersize=13,
            linewidth=4.5,
            color=PALETTE[marker],
            label=ABUNDANCE_LABELS[marker],
            zorder=3,
        )
        for _, row in grp.iterrows():
            ax.text(
                row["radius_um"],
                row["effect_size"] + 0.18,
                significance_star(row["q_value"]),
                ha="center",
                va="bottom",
                fontsize=36,
                color=PALETTE[marker],
            )

    finish_axis(ax, abundance["effect_size"], star_pad=0.18)
    add_line_legend(fig, ABUNDANCE_ORDER, font_size=28, ncol=1)
    fig.tight_layout(rect=[0, 0.19, 1, 1])
    save_figure(fig, output_base)


def plot_pd1_composition_panel(master: pd.DataFrame, output_base: Path) -> None:
    comp = prepare_progression_subset(master, "progression_marker_composition")
    composition = comp.loc[comp["cell_type"].isin(COMPOSITION_ORDER)].copy()
    composition = composition.loc[composition["axon_a"].isin(AXON_ORDER)].copy()
    composition = composition.groupby(["axon_a", "radius_um", "cell_type"], as_index=False).agg(
        effect_size=("effect_size", "mean"),
        q_value=("q_value", "min"),
    )
    composition["axon_a"] = pd.Categorical(composition["axon_a"], categories=AXON_ORDER, ordered=True)
    composition = composition.sort_values(["axon_a", "radius_um"])

    fig, ax = plt.subplots(figsize=(8.4, 9.2))
    for axon in AXON_ORDER:
        grp = composition.loc[composition["axon_a"].eq(axon)].sort_values("radius_um")
        if grp.empty:
            continue
        ax.plot(
            grp["radius_um"],
            grp["effect_size"],
            marker="o",
            markersize=13,
            linewidth=4.5,
            color=PALETTE[axon],
            label=AXON_LABELS[axon],
            zorder=3,
        )
        for _, row in grp.iterrows():
            ax.text(
                row["radius_um"],
                row["effect_size"] + 1.6,
                significance_star(row["q_value"]),
                ha="center",
                va="bottom",
                fontsize=36,
                color=PALETTE[axon],
            )

    finish_axis(ax, composition["effect_size"], star_pad=1.6)
    add_line_legend(fig, AXON_ORDER, font_size=25, ncol=1)
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    save_figure(fig, output_base)


def main() -> None:
    args = parse_args()
    setup_style()
    master = pd.read_csv(args.master)
    args.outdir.mkdir(parents=True, exist_ok=True)

    abundance_base = args.outdir / f"{args.prefix}_abundance_separate_panel"
    pd1_base = args.outdir / f"{args.prefix}_pd1_fraction_separate_panel"
    plot_abundance_panel(master, abundance_base)
    plot_pd1_composition_panel(master, pd1_base)

    print(abundance_base.with_suffix(".png"))
    print(abundance_base.with_suffix(".pdf"))
    print(pd1_base.with_suffix(".png"))
    print(pd1_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
