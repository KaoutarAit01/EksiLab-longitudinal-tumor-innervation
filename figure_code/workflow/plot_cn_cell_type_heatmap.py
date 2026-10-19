#!/usr/bin/env python3
"""Plot cellular-neighborhood cell-type heatmaps.

This script reproduces the heatmap plot from
https://github.com/nolanlab/NeighborhoodCoordination using
the CN labels already stored in the AnnData files. For each cohort and CN, it
computes cell-type composition and plots log2 enrichment relative to the
whole-tissue cell-type composition for that cohort.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


DEFAULT_H5ADS = [
    Path("data/AS_cellular_neighborhoods.h5ad"),
    Path("data/RP_cellular_neighborhoods.h5ad"),
]
DEFAULT_OUTDIR = Path("manuscript final figures/figures")
DEFAULT_TABLE_DIR = Path("manuscript final figures/tables")
DEFAULT_PREFIX = "fig_cn_cell_type_heatmap"

MPLCONFIG_DIR = Path("manuscript final figures/mplconfig")
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import anndata as ad
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
    "CD90+ Axon",
    "CD90+ / INA+ Axon",
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
    "CD90+ Axon": "CD90+ axon",
    "CD90+ / INA+ Axon": "CD90+ / INA+ axon",
    "T Cells": "T cells",
    "CD4+ T Cells": "CD4+ T cells",
    "FOXP3+ T Cells": "FOXP3+ T cells",
    "Macrophages": "Macrophages",
    "CD20+": "CD20+",
    "CD66b+": "CD66b+",
    "Tryp+": "Tryp+",
    "AR+ Epithelia": "AR+ epithelia",
    "Basal Epithelia": "Basal epithelia",
    "AMACR-high Epithelia": "AMACR-high epithelia",
    "aSMA+": "aSMA+",
    "AR+aSMA+": "AR+ aSMA+",
    "Other Stroma": "Other stroma",
    "Mesenchymal/Endothelial": "Mesenchymal /\nEndothelial",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper-ready CN cell-type log2 enrichment heatmaps."
    )
    parser.add_argument("--h5ad", type=Path, nargs="+", default=DEFAULT_H5ADS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--cn-column", default="CN_k50_n10")
    parser.add_argument("--cell-type-column", default="cell_type_sync")
    parser.add_argument(
        "--cohort",
        choices=["all", "as", "rp", "combined"],
        default="all",
        help="Which cohort heatmap to write. 'all' writes AS, RP, and combined.",
    )
    parser.add_argument(
        "--orientation",
        choices=["cell_types_rows", "cn_rows"],
        default="cell_types_rows",
        help="Use cell types as rows, or match the notebook with CNs as rows.",
    )
    parser.add_argument(
        "--exclude-cn",
        nargs="*",
        default=[],
        help="Optional CN labels to remove, e.g. --exclude-cn 1.",
    )
    parser.add_argument(
        "--max-abs",
        type=float,
        default=3.0,
        help="Symmetric color limit. The notebook used -3 to 3.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Override figure size in inches.",
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


def infer_cohort(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("as_"):
        return "as"
    if name.startswith("rp_"):
        return "rp"
    return path.stem


def clean_category_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.select_dtypes(["category"]).columns:
        out[column] = out[column].astype(str)
    return out


def read_cn_obs(paths: list[Path], cn_column: str, cell_type_column: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        cohort = infer_cohort(path)
        adata = ad.read_h5ad(path, backed="r")
        try:
            missing = [column for column in [cn_column, cell_type_column] if column not in adata.obs.columns]
            if missing:
                raise KeyError(f"{path} is missing required obs columns: {missing}")
            obs = clean_category_frame(adata.obs[[cn_column, cell_type_column]].copy())
        finally:
            adata.file.close()

        obs = obs.rename(columns={cn_column: "cn", cell_type_column: "cell_type"})
        obs["cohort"] = cohort
        obs["dataset"] = path.name
        obs["cn"] = obs["cn"].astype(str)
        obs["cell_type"] = obs["cell_type"].astype(str)
        obs = obs.loc[
            obs["cn"].notna()
            & obs["cell_type"].notna()
            & ~obs["cn"].str.lower().isin(["", "nan", "none", "na"])
            & ~obs["cell_type"].str.lower().isin(["", "nan", "none", "na"])
        ].copy()
        frames.append(obs[["dataset", "cohort", "cn", "cell_type"]])

    if not frames:
        raise ValueError("No CN observations were read.")
    return pd.concat(frames, ignore_index=True)


def sort_cn_labels(labels: pd.Index | list[str]) -> list[str]:
    def key(label: str) -> tuple[int, object]:
        text = str(label)
        try:
            return (0, int(float(text)))
        except ValueError:
            return (1, text)

    return sorted([str(label) for label in labels], key=key)


def ordered_cell_types(labels: pd.Index | list[str]) -> list[str]:
    labels = [str(label) for label in labels]
    preferred = [label for label in CELL_TYPE_ORDER if label in labels]
    rest = sorted(label for label in labels if label not in preferred)
    return preferred + rest


def display_cell_type(label: str) -> str:
    return CELL_LABELS.get(label, str(label).replace("_", " "))


def display_cell_type_single_line(label: str) -> str:
    return display_cell_type(label).replace("\n", " ")


def cohort_title_label(cohort: str) -> str:
    if cohort == "as":
        return "AS"
    if cohort == "rp":
        return "RP"
    if cohort == "combined":
        return "AS + RP"
    return cohort


def compute_enrichment(obs: pd.DataFrame, exclude_cn: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = obs.loc[~obs["cn"].astype(str).isin(exclude_cn)].copy()
    if obs.empty:
        raise ValueError("No observations remain after CN filtering.")

    counts = (
        obs.groupby(["cn", "cell_type"], observed=True)
        .size()
        .unstack("cell_type", fill_value=0)
    )
    counts = counts.reindex(index=sort_cn_labels(counts.index), columns=ordered_cell_types(counts.columns), fill_value=0)

    overall_counts = obs["cell_type"].value_counts().reindex(counts.columns, fill_value=0).astype(float)
    overall_fraction = overall_counts / overall_counts.sum()
    if float(overall_fraction.sum()) <= 0:
        raise ValueError("Overall cell-type fractions could not be computed.")

    row_totals = counts.sum(axis=1).astype(float)
    smoothed_fractions = counts.astype(float).add(overall_fraction, axis=1)
    smoothed_fractions = smoothed_fractions.div(row_totals + overall_fraction.sum(), axis=0)
    enrichment = np.log2(smoothed_fractions.div(overall_fraction, axis=1))
    enrichment = enrichment.replace([np.inf, -np.inf], np.nan).fillna(0)

    composition = counts.astype(float).div(row_totals, axis=0).fillna(0)
    return enrichment, composition, counts


def default_figsize(matrix: pd.DataFrame, orientation: str) -> tuple[float, float]:
    if orientation == "cell_types_rows":
        width = max(13.5, 1.05 * matrix.shape[1] + 5.2)
        height = max(17.5, 1.05 * matrix.shape[0] + 3.0)
    else:
        width = max(24.0, 1.35 * matrix.shape[1] + 7.5)
        height = max(9.5, 0.82 * matrix.shape[0] + 3.0)
    return width, height


def save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.45)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.45)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.45)
    plt.close(fig)


def write_tables(
    output_base: Path,
    table_dir: Path,
    enrichment: pd.DataFrame,
    composition: pd.DataFrame,
    counts: pd.DataFrame,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    enrichment.to_csv(table_dir / f"{output_base.name}_log2_enrichment_matrix.csv")
    composition.to_csv(table_dir / f"{output_base.name}_composition_matrix.csv")
    counts.to_csv(table_dir / f"{output_base.name}_counts_matrix.csv")


def plot_heatmap(
    enrichment: pd.DataFrame,
    output_base: Path,
    orientation: str,
    max_abs: float,
    figsize: tuple[float, float] | None,
) -> None:
    if orientation == "cell_types_rows":
        heat = enrichment.T.copy()
        heat.index = [display_cell_type(label) for label in heat.index]
        heat.columns = [f"CN {label}" for label in heat.columns]
        x_label = "Cellular neighborhood"
        y_label = "Cell type"
        xtick_rotation = 35
        ytick_rotation = 0
    else:
        heat = enrichment.copy()
        heat.index = [f"CN {label}" for label in heat.index]
        heat.columns = [display_cell_type_single_line(label) for label in heat.columns]
        x_label = "Cell type"
        y_label = "Cellular neighborhood"
        xtick_rotation = 55
        ytick_rotation = 0

    if figsize is None:
        figsize = default_figsize(heat, orientation)

    max_abs = float(max_abs)
    if not np.isfinite(max_abs) or max_abs <= 0:
        finite = heat.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        max_abs = max(1.0, math.ceil(float(np.nanpercentile(np.abs(finite), 98)) * 10.0) / 10.0)

    fig, ax = plt.subplots(figsize=figsize)
    plot = sns.heatmap(
        heat,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        linewidths=1.1,
        linecolor="white",
        cbar_kws={"pad": 0.025, "shrink": 0.75},
    )

    ax.set_title("")
    ax.set_xlabel(x_label, fontsize=38, labelpad=22)
    ax.set_ylabel(y_label, fontsize=38, labelpad=24)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, ha="right" if xtick_rotation else "center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation)
    ax.tick_params(axis="both", labelsize=30, length=0)

    cbar = plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28, width=1.8, length=8)
    cbar.ax.set_ylabel("log2 enrichment vs tissue", fontsize=32, labelpad=22)

    sns.despine(fig=fig, left=True, bottom=True)
    if orientation == "cell_types_rows":
        fig.subplots_adjust(left=0.32, right=0.88, bottom=0.2, top=0.985)
    else:
        fig.subplots_adjust(left=0.16, right=0.9, bottom=0.54, top=0.96)
    save_all(fig, output_base)


def cohorts_to_plot(obs: pd.DataFrame, choice: str) -> dict[str, pd.DataFrame]:
    if choice == "combined":
        return {"combined": obs.copy()}
    if choice == "all":
        out = {cohort: group.copy() for cohort, group in obs.groupby("cohort", observed=True)}
        out["combined"] = obs.copy()
        return out
    return {choice: obs.loc[obs["cohort"].eq(choice)].copy()}


def main() -> None:
    args = parse_args()
    setup_style()

    obs = read_cn_obs(args.h5ad, args.cn_column, args.cell_type_column)
    exclude_cn = {str(label) for label in args.exclude_cn}
    figure_paths = []
    for cohort, cohort_obs in cohorts_to_plot(obs, args.cohort).items():
        if cohort_obs.empty:
            continue
        enrichment, composition, counts = compute_enrichment(cohort_obs, exclude_cn)
        output_base = args.outdir / f"{args.prefix}_{cohort}"
        write_tables(output_base, args.table_dir, enrichment, composition, counts)
        plot_heatmap(
            enrichment,
            output_base,
            orientation=args.orientation,
            max_abs=args.max_abs,
            figsize=tuple(args.figsize) if args.figsize is not None else None,
        )
        figure_paths.extend(
            [
                output_base.with_suffix(".png"),
                output_base.with_suffix(".pdf"),
                output_base.with_suffix(".svg"),
            ]
        )
        print(f"Wrote {cohort_title_label(cohort)} heatmap: {output_base.with_suffix('.pdf')}")

    if not figure_paths:
        raise RuntimeError("No heatmaps were generated.")


if __name__ == "__main__":
    main()
