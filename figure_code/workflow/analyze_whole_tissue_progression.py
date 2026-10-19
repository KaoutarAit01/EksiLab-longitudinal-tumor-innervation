#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

Path("manuscript_panels_results/logs/mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "manuscript_panels_results/logs/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import anndata as ad
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


DEFAULT_H5AD = Path("data/AS_cellular_neighborhoods.h5ad")
DEFAULT_GRADE_OUTDIR = Path("manuscript_panels_results/whole_tissue_grade")
DEFAULT_OUTDIR = Path("manuscript_panels_results/whole_tissue_progression")

PROGRESSION_ORDER = ["NonProgressor", "Progressor"]
GRADE_ORDER = ["Low-grade", "High-grade"]
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

ANALYSIS_SETTINGS = {
    "progression": {
        "group_col": "progression",
        "group_order": PROGRESSION_ORDER,
        "output_prefix": "whole_tissue_progression",
        "default_outdir": DEFAULT_OUTDIR,
        "title_context": "progressors vs non-progressors",
    },
    "grade": {
        "group_col": "condition",
        "group_order": GRADE_ORDER,
        "output_prefix": "whole_tissue_grade",
        "default_outdir": DEFAULT_GRADE_OUTDIR,
        "title_context": "low-grade vs high-grade",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whole-tissue immune-burden analyses.")
    parser.add_argument("--h5ad", type=Path, nargs="+", default=[DEFAULT_H5AD])
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--analysis", choices=sorted(ANALYSIS_SETTINGS), default="progression")
    parser.add_argument(
        "--cohort-label",
        default=None,
        help="Display label for the figure title, for example AS, RP, or AS + RP.",
    )
    return parser.parse_args()


def infer_cohort(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("rp_"):
        return "rp"
    if name.startswith("as_"):
        return "as"
    return path.stem


def default_cohort_label(paths: list[Path]) -> str:
    cohorts = [infer_cohort(path) for path in paths]
    unique = list(dict.fromkeys(cohorts))
    if unique == ["as"]:
        return "AS"
    if unique == ["rp"]:
        return "RP"
    if set(unique) == {"as", "rp"}:
        return "AS + RP"
    return " + ".join(unique)


def load_obs(paths: list[Path], group_col: str, group_order: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        a = ad.read_h5ad(path, backed="r")
        required = [group_col, "unique_region", "x", "y", "cell_type_sync"]
        missing = [column for column in required if column not in a.obs.columns]
        if missing:
            raise ValueError(f"{path} is missing required obs columns: {missing}")

        obs = a.obs[required].copy()
        obs[group_col] = obs[group_col].astype(str)
        obs["unique_region"] = obs["unique_region"].astype(str)
        obs["x"] = pd.to_numeric(obs["x"], errors="coerce")
        obs["y"] = pd.to_numeric(obs["y"], errors="coerce")
        obs["cell_type_sync"] = obs["cell_type_sync"].astype(str)
        obs = obs.loc[obs[group_col].isin(group_order)].copy()
        obs["cohort"] = infer_cohort(path)
        obs["source_h5ad"] = path.name
        frames.append(obs)

    if not frames:
        raise ValueError("No H5AD inputs were provided.")

    obs = pd.concat(frames, ignore_index=True)
    if len(paths) > 1:
        obs["unique_region"] = obs["cohort"].astype(str) + ":" + obs["unique_region"].astype(str)
    return obs


def compute_whole_tissue_composition(obs: pd.DataFrame, group_col: str, group_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = (
        obs.groupby(group_col)
        .size()
        .reindex(group_order)
        .rename("total_cells")
        .reset_index()
    )
    immune = obs.loc[obs["cell_type_sync"].isin(IMMUNE_ORDER)].copy()
    immune_counts = (
        immune.groupby([group_col, "cell_type_sync"])
        .size()
        .rename("n_cells")
        .reset_index()
    )
    immune_counts = immune_counts.merge(total, on=group_col, how="left")
    immune_counts["fraction_of_all_cells"] = immune_counts["n_cells"] / immune_counts["total_cells"]
    immune_counts["fraction_within_immune"] = immune_counts.groupby(group_col)["n_cells"].transform(lambda s: s / s.sum())

    immune_summary = (
        immune.groupby(group_col)
        .size()
        .rename("immune_cells")
        .reset_index()
    )
    summary = total.merge(immune_summary, on=group_col, how="left").fillna({"immune_cells": 0})
    summary["immune_fraction_of_all_cells"] = summary["immune_cells"] / summary["total_cells"]
    summary["nonimmune_cells"] = summary["total_cells"] - summary["immune_cells"]
    summary["nonimmune_fraction_of_all_cells"] = summary["nonimmune_cells"] / summary["total_cells"]
    return summary, immune_counts


def region_level_fractions(obs: pd.DataFrame, group_col: str) -> pd.DataFrame:
    region_total = obs.groupby([group_col, "unique_region"], as_index=False).size().rename(columns={"size": "total_cells"})
    region_immune = (
        obs.loc[obs["cell_type_sync"].isin(IMMUNE_ORDER)]
        .groupby([group_col, "unique_region"], as_index=False)
        .size()
        .rename(columns={"size": "immune_cells"})
    )
    region = region_total.merge(region_immune, on=[group_col, "unique_region"], how="left").fillna({"immune_cells": 0})
    region["immune_fraction"] = region["immune_cells"] / region["total_cells"]

    per_cell = (
        obs.loc[obs["cell_type_sync"].isin(IMMUNE_ORDER)]
        .groupby([group_col, "unique_region", "cell_type_sync"], as_index=False)
        .size()
        .rename(columns={"size": "n_cells"})
    )
    per_cell = per_cell.merge(region_total, on=[group_col, "unique_region"], how="left")
    per_cell["fraction_of_all_cells"] = per_cell["n_cells"] / per_cell["total_cells"]
    return region, per_cell


def convex_hull_area(points: np.ndarray) -> float:
    pts = points[np.isfinite(points).all(axis=1)]
    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 3:
        return np.nan
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)
    except QhullError:
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        return float(max(xmax - xmin, 0.0) * max(ymax - ymin, 0.0))


def density_proxy(obs: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for (group, region), grp in obs.groupby([group_col, "unique_region"]):
        area = convex_hull_area(grp[["x", "y"]].to_numpy())
        if not np.isfinite(area) or area <= 0:
            continue
        immune = grp.loc[grp["cell_type_sync"].isin(IMMUNE_ORDER)]
        rows.append(
            {
                group_col: group,
                "unique_region": region,
                "hull_area_proxy": area,
                "total_cells": int(grp.shape[0]),
                "immune_cells": int(immune.shape[0]),
                "immune_density_proxy": float(immune.shape[0] / area),
                "all_cell_density_proxy": float(grp.shape[0] / area),
            }
        )
    return pd.DataFrame(rows)


def test_groups(df: pd.DataFrame, value_col: str, group_col: str, group_order: list[str]) -> pd.DataFrame:
    rows = []
    group_a, group_b = group_order[:2]
    a = df.loc[df[group_col].eq(group_a), value_col].dropna().to_numpy()
    b = df.loc[df[group_col].eq(group_b), value_col].dropna().to_numpy()
    if a.size and b.size:
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
    else:
        stat, p = np.nan, np.nan
    rows.append(
        {
            "analysis": value_col,
            "group_column": group_col,
            "group_a": group_a,
            "group_b": group_b,
            "n_group_a": int(a.size),
            "n_group_b": int(b.size),
            "median_group_a": float(np.median(a)) if a.size else np.nan,
            "median_group_b": float(np.median(b)) if b.size else np.nan,
            "delta_median_group_b_minus_group_a": float(np.median(b) - np.median(a)) if a.size and b.size else np.nan,
            "mannwhitney_u": float(stat),
            "p_value": float(p),
        }
    )
    return pd.DataFrame(rows)


def test_region_cell_types(region_per_cell: pd.DataFrame, group_col: str, group_order: list[str]) -> pd.DataFrame:
    rows = []
    group_a, group_b = group_order[:2]
    for cell_type in IMMUNE_ORDER:
        sub = region_per_cell.loc[region_per_cell["cell_type_sync"].eq(cell_type)].copy()
        if sub.empty:
            continue
        a = sub.loc[sub[group_col].eq(group_a), "fraction_of_all_cells"].dropna().to_numpy()
        b = sub.loc[sub[group_col].eq(group_b), "fraction_of_all_cells"].dropna().to_numpy()
        if a.size < 2 or b.size < 2:
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "cell_type_sync": cell_type,
                "group_column": group_col,
                "group_a": group_a,
                "group_b": group_b,
                "n_group_a": int(a.size),
                "n_group_b": int(b.size),
                "median_group_a": float(np.median(a)),
                "median_group_b": float(np.median(b)),
                "delta_median_group_b_minus_group_a": float(np.median(b) - np.median(a)),
                "mannwhitney_u": float(stat),
                "p_value": float(p),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def plot_overview(
    summary: pd.DataFrame,
    immune_counts: pd.DataFrame,
    region: pd.DataFrame,
    density_tests: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    title: str,
    out: Path,
) -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    colors = {
        "T Cells": "#1f77b4",
        "CD4+ T Cells": "#ff7f0e",
        "FOXP3+ T Cells": "#2ca02c",
        "Macrophages": "#d62728",
        "Tryp+": "#9467bd",
        "CD20+": "#8c564b",
        "CD66b+": "#e377c2",
        NONIMMUNE_LABEL: "#d9d9d9",
    }
    fig = plt.figure(figsize=(15.0, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel 1: whole-tissue composition
    pivot = (
        immune_counts.pivot(index=group_col, columns="cell_type_sync", values="fraction_of_all_cells")
        .reindex(group_order)
        .reindex(columns=IMMUNE_ORDER)
        .fillna(0.0)
    )
    nonimmune = summary.set_index(group_col).reindex(group_order)["nonimmune_fraction_of_all_cells"]
    x = np.arange(len(group_order))
    bottom = np.zeros(len(group_order))
    for label in IMMUNE_ORDER:
        vals = pivot[label].to_numpy()
        ax0.bar(x, vals, bottom=bottom, width=0.7, color=colors[label], edgecolor="white", linewidth=0.5, label=label)
        bottom += vals
    ax0.bar(x, nonimmune.to_numpy(), bottom=bottom, width=0.7, color=colors[NONIMMUNE_LABEL], edgecolor="white", linewidth=0.5, label=NONIMMUNE_LABEL)
    for i, group in enumerate(group_order):
        ax0.text(i, 1.02, f"immune={summary.set_index(group_col).loc[group, 'immune_fraction_of_all_cells']:.3f}", ha="center", va="bottom", fontsize=9)
    ax0.set_xticks(x)
    ax0.set_xticklabels(group_order)
    ax0.set_ylim(0, 1.08)
    ax0.set_ylabel("Fraction of all cells")
    ax0.set_title("Whole tissue composition")

    # Panel 2: region-level immune fraction
    sns.boxplot(data=region, x=group_col, y="immune_fraction", order=group_order, ax=ax1, palette=[colors["T Cells"], colors["Macrophages"]], width=0.55, fliersize=0)
    sns.stripplot(data=region, x=group_col, y="immune_fraction", order=group_order, ax=ax1, color="black", alpha=0.45, size=4, jitter=0.18)
    ax1.set_xlabel("")
    ax1.set_ylabel("Immune fraction per region")
    ax1.set_title("Region-level immune fraction")
    frac_test = test_groups(region, "immune_fraction", group_col, group_order).iloc[0]
    ax1.text(0.5, 0.98, f"MWU p={frac_test['p_value']:.3g}", transform=ax1.transAxes, ha="center", va="top", fontsize=9)

    # Panel 3: density proxy
    sns.boxplot(data=density_tests, x=group_col, y="immune_density_proxy", order=group_order, ax=ax2, palette=[colors["T Cells"], colors["Macrophages"]], width=0.55, fliersize=0)
    sns.stripplot(data=density_tests, x=group_col, y="immune_density_proxy", order=group_order, ax=ax2, color="black", alpha=0.45, size=4, jitter=0.18)
    ax2.set_xlabel("")
    ax2.set_ylabel("Immune density proxy")
    ax2.set_title("Coordinate-space density proxy")
    dens_test = test_groups(density_tests, "immune_density_proxy", group_col, group_order).iloc[0]
    ax2.text(0.5, 0.98, f"MWU p={dens_test['p_value']:.3g}", transform=ax2.transAxes, ha="center", va="top", fontsize=9)
    ax2.text(
        0.02,
        -0.22,
        "The density panel uses a convex-hull area proxy from cell coordinates because no tissue-mask area is stored in the repo.",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(title, y=1.06, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    settings = ANALYSIS_SETTINGS[args.analysis]
    group_col = settings["group_col"]
    group_order = settings["group_order"]
    output_prefix = settings["output_prefix"]
    outdir = args.outdir or settings["default_outdir"]
    cohort_label = args.cohort_label or default_cohort_label(args.h5ad)
    title = f"Whole-tissue immune burden in {cohort_label} {settings['title_context']}"

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    obs = load_obs(args.h5ad, group_col, group_order)
    summary, immune_counts = compute_whole_tissue_composition(obs, group_col, group_order)
    region, region_per_cell = region_level_fractions(obs, group_col)
    density = density_proxy(obs, group_col)

    summary_path = outdir / "tables" / f"{output_prefix}_summary.csv"
    immune_counts_path = outdir / "tables" / f"{output_prefix}_immune_counts.csv"
    region_path = outdir / "tables" / f"{output_prefix}_region_stats.csv"
    region_per_cell_path = outdir / "tables" / f"{output_prefix}_region_cell_type_stats.csv"
    density_path = outdir / "tables" / f"{output_prefix}_density_proxy.csv"
    region_tests_path = outdir / "tables" / f"{output_prefix}_region_tests.csv"
    density_tests_path = outdir / "tables" / f"{output_prefix}_density_tests.csv"
    cell_type_tests_path = outdir / "tables" / f"{output_prefix}_cell_type_tests.csv"
    figure_path = outdir / "figures" / f"fig_{output_prefix}_overview.png"

    summary.to_csv(summary_path, index=False)
    immune_counts.to_csv(immune_counts_path, index=False)
    region.to_csv(region_path, index=False)
    region_per_cell.to_csv(region_per_cell_path, index=False)
    density.to_csv(density_path, index=False)

    region_tests = test_groups(region, "immune_fraction", group_col, group_order)
    density_tests = test_groups(density, "immune_density_proxy", group_col, group_order)
    cell_type_tests = test_region_cell_types(region_per_cell, group_col, group_order)
    region_tests.to_csv(region_tests_path, index=False)
    density_tests.to_csv(density_tests_path, index=False)
    cell_type_tests.to_csv(cell_type_tests_path, index=False)

    plot_overview(summary, immune_counts, region, density, group_col, group_order, title, figure_path)

    print(outdir)
    print(summary_path)
    print(region_tests_path)
    print(density_tests_path)
    print(cell_type_tests_path)
    print(figure_path)


if __name__ == "__main__":
    main()
