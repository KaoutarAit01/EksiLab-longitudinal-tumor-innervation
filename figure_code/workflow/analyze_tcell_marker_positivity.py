#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

Path("manuscript_panels_results/logs/mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "manuscript_panels_results/logs/mplconfig")

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


DEFAULT_AS_MARKER_H5AD = Path(
    "data/AS_T_cell_marker_cell_types.h5ad"
)
DEFAULT_OUTDIR = Path("manuscript_panels_results/whole_tissue_tcell_marker_progression")

PROGRESSION_ORDER = ["NonProgressor", "Progressor"]
GRADE_ORDER = ["Low-grade", "High-grade"]
MARKER_ORDER = ["AR", "Ki67", "GZMB", "PD1"]

ANALYSIS_SETTINGS = {
    "progression": {
        "group_col": "progression",
        "group_order": PROGRESSION_ORDER,
        "output_prefix": "whole_tissue_tcell_marker_progression",
        "default_outdir": DEFAULT_OUTDIR,
        "title_context": "progressors vs non-progressors",
    },
    "grade": {
        "group_col": "condition",
        "group_order": GRADE_ORDER,
        "output_prefix": "whole_tissue_tcell_marker_grade",
        "default_outdir": Path("manuscript_panels_results/whole_tissue_tcell_marker_grade"),
        "title_context": "low-grade vs high-grade",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whole-sample T-cell marker positivity analyses.")
    parser.add_argument("--h5ad", type=Path, nargs="+", default=[DEFAULT_AS_MARKER_H5AD])
    parser.add_argument("--analysis", choices=sorted(ANALYSIS_SETTINGS), default="progression")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--cohort-label", default=None)
    parser.add_argument("--markers", nargs="+", default=None)
    parser.add_argument(
        "--allow-partial-markers",
        action="store_true",
        help="Include markers present in at least one input. By default, combined runs use only markers present in every input.",
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


def available_markers(path: Path) -> list[str]:
    a = ad.read_h5ad(path, backed="r")
    markers = []
    for marker in MARKER_ORDER:
        positivity_col = f"{marker}_cell_positivity"
        split_col = f"cell_type_sync_{marker}"
        if positivity_col in a.obs.columns:
            markers.append(marker)
            continue
        if split_col in a.obs.columns:
            split = a.obs[split_col].astype(str)
            has_marker_labels = split.str.contains(rf"{marker}[-+] T Cells", case=False, regex=True, na=False).any()
            if has_marker_labels:
                markers.append(marker)
    return markers


def selected_markers(paths: list[Path], requested: list[str] | None, allow_partial: bool) -> list[str]:
    available = [set(available_markers(path)) for path in paths]
    if not available:
        return []
    pool = set.union(*available) if allow_partial else set.intersection(*available)
    if requested:
        requested_set = set(requested)
        pool &= requested_set
    return [marker for marker in MARKER_ORDER if marker in pool]


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


def load_marker_obs(
    paths: list[Path],
    group_col: str,
    group_order: list[str],
    markers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_cells = []
    marker_frames = []

    for path in paths:
        a = ad.read_h5ad(path, backed="r")
        required = [group_col, "unique_region", "x", "y", "cell_type_sync"]
        missing = [column for column in required if column not in a.obs.columns]
        if missing:
            raise ValueError(f"{path} is missing required obs columns: {missing}")

        marker_cols = []
        for marker in markers:
            for column in [f"{marker}_cell_positivity", f"cell_type_sync_{marker}"]:
                if column in a.obs.columns:
                    marker_cols.append(column)

        obs = a.obs[required + sorted(set(marker_cols))].copy()
        obs[group_col] = obs[group_col].astype(str)
        obs = obs.loc[obs[group_col].isin(group_order)].copy()
        obs["unique_region"] = obs["unique_region"].astype(str)
        obs["x"] = pd.to_numeric(obs["x"], errors="coerce")
        obs["y"] = pd.to_numeric(obs["y"], errors="coerce")
        obs["cell_type_sync"] = obs["cell_type_sync"].astype(str)
        obs["cohort"] = infer_cohort(path)
        obs["source_h5ad"] = path.name

        all_cells.append(obs[[group_col, "unique_region", "x", "y", "cell_type_sync", "cohort", "source_h5ad"]])

        t_cells = obs.loc[obs["cell_type_sync"].str.contains("T Cells", case=False, na=False)].copy()
        for marker in markers:
            positivity_col = f"{marker}_cell_positivity"
            split_col = f"cell_type_sync_{marker}"
            if positivity_col in t_cells.columns:
                status = t_cells[positivity_col].astype(str).str.lower()
                positive = status.eq("positive")
                valid = status.isin(["positive", "negative"])
            elif split_col in t_cells.columns:
                split = t_cells[split_col].astype(str)
                positive = split.str.contains(rf"{marker}\+ T Cells", case=False, regex=True, na=False)
                valid = split.str.contains(rf"{marker}[-+] T Cells", case=False, regex=True, na=False)
            else:
                continue

            frame = t_cells.loc[valid, [group_col, "unique_region", "x", "y", "cell_type_sync", "cohort", "source_h5ad"]].copy()
            frame["marker"] = marker
            frame["marker_positive"] = positive.loc[valid].to_numpy(dtype=bool)
            marker_frames.append(frame)

    if not all_cells:
        raise ValueError("No H5AD inputs were provided.")

    cell_obs = pd.concat(all_cells, ignore_index=True)
    marker_obs = pd.concat(marker_frames, ignore_index=True) if marker_frames else pd.DataFrame()
    if len(paths) > 1:
        cell_obs["unique_region"] = cell_obs["cohort"].astype(str) + ":" + cell_obs["unique_region"].astype(str)
        if not marker_obs.empty:
            marker_obs["unique_region"] = marker_obs["cohort"].astype(str) + ":" + marker_obs["unique_region"].astype(str)
    return cell_obs, marker_obs


def compute_region_areas(cell_obs: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for (group, region), grp in cell_obs.groupby([group_col, "unique_region"], dropna=False):
        area = convex_hull_area(grp[["x", "y"]].to_numpy())
        if not np.isfinite(area) or area <= 0:
            continue
        rows.append(
            {
                group_col: group,
                "unique_region": region,
                "hull_area_proxy": area,
                "total_cells": int(grp.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_markers(
    marker_obs: pd.DataFrame,
    region_areas: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    markers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if marker_obs.empty:
        raise ValueError("No T-cell marker observations were available after filtering.")

    summary = (
        marker_obs.groupby([group_col, "marker"], as_index=False, observed=True)
        .agg(
            n_t_cells=("marker_positive", "size"),
            n_marker_positive=("marker_positive", "sum"),
        )
    )
    summary["positive_fraction"] = summary["n_marker_positive"] / summary["n_t_cells"]
    summary[group_col] = pd.Categorical(summary[group_col], categories=group_order, ordered=True)
    summary["marker"] = pd.Categorical(summary["marker"], categories=markers, ordered=True)
    summary = summary.sort_values([group_col, "marker"]).reset_index(drop=True)

    region = (
        marker_obs.groupby([group_col, "unique_region", "marker"], as_index=False, observed=True)
        .agg(
            n_t_cells=("marker_positive", "size"),
            n_marker_positive=("marker_positive", "sum"),
        )
    )
    region = region.merge(region_areas, on=[group_col, "unique_region"], how="left")
    region["positive_fraction"] = region["n_marker_positive"] / region["n_t_cells"]
    region["marker_positive_density_proxy"] = region["n_marker_positive"] / region["hull_area_proxy"]
    region["tcell_density_proxy"] = region["n_t_cells"] / region["hull_area_proxy"]
    region[group_col] = pd.Categorical(region[group_col], categories=group_order, ordered=True)
    region["marker"] = pd.Categorical(region["marker"], categories=markers, ordered=True)
    region = region.sort_values([group_col, "unique_region", "marker"]).reset_index(drop=True)
    return summary, region


def marker_tests(region: pd.DataFrame, value_col: str, group_col: str, group_order: list[str], markers: list[str]) -> pd.DataFrame:
    rows = []
    group_a, group_b = group_order[:2]
    for marker in markers:
        sub = region.loc[region["marker"].astype(str).eq(marker)]
        a = sub.loc[sub[group_col].astype(str).eq(group_a), value_col].dropna().to_numpy()
        b = sub.loc[sub[group_col].astype(str).eq(group_b), value_col].dropna().to_numpy()
        if a.size and b.size:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
        else:
            stat, p = np.nan, np.nan
        rows.append(
            {
                "endpoint": value_col,
                "marker": marker,
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
    out = pd.DataFrame(rows)
    if out["p_value"].notna().any():
        out.loc[out["p_value"].notna(), "q_value"] = multipletests(out.loc[out["p_value"].notna(), "p_value"], method="fdr_bh")[1]
    return out


def annotate_pvalues(ax, tests: pd.DataFrame, markers: list[str]) -> None:
    ymin, ymax = ax.get_ylim()
    y = ymax - 0.05 * (ymax - ymin)
    for i, marker in enumerate(markers):
        row = tests.loc[tests["marker"].eq(marker)]
        if row.empty or pd.isna(row.iloc[0]["p_value"]):
            text = "p=NA"
        else:
            text = f"p={row.iloc[0]['p_value']:.2g}"
        ax.text(i, y, text, ha="center", va="top", fontsize=8)


def plot_overview(
    summary: pd.DataFrame,
    region: pd.DataFrame,
    fraction_tests: pd.DataFrame,
    density_tests: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    markers: list[str],
    title: str,
    out: Path,
) -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    palette = dict(zip(group_order, ["#1f77b4", "#c43c3c"]))

    fig = plt.figure(figsize=(15.2, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.25], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    sns.barplot(
        data=summary,
        x="marker",
        y="positive_fraction",
        hue=group_col,
        order=markers,
        hue_order=group_order,
        palette=palette,
        ax=ax0,
    )
    ax0.set_xlabel("")
    ax0.set_ylabel("Fraction of T cells positive")
    ax0.set_title("Whole-sample T-cell marker positivity")
    ax0.set_ylim(0, max(0.05, min(1.0, summary["positive_fraction"].max() * 1.25)))
    n_text = []
    for group in group_order:
        sub = summary.loc[summary[group_col].astype(str).eq(group)]
        if not sub.empty:
            n_text.append(f"{group} nT={int(sub['n_t_cells'].max()):,}")
    ax0.text(0.02, 0.98, "; ".join(n_text), transform=ax0.transAxes, ha="left", va="top", fontsize=8)

    sns.boxplot(
        data=region,
        x="marker",
        y="positive_fraction",
        hue=group_col,
        order=markers,
        hue_order=group_order,
        palette=palette,
        ax=ax1,
        width=0.65,
        fliersize=0,
    )
    sns.stripplot(
        data=region,
        x="marker",
        y="positive_fraction",
        hue=group_col,
        order=markers,
        hue_order=group_order,
        dodge=True,
        ax=ax1,
        color="black",
        alpha=0.35,
        size=3,
        legend=False,
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Positive fraction per region")
    ax1.set_title("Region-level T-cell marker positivity")
    annotate_pvalues(ax1, fraction_tests, markers)

    sns.boxplot(
        data=region,
        x="marker",
        y="marker_positive_density_proxy",
        hue=group_col,
        order=markers,
        hue_order=group_order,
        palette=palette,
        ax=ax2,
        width=0.65,
        fliersize=0,
    )
    sns.stripplot(
        data=region,
        x="marker",
        y="marker_positive_density_proxy",
        hue=group_col,
        order=markers,
        hue_order=group_order,
        dodge=True,
        ax=ax2,
        color="black",
        alpha=0.35,
        size=3,
        legend=False,
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("Positive T-cell density proxy")
    ax2.set_title("Coordinate-space positive T-cell density")
    annotate_pvalues(ax2, density_tests, markers)
    ax2.text(
        0.02,
        -0.22,
        "Density uses a convex-hull area proxy from cell coordinates because no tissue-mask area is stored in the repo.",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    handles, labels = ax0.get_legend_handles_labels()
    ax0.get_legend().remove()
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(group_order), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.06, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
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
    markers = selected_markers(args.h5ad, args.markers, args.allow_partial_markers)
    if not markers:
        raise ValueError("No marker positivity columns were available for the requested inputs.")

    title = f"T-cell marker positivity in {cohort_label} {settings['title_context']}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    cell_obs, marker_obs = load_marker_obs(args.h5ad, group_col, group_order, markers)
    region_areas = compute_region_areas(cell_obs, group_col)
    summary, region = summarize_markers(marker_obs, region_areas, group_col, group_order, markers)
    fraction_tests = marker_tests(region, "positive_fraction", group_col, group_order, markers)
    density_tests = marker_tests(region, "marker_positive_density_proxy", group_col, group_order, markers)
    all_tests = pd.concat([fraction_tests, density_tests], ignore_index=True)
    if all_tests["p_value"].notna().any():
        all_tests.loc[all_tests["p_value"].notna(), "q_value_all_endpoints"] = multipletests(
            all_tests.loc[all_tests["p_value"].notna(), "p_value"],
            method="fdr_bh",
        )[1]

    manifest = pd.DataFrame(
        {
            "source_h5ad": [str(path) for path in args.h5ad],
            "cohort": [infer_cohort(path) for path in args.h5ad],
            "markers_used": [",".join(markers)] * len(args.h5ad),
            "analysis": args.analysis,
            "group_column": group_col,
        }
    )

    summary_path = outdir / "tables" / f"{output_prefix}_summary.csv"
    region_path = outdir / "tables" / f"{output_prefix}_region_stats.csv"
    area_path = outdir / "tables" / f"{output_prefix}_region_area_proxy.csv"
    fraction_tests_path = outdir / "tables" / f"{output_prefix}_region_fraction_tests.csv"
    density_tests_path = outdir / "tables" / f"{output_prefix}_density_tests.csv"
    all_tests_path = outdir / "tables" / f"{output_prefix}_tests.csv"
    manifest_path = outdir / "tables" / f"{output_prefix}_manifest.csv"
    figure_path = outdir / "figures" / f"fig_{output_prefix}_overview.png"

    summary.to_csv(summary_path, index=False)
    region.to_csv(region_path, index=False)
    region_areas.to_csv(area_path, index=False)
    fraction_tests.to_csv(fraction_tests_path, index=False)
    density_tests.to_csv(density_tests_path, index=False)
    all_tests.to_csv(all_tests_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    plot_overview(summary, region, fraction_tests, density_tests, group_col, group_order, markers, title, figure_path)

    print(outdir)
    print(summary_path)
    print(region_path)
    print(fraction_tests_path)
    print(density_tests_path)
    print(all_tests_path)
    print(figure_path)


if __name__ == "__main__":
    main()
