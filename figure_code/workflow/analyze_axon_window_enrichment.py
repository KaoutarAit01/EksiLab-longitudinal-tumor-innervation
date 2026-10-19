#!/usr/bin/env python3
"""Analyze axon-window enrichment in the AS and RP cohorts."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

DEFAULT_OUTDIR = Path("results/axon_immune_spatial_enrichment_as_rp_combined_plot")
DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTDIR / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_OUTDIR / "cache"))
(DEFAULT_OUTDIR / "mplconfig").mkdir(parents=True, exist_ok=True)
(DEFAULT_OUTDIR / "cache").mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial import ConvexHull, QhullError, cKDTree


DEFAULT_H5ADS = [
    Path("data/AS_cellular_neighborhoods.h5ad"),
    Path("data/RP_cellular_neighborhoods.h5ad"),
]
DEFAULT_IMMUNE_LABELS = [
    "T Cells",
    "CD4+ T Cells",
    "FOXP3+ T Cells",
    "Macrophages",
    "Tryp+",
    "CD20+",
    "CD66b+",
]
DENSITY_SCALE = 1_000_000.0
EPS = 1e-12
COMBINED_COHORT_LABEL = "AS and RP"
COMBINED_PLOT_SUFFIX = "as_rp_combined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare immune-cell density near axons with whole-tissue density, "
            "and summarize fixed-radius local windows."
        )
    )
    parser.add_argument("--h5ad", type=Path, nargs="+", default=DEFAULT_H5ADS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--radius-um", type=float, nargs="+", default=[100.0])
    parser.add_argument(
        "--immune-label",
        "--target-label",
        dest="target_label",
        type=str,
        nargs="+",
        default=DEFAULT_IMMUNE_LABELS,
        help="Cell-type labels to analyze. --immune-label is kept as a backward-compatible alias.",
    )
    parser.add_argument("--aggregate-label", type=str, default="all_immune")
    parser.add_argument("--target-kind", type=str, default="immune")
    parser.add_argument("--file-prefix", type=str, default="axon_immune")
    parser.add_argument("--report-title", type=str, default="Axon-Immune Spatial Enrichment")
    parser.add_argument("--density-scale", type=float, default=DENSITY_SCALE)
    parser.add_argument(
        "--area-grid-step-um",
        type=float,
        default=25.0,
        help="Grid spacing for deterministic area estimates inside the convex-hull tissue proxy.",
    )
    parser.add_argument(
        "--window-spacing-um",
        type=float,
        default=None,
        help="Horizontal spacing between hex-grid window centers. Default sqrt(3) * radius.",
    )
    parser.add_argument(
        "--min-window-area-fraction",
        type=float,
        default=0.9,
        help="Discard local windows with sampled tissue area below this fraction of pi*r^2.",
    )
    parser.add_argument(
        "--min-axon-disk-area-fraction",
        type=float,
        default=0.9,
        help="Discard axon-centered disks with sampled tissue area below this fraction of pi*r^2.",
    )
    parser.add_argument(
        "--min-window-total-cells",
        type=int,
        default=1,
        help="Discard local windows containing fewer than this many observed cells.",
    )
    parser.add_argument(
        "--min-window-axon-count",
        type=int,
        default=1,
        help="Minimum axons inside a 100 um window for axon-associated classification.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for KD-tree radius queries.",
    )
    parser.add_argument(
        "--write-window-table",
        action="store_true",
        help="Write one row per local window for the aggregate target. Region summaries are always written.",
    )
    return parser.parse_args()


def infer_cohort(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("rp_"):
        return "rp"
    if name.startswith("as_"):
        return "as"
    return path.stem


def parse_patient_id(cohort: str, unique_region: object) -> str:
    text = str(unique_region)
    if cohort == "as":
        match = re.search(r"/Batch\d+/([^_]+_[^_]+_[AB])(?:_|$)", text)
        if match:
            return match.group(1)
    return text


def clean_str(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "na"}:
        return ""
    return text


def display_label(value: object) -> str:
    text = clean_str(value).replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    overrides = {
        "as": "AS",
        "rp": "RP",
        "as and rp": "AS and RP",
        "all immune": "All immune",
    }
    return overrides.get(text.lower(), text)


def mode_or_join(series: pd.Series) -> str:
    values = series.map(clean_str)
    values = values.loc[values.ne("")]
    if values.empty:
        return ""
    modes = values.mode()
    if len(modes) == 1:
        return str(modes.iloc[0])
    return "|".join(sorted(values.unique()))


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".6g", max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    if max_rows is not None:
        df = df.head(max_rows)

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value).replace("\n", " ").replace("|", r"\|")

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def bh_qvalues(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(q, index=p_values.index)
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(p[valid])]
    ranked = p[order]
    m = ranked.size
    ranked_q = ranked * m / np.arange(1, m + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    q[order] = np.clip(ranked_q, 0, 1)
    return pd.Series(q, index=p_values.index)


def safe_name(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def read_obs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    required = ["unique_region", "condition", "cell_type_sync", "x", "y"]
    optional = ["progression", "batch", "is_axon", "cell_type", "CN_k50_n10"]
    for path in paths:
        cohort = infer_cohort(path)
        adata = ad.read_h5ad(path, backed="r")
        try:
            missing = [column for column in required if column not in adata.obs.columns]
            if missing:
                raise ValueError(f"{path} is missing required obs columns: {missing}")
            columns = required + [column for column in optional if column in adata.obs.columns]
            obs = adata.obs[columns].copy()
        finally:
            adata.file.close()

        for column in obs.select_dtypes(["category"]).columns:
            obs[column] = obs[column].astype(str)
        obs["x"] = pd.to_numeric(obs["x"], errors="coerce")
        obs["y"] = pd.to_numeric(obs["y"], errors="coerce")
        obs = obs.loc[np.isfinite(obs["x"]) & np.isfinite(obs["y"])].copy()
        obs["cohort"] = cohort
        obs["dataset"] = path.name
        obs["patient_id"] = obs["unique_region"].map(lambda value: parse_patient_id(cohort, value))
        obs["cell_type_sync"] = obs["cell_type_sync"].astype(str)
        frames.append(obs)

    if not frames:
        raise ValueError("No H5AD inputs were provided.")
    return pd.concat(frames, ignore_index=True)


def axon_mask(obs: pd.DataFrame) -> pd.Series:
    mask = obs["cell_type_sync"].astype(str).str.contains("Axon", case=False, na=False)
    if "is_axon" in obs.columns:
        truthy = obs["is_axon"].astype(str).str.lower().isin(["true", "1", "yes"])
        mask = mask | truthy
    return mask


def convex_hull(points: np.ndarray) -> tuple[np.ndarray, float]:
    pts = points[np.isfinite(points).all(axis=1)]
    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 3:
        return np.empty((0, 2)), np.nan
    try:
        hull = ConvexHull(pts)
        vertices = pts[hull.vertices]
        return vertices, float(hull.volume)
    except QhullError:
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        vertices = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)
        return vertices, float(max(xmax - xmin, 0.0) * max(ymax - ymin, 0.0))


def make_hull_grid(vertices: np.ndarray, step: float) -> np.ndarray:
    if vertices.size == 0 or not np.isfinite(step) or step <= 0:
        return np.empty((0, 2))
    xmin, ymin = vertices.min(axis=0)
    xmax, ymax = vertices.max(axis=0)
    xs = np.arange(xmin + step / 2.0, xmax, step)
    ys = np.arange(ymin + step / 2.0, ymax, step)
    if xs.size == 0 or ys.size == 0:
        return np.empty((0, 2))
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    inside = MplPath(vertices).contains_points(grid, radius=1e-9)
    return grid[inside]


def make_hex_window_centers(vertices: np.ndarray, radius: float, spacing: float | None) -> np.ndarray:
    if vertices.size == 0 or radius <= 0:
        return np.empty((0, 2))
    dx = float(spacing) if spacing is not None and spacing > 0 else math.sqrt(3.0) * radius
    dy = 1.5 * radius
    xmin, ymin = vertices.min(axis=0)
    xmax, ymax = vertices.max(axis=0)
    xs_base = np.arange(xmin, xmax + dx, dx)
    ys = np.arange(ymin, ymax + dy, dy)
    centers = []
    for i, y in enumerate(ys):
        xs = xs_base + (0.5 * dx if i % 2 else 0.0)
        centers.extend((x, y) for x in xs)
    if not centers:
        return np.empty((0, 2))
    arr = np.asarray(centers, dtype=float)
    inside = MplPath(vertices).contains_points(arr, radius=1e-9)
    return arr[inside]


def radius_counts(tree: cKDTree | None, query_points: np.ndarray, radius: float, batch_size: int) -> np.ndarray:
    if tree is None or query_points.size == 0:
        return np.zeros(query_points.shape[0], dtype=int)
    counts = np.zeros(query_points.shape[0], dtype=int)
    for start in range(0, query_points.shape[0], batch_size):
        stop = min(start + batch_size, query_points.shape[0])
        batch = query_points[start:stop]
        try:
            counts[start:stop] = tree.query_ball_point(batch, radius, return_length=True)
        except TypeError:
            counts[start:stop] = [len(items) for items in tree.query_ball_point(batch, radius)]
    return counts


def count_points_near_centers(points: np.ndarray, center_tree: cKDTree | None, radius: float) -> int:
    if center_tree is None or points.size == 0:
        return 0
    dist, _ = center_tree.query(points, k=1, distance_upper_bound=radius)
    return int(np.isfinite(dist).sum())


def count_grid_near_centers(grid: np.ndarray, center_tree: cKDTree | None, radius: float) -> int:
    if center_tree is None or grid.size == 0:
        return 0
    dist, _ = center_tree.query(grid, k=1, distance_upper_bound=radius)
    return int(np.isfinite(dist).sum())


def safe_density(count: float, area: float) -> float:
    return float(count / area) if np.isfinite(area) and area > 0 else np.nan


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and den > 0 else np.nan


def enrichment_columns(df: pd.DataFrame, numerator: str, denominator: str, out: str) -> pd.DataFrame:
    result = df.copy()
    result[out] = result[numerator] / result[denominator].replace(0, np.nan)
    result.loc[~np.isfinite(result[out]), out] = np.nan
    result[f"log2_{out}"] = np.log2(result[out].where(result[out] > 0))
    return result


def target_masks(
    obs: pd.DataFrame,
    target_labels: list[str],
    aggregate_label: str,
) -> dict[str, pd.Series]:
    masks = {label: obs["cell_type_sync"].eq(label) for label in target_labels}
    aggregate = pd.Series(False, index=obs.index)
    for mask in masks.values():
        aggregate = aggregate | mask
    return {aggregate_label: aggregate, **masks}


def compute_region(
    keys: tuple[object, ...],
    group: pd.DataFrame,
    target_labels: list[str],
    aggregate_label: str,
    radii: list[float],
    area_grid_step: float,
    window_spacing: float | None,
    min_window_area_fraction: float,
    min_axon_disk_area_fraction: float,
    min_window_total_cells: int,
    min_window_axon_count: int,
    batch_size: int,
    density_scale: float,
    write_window_table: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cohort, dataset, patient_id, unique_region = keys
    points = group[["x", "y"]].to_numpy(dtype=float)
    vertices, tissue_area = convex_hull(points)
    grid = make_hull_grid(vertices, area_grid_step)
    grid_area_unit = tissue_area / grid.shape[0] if grid.shape[0] and np.isfinite(tissue_area) else np.nan

    is_axon = axon_mask(group)
    axon_points = group.loc[is_axon, ["x", "y"]].to_numpy(dtype=float)
    axon_tree = cKDTree(axon_points) if axon_points.size else None
    cell_tree = cKDTree(points) if points.size else None
    masks = target_masks(group, target_labels, aggregate_label)
    region_rows: list[dict[str, object]] = []
    window_summary_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []

    meta = {
        "cohort": cohort,
        "dataset": dataset,
        "patient_id": patient_id,
        "unique_region": unique_region,
        "condition": mode_or_join(group["condition"]) if "condition" in group else "",
        "progression": mode_or_join(group["progression"]) if "progression" in group else "",
        "n_cells": int(group.shape[0]),
        "n_axons": int(is_axon.sum()),
        "tissue_area_proxy": tissue_area,
        "area_grid_step_um": area_grid_step,
        "area_grid_points_in_tissue": int(grid.shape[0]),
    }

    for radius in radii:
        full_circle_area = math.pi * radius * radius
        grid_tree = cKDTree(grid) if grid.size else None
        if axon_points.size and grid_tree is not None and np.isfinite(grid_area_unit):
            axon_disk_area_counts = radius_counts(grid_tree, axon_points, radius, batch_size)
            axon_disk_area_fractions = axon_disk_area_counts.astype(float) * grid_area_unit / full_circle_area
            axon_disk_area_fractions = np.minimum(axon_disk_area_fractions, 1.0)
            interior_axon_mask = axon_disk_area_fractions >= min_axon_disk_area_fraction
        else:
            axon_disk_area_counts = np.zeros(axon_points.shape[0], dtype=int)
            axon_disk_area_fractions = np.full(axon_points.shape[0], np.nan)
            interior_axon_mask = np.zeros(axon_points.shape[0], dtype=bool)
        interior_axon_points = axon_points[interior_axon_mask]
        interior_axon_tree = cKDTree(interior_axon_points) if interior_axon_points.size else None

        axon_neighborhood_grid_points = count_grid_near_centers(grid, interior_axon_tree, radius)
        axon_neighborhood_area = (
            axon_neighborhood_grid_points * grid_area_unit
            if np.isfinite(grid_area_unit)
            else np.nan
        )
        axon_neighborhood_area_fraction = safe_ratio(axon_neighborhood_area, tissue_area)

        centers = make_hex_window_centers(vertices, radius, window_spacing)
        sampled_area_counts = radius_counts(grid_tree, centers, radius, batch_size)
        sampled_areas = (
            sampled_area_counts.astype(float) * grid_area_unit
            if np.isfinite(grid_area_unit)
            else np.full(centers.shape[0], np.nan)
        )
        sampled_areas = np.minimum(sampled_areas, full_circle_area)
        sampled_area_fractions = sampled_areas / full_circle_area
        total_window_counts = radius_counts(cell_tree, centers, radius, batch_size)
        area_valid_windows = np.isfinite(sampled_area_fractions) & (
            sampled_area_fractions >= min_window_area_fraction
        )
        nonempty_windows = total_window_counts >= min_window_total_cells
        valid_windows = area_valid_windows & nonempty_windows
        centers_valid = centers[valid_windows]
        sampled_areas_valid = sampled_areas[valid_windows]
        sampled_area_fractions_valid = sampled_area_fractions[valid_windows]
        total_window_counts_valid = total_window_counts[valid_windows]
        axon_window_counts = radius_counts(axon_tree, centers_valid, radius, batch_size)
        axon_associated = axon_window_counts >= min_window_axon_count

        for target, mask in masks.items():
            immune_points = group.loc[mask, ["x", "y"]].to_numpy(dtype=float)
            immune_tree = cKDTree(immune_points) if immune_points.size else None
            n_immune_total = int(mask.sum())
            n_near_axon = count_points_near_centers(immune_points, interior_axon_tree, radius)
            whole_density = safe_density(n_immune_total, tissue_area)
            local_density = safe_density(n_near_axon, axon_neighborhood_area)
            cooccurrence_probability = safe_ratio(n_near_axon, n_immune_total)
            cooccurrence_enrichment = safe_ratio(
                cooccurrence_probability,
                axon_neighborhood_area_fraction,
            )
            density_enrichment = safe_ratio(local_density, whole_density)

            region_rows.append(
                {
                    **meta,
                    "radius_um": radius,
                    "target": target,
                    "min_axon_disk_area_fraction": min_axon_disk_area_fraction,
                    "n_axons_used_for_disk_union": int(interior_axon_mask.sum()),
                    "n_axons_excluded_by_edge_filter": int((~interior_axon_mask).sum()),
                    "median_axon_disk_area_fraction": float(np.nanmedian(axon_disk_area_fractions))
                    if axon_disk_area_fractions.size
                    else np.nan,
                    "n_immune_total": n_immune_total,
                    "n_immune_near_axon": n_near_axon,
                    "axon_neighborhood_area_proxy": axon_neighborhood_area,
                    "axon_neighborhood_area_fraction": axon_neighborhood_area_fraction,
                    "area_grid_points_near_axon": axon_neighborhood_grid_points,
                    "whole_tissue_density": whole_density,
                    f"whole_tissue_density_per_{int(density_scale):g}_area": whole_density
                    * density_scale
                    if np.isfinite(whole_density)
                    else np.nan,
                    "axon_local_density": local_density,
                    f"axon_local_density_per_{int(density_scale):g}_area": local_density
                    * density_scale
                    if np.isfinite(local_density)
                    else np.nan,
                    "density_enrichment": density_enrichment,
                    "log2_density_enrichment": np.log2(density_enrichment)
                    if density_enrichment > 0
                    else np.nan,
                    "cooccurrence_probability": cooccurrence_probability,
                    "cooccurrence_enrichment": cooccurrence_enrichment,
                    "log2_cooccurrence_enrichment": np.log2(cooccurrence_enrichment)
                    if cooccurrence_enrichment > 0
                    else np.nan,
                }
            )

            immune_window_counts = radius_counts(immune_tree, centers_valid, radius, batch_size)
            assoc_count = int(immune_window_counts[axon_associated].sum())
            bg_count = int(immune_window_counts[~axon_associated].sum())
            assoc_area = float(np.nansum(sampled_areas_valid[axon_associated]))
            bg_area = float(np.nansum(sampled_areas_valid[~axon_associated]))
            assoc_density = safe_density(assoc_count, assoc_area)
            bg_density = safe_density(bg_count, bg_area)
            window_enrichment = safe_ratio(assoc_density, bg_density)

            window_summary_rows.append(
                {
                    **meta,
                    "radius_um": radius,
                    "target": target,
                    "min_window_area_fraction": min_window_area_fraction,
                    "min_window_total_cells": int(min_window_total_cells),
                    "n_candidate_windows": int(centers.shape[0]),
                    "n_windows_excluded_by_edge_filter": int((~area_valid_windows).sum()),
                    "n_windows_excluded_by_empty_filter": int((area_valid_windows & ~nonempty_windows).sum()),
                    "n_windows": int(centers_valid.shape[0]),
                    "n_axon_associated_windows": int(axon_associated.sum()),
                    "n_background_windows": int((~axon_associated).sum()),
                    "median_window_area_fraction": float(np.nanmedian(sampled_area_fractions_valid))
                    if sampled_area_fractions_valid.size
                    else np.nan,
                    "min_valid_window_area_fraction": float(np.nanmin(sampled_area_fractions_valid))
                    if sampled_area_fractions_valid.size
                    else np.nan,
                    "total_observed_cells_in_valid_windows": int(total_window_counts_valid.sum()),
                    "axon_associated_window_area_proxy": assoc_area,
                    "background_window_area_proxy": bg_area,
                    "axon_associated_immune_count": assoc_count,
                    "background_immune_count": bg_count,
                    "axon_associated_window_density": assoc_density,
                    f"axon_associated_window_density_per_{int(density_scale):g}_area": assoc_density
                    * density_scale
                    if np.isfinite(assoc_density)
                    else np.nan,
                    "background_window_density": bg_density,
                    f"background_window_density_per_{int(density_scale):g}_area": bg_density
                    * density_scale
                    if np.isfinite(bg_density)
                    else np.nan,
                    "window_density_enrichment": window_enrichment,
                    "log2_window_density_enrichment": np.log2(window_enrichment)
                    if window_enrichment > 0
                    else np.nan,
                }
            )

            if write_window_table and target == aggregate_label:
                for i in range(centers_valid.shape[0]):
                    window_area = sampled_areas_valid[i]
                    density = safe_density(int(immune_window_counts[i]), window_area)
                    window_rows.append(
                        {
                            "cohort": cohort,
                            "dataset": dataset,
                            "patient_id": patient_id,
                            "unique_region": unique_region,
                            "condition": meta["condition"],
                            "progression": meta["progression"],
                            "radius_um": radius,
                            "window_id": i,
                            "x": centers_valid[i, 0],
                            "y": centers_valid[i, 1],
                            "window_area_proxy": window_area,
                            "window_area_fraction": sampled_area_fractions_valid[i],
                            "total_cell_count": int(total_window_counts_valid[i]),
                            "axon_count": int(axon_window_counts[i]),
                            "axon_associated": bool(axon_associated[i]),
                            f"{aggregate_label}_count": int(immune_window_counts[i]),
                            f"{aggregate_label}_density": density,
                            f"{aggregate_label}_density_per_{int(density_scale):g}_area": density
                            * density_scale
                            if np.isfinite(density)
                            else np.nan,
                        }
                    )

    return region_rows, window_summary_rows, window_rows


def aggregate_region_to_patient(region: pd.DataFrame, density_scale: float) -> pd.DataFrame:
    if region.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["cohort", "dataset", "patient_id", "radius_um", "target"]
    for keys, group in region.groupby(group_cols, observed=True, sort=False):
        tissue_area = group["tissue_area_proxy"].sum()
        axon_area = group["axon_neighborhood_area_proxy"].sum()
        n_total = int(group["n_immune_total"].sum())
        n_near = int(group["n_immune_near_axon"].sum())
        whole_density = safe_density(n_total, tissue_area)
        local_density = safe_density(n_near, axon_area)
        area_fraction = safe_ratio(axon_area, tissue_area)
        co_prob = safe_ratio(n_near, n_total)
        density_enrichment = safe_ratio(local_density, whole_density)
        co_enrichment = safe_ratio(co_prob, area_fraction)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "condition": mode_or_join(group["condition"]),
                "progression": mode_or_join(group["progression"]),
                "n_regions": int(group["unique_region"].nunique()),
                "n_cells": int(group["n_cells"].sum()),
                "n_axons": int(group["n_axons"].sum()),
                "n_axons_used_for_disk_union": int(group["n_axons_used_for_disk_union"].sum()),
                "n_axons_excluded_by_edge_filter": int(group["n_axons_excluded_by_edge_filter"].sum()),
                "tissue_area_proxy": tissue_area,
                "axon_neighborhood_area_proxy": axon_area,
                "axon_neighborhood_area_fraction": area_fraction,
                "n_immune_total": n_total,
                "n_immune_near_axon": n_near,
                "whole_tissue_density": whole_density,
                f"whole_tissue_density_per_{int(density_scale):g}_area": whole_density
                * density_scale
                if np.isfinite(whole_density)
                else np.nan,
                "axon_local_density": local_density,
                f"axon_local_density_per_{int(density_scale):g}_area": local_density
                * density_scale
                if np.isfinite(local_density)
                else np.nan,
                "density_enrichment": density_enrichment,
                "log2_density_enrichment": np.log2(density_enrichment)
                if density_enrichment > 0
                else np.nan,
                "cooccurrence_probability": co_prob,
                "cooccurrence_enrichment": co_enrichment,
                "log2_cooccurrence_enrichment": np.log2(co_enrichment)
                if co_enrichment > 0
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_window_to_patient(window_summary: pd.DataFrame, density_scale: float) -> pd.DataFrame:
    if window_summary.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["cohort", "dataset", "patient_id", "radius_um", "target"]
    for keys, group in window_summary.groupby(group_cols, observed=True, sort=False):
        assoc_count = int(group["axon_associated_immune_count"].sum())
        bg_count = int(group["background_immune_count"].sum())
        assoc_area = group["axon_associated_window_area_proxy"].sum()
        bg_area = group["background_window_area_proxy"].sum()
        assoc_density = safe_density(assoc_count, assoc_area)
        bg_density = safe_density(bg_count, bg_area)
        enrichment = safe_ratio(assoc_density, bg_density)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "condition": mode_or_join(group["condition"]),
                "progression": mode_or_join(group["progression"]),
                "n_regions": int(group["unique_region"].nunique()),
                "n_candidate_windows": int(group["n_candidate_windows"].sum()),
                "n_windows_excluded_by_edge_filter": int(group["n_windows_excluded_by_edge_filter"].sum()),
                "n_windows_excluded_by_empty_filter": int(group["n_windows_excluded_by_empty_filter"].sum()),
                "n_windows": int(group["n_windows"].sum()),
                "n_axon_associated_windows": int(group["n_axon_associated_windows"].sum()),
                "n_background_windows": int(group["n_background_windows"].sum()),
                "axon_associated_window_area_proxy": assoc_area,
                "background_window_area_proxy": bg_area,
                "axon_associated_immune_count": assoc_count,
                "background_immune_count": bg_count,
                "axon_associated_window_density": assoc_density,
                f"axon_associated_window_density_per_{int(density_scale):g}_area": assoc_density
                * density_scale
                if np.isfinite(assoc_density)
                else np.nan,
                "background_window_density": bg_density,
                f"background_window_density_per_{int(density_scale):g}_area": bg_density
                * density_scale
                if np.isfinite(bg_density)
                else np.nan,
                "window_density_enrichment": enrichment,
                "log2_window_density_enrichment": np.log2(enrichment)
                if enrichment > 0
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_target_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "n_immune_total": "n_target_total",
        "n_immune_near_axon": "n_target_near_axon",
        "axon_associated_immune_count": "axon_associated_target_count",
        "background_immune_count": "background_target_count",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    return out


def one_sample_tests(
    df: pd.DataFrame,
    metric: str,
    log_metric: str,
    unit_label: str,
    group_cols: list[str],
) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for keys, group in df.groupby(group_cols, observed=True, sort=False):
        clean = pd.to_numeric(group[log_metric], errors="coerce").dropna()
        enrich = pd.to_numeric(group.loc[clean.index, metric], errors="coerce")
        nonzero = clean.loc[clean != 0]
        n_positive = int((clean > 0).sum())
        n_negative = int((clean < 0).sum())
        n_tie = int((clean == 0).sum())
        if len(clean) >= 2 and clean.std(ddof=1) > 0:
            try:
                t_result = stats.ttest_1samp(clean, 0.0, alternative="greater")
                t_stat = float(t_result.statistic)
                t_p = float(t_result.pvalue)
            except TypeError:
                t_two = stats.ttest_1samp(clean, 0.0)
                t_stat = float(t_two.statistic)
                t_p = float(t_two.pvalue / 2.0 if t_stat > 0 else 1.0 - t_two.pvalue / 2.0)
        else:
            t_stat = np.nan
            t_p = np.nan
        if len(nonzero) > 0:
            try:
                w_result = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="greater")
                w_stat = float(w_result.statistic)
                w_p = float(w_result.pvalue)
            except ValueError:
                w_stat = np.nan
                w_p = np.nan
            sign_p = float(stats.binomtest(n_positive, n_positive + n_negative, 0.5, alternative="greater").pvalue)
        else:
            w_stat = np.nan
            w_p = np.nan
            sign_p = np.nan
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "unit": unit_label,
                "metric": metric,
                "n_units": int(len(clean)),
                "n_enriched": n_positive,
                "n_depleted": n_negative,
                "n_tie": n_tie,
                "median_enrichment": float(enrich.median()) if len(enrich) else np.nan,
                "geometric_mean_enrichment": float(2 ** clean.mean()) if len(clean) else np.nan,
                "median_log2_enrichment": float(clean.median()) if len(clean) else np.nan,
                "mean_log2_enrichment": float(clean.mean()) if len(clean) else np.nan,
                "ttest_log2_gt0_statistic": t_stat,
                "ttest_log2_gt0_p": t_p,
                "wilcoxon_log2_gt0_statistic": w_stat,
                "wilcoxon_log2_gt0_p": w_p,
                "sign_test_gt_half_p": sign_p,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        for p_col in ["ttest_log2_gt0_p", "wilcoxon_log2_gt0_p", "sign_test_gt_half_p"]:
            q_col = p_col.replace("_p", "_q_bh_by_unit_metric_cohort")
            out[q_col] = out.groupby(["unit", "metric", "cohort"], group_keys=False)[p_col].apply(bh_qvalues)
    return out


def plot_enrichment_points(
    df: pd.DataFrame,
    metric: str,
    aggregate_label: str,
    out: Path,
    title: str,
) -> Path | None:
    data = df.loc[df["target"].eq(aggregate_label)].copy()
    data = data.loc[np.isfinite(data[metric]) & (data[metric] > 0)]
    if data.empty:
        return None
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sns.boxplot(
        data=data,
        x="cohort",
        y=metric,
        hue="condition",
        dodge=True,
        fliersize=0,
        linewidth=1.2,
        ax=ax,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Enrichment ratio")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([display_label(label.get_text()) for label in ax.get_xticklabels()])
    ax.set_title("")
    ax.legend(frameon=False, title="")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_target_heatmap(
    df: pd.DataFrame,
    metric: str,
    aggregate_label: str,
    out: Path,
    title: str,
) -> Path | None:
    data = df.loc[np.isfinite(df[metric])].copy()
    if data.empty:
        return None
    pivot = data.pivot_table(index="target", columns="cohort", values=metric, aggfunc="median")
    if aggregate_label in pivot.index:
        order = [aggregate_label] + [idx for idx in pivot.index if idx != aggregate_label]
        pivot = pivot.loc[order]
    vmax = max(0.5, np.nanpercentile(np.abs(pivot.to_numpy(dtype=float)), 95))
    pivot = pivot.rename(index=display_label, columns=display_label)
    fig, ax = plt.subplots(figsize=(4.6, max(3.2, 0.35 * len(pivot))))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Median log2 enrichment"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def window_context_long(window_region: pd.DataFrame, density_scale: float) -> pd.DataFrame:
    if window_region.empty:
        return pd.DataFrame()
    rows = []
    density_suffix = f"per_{int(density_scale):g}_area"
    for _, values in window_region.iterrows():
        base = {
            "cohort": values["cohort"],
            "dataset": values["dataset"],
            "patient_id": values["patient_id"],
            "unique_region": values["unique_region"],
            "condition": values["condition"],
            "progression": values["progression"],
            "radius_um": values["radius_um"],
            "target": values["target"],
        }
        rows.append(
            {
                **base,
                "window_context": "Background windows",
                "n_windows": int(values["n_background_windows"]),
                "n_target_cells": int(values.get("background_target_count", values["background_immune_count"])),
                "window_area_proxy": values["background_window_area_proxy"],
                "target_density": values["background_window_density"],
                f"target_density_{density_suffix}": values[f"background_window_density_{density_suffix}"],
            }
        )
        rows.append(
            {
                **base,
                "window_context": "Axon-associated windows",
                "n_windows": int(values["n_axon_associated_windows"]),
                "n_target_cells": int(values.get("axon_associated_target_count", values["axon_associated_immune_count"])),
                "window_area_proxy": values["axon_associated_window_area_proxy"],
                "target_density": values["axon_associated_window_density"],
                f"target_density_{density_suffix}": values[f"axon_associated_window_density_{density_suffix}"],
            }
        )
    return pd.DataFrame(rows)


def paired_window_context_tests(window_region: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if window_region.empty:
        return pd.DataFrame()
    group_cols = ["cohort", "radius_um", "target"]
    for keys, group in window_region.groupby(group_cols, observed=True, sort=False):
        clean = group.loc[
            np.isfinite(group["log2_window_density_enrichment"])
            & group["n_axon_associated_windows"].gt(0)
            & group["n_background_windows"].gt(0)
        ].copy()
        values = clean["log2_window_density_enrichment"].astype(float)
        nonzero = values.loc[values != 0]
        n_positive = int((values > 0).sum())
        n_negative = int((values < 0).sum())
        n_tie = int((values == 0).sum())
        if len(nonzero):
            try:
                two = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
                greater = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="greater")
                less = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="less")
                two_stat = float(two.statistic)
                two_p = float(two.pvalue)
                greater_p = float(greater.pvalue)
                less_p = float(less.pvalue)
            except ValueError:
                two_stat = np.nan
                two_p = np.nan
                greater_p = np.nan
                less_p = np.nan
        else:
            two_stat = np.nan
            two_p = np.nan
            greater_p = np.nan
            less_p = np.nan
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_regions": int(len(values)),
                "n_axon_gt_background": n_positive,
                "n_axon_lt_background": n_negative,
                "n_tie": n_tie,
                "median_background_window_density": float(clean["background_window_density"].median())
                if len(clean)
                else np.nan,
                "median_axon_associated_window_density": float(clean["axon_associated_window_density"].median())
                if len(clean)
                else np.nan,
                "median_window_density_enrichment": float(clean["window_density_enrichment"].median())
                if len(clean)
                else np.nan,
                "median_log2_window_density_enrichment": float(values.median()) if len(values) else np.nan,
                "wilcoxon_log2_two_sided_statistic": two_stat,
                "wilcoxon_log2_two_sided_p": two_p,
                "wilcoxon_log2_axon_gt_background_p": greater_p,
                "wilcoxon_log2_axon_lt_background_p": less_p,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        for p_col in [
            "wilcoxon_log2_two_sided_p",
            "wilcoxon_log2_axon_gt_background_p",
            "wilcoxon_log2_axon_lt_background_p",
        ]:
            q_col = p_col.replace("_p", "_q_bh_by_cohort")
            out[q_col] = out.groupby(["cohort"], group_keys=False)[p_col].apply(bh_qvalues)
    return out


def median_iqr(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return pd.Series(
        {
            "n_regions": int(clean.shape[0]),
            "median": float(clean.median()) if not clean.empty else np.nan,
            "q1": float(clean.quantile(0.25)) if not clean.empty else np.nan,
            "q3": float(clean.quantile(0.75)) if not clean.empty else np.nan,
        }
    )


def p_value_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_window_context_density(
    window_region: pd.DataFrame,
    window_tests: pd.DataFrame,
    aggregate_label: str,
    density_scale: float,
    out_png: Path,
    out_pdf: Path,
    title: str,
    aggregate_only: bool = False,
    combine_cohorts: bool = False,
    combined_cohort_label: str = COMBINED_COHORT_LABEL,
    font_multiplier: float = 1.0,
    orientation: str = "horizontal",
) -> tuple[Path, Path] | None:
    plot_window_region = window_region.copy()
    plot_window_tests = window_tests.copy()
    if combine_cohorts and not plot_window_region.empty:
        plot_window_region["cohort"] = combined_cohort_label
        plot_window_tests = paired_window_context_tests(plot_window_region)

    long = window_context_long(plot_window_region, density_scale)
    if long.empty:
        return None
    if aggregate_only:
        long = long.loc[long["target"].eq(aggregate_label)].copy()
    if long.empty:
        return None

    density_col = f"target_density_per_{int(density_scale):g}_area"
    long = long.loc[np.isfinite(long[density_col])].copy()
    if long.empty:
        return None

    if aggregate_only:
        target_order = [aggregate_label]
    else:
        target_order = [label for label in long["target"].dropna().unique() if label != aggregate_label]
    if not target_order:
        return None
    context_order = ["Background windows", "Axon-associated windows"]
    colors = {"Background windows": "#b8b8b8", "Axon-associated windows": "#2F6FA3"}
    cohorts = list(pd.unique(long["cohort"].dropna().astype(str)))
    if not cohorts:
        return None

    summary = (
        long.groupby(["cohort", "target", "window_context"], observed=True)[density_col]
        .apply(median_iqr)
        .unstack()
        .reset_index()
    )

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
        }
    )
    sns.set_theme(style="white", context="talk")
    font_multiplier = float(font_multiplier)
    if not np.isfinite(font_multiplier) or font_multiplier <= 0:
        font_multiplier = 1.0
    axis_label_size = 24 * font_multiplier
    tick_label_size = 20 * font_multiplier
    legend_size = 22 * font_multiplier
    annotation_size = (20 if aggregate_only else 24) * font_multiplier
    orientation = clean_str(orientation).lower()

    if orientation == "vertical":
        if len(cohorts) != 1:
            raise ValueError("Vertical window-density plotting expects one cohort.")
        cohort = cohorts[0]
        cohort_summary = summary.loc[summary["cohort"].astype(str).eq(cohort)].copy()
        cohort_points = long.loc[long["cohort"].astype(str).eq(cohort)].copy()
        cohort_tests = (
            plot_window_tests.loc[plot_window_tests["cohort"].astype(str).eq(cohort)].set_index("target")
            if not plot_window_tests.empty
            else pd.DataFrame()
        )

        if len(target_order) > 1:
            square_size = max(17.5, 2.35 * len(target_order) * min(font_multiplier, 2.2))
            fig, ax = plt.subplots(figsize=(square_size, square_size))
            bar_width = 0.34
            offsets = np.array([-bar_width / 1.8, bar_width / 1.8])
            x = np.arange(len(target_order))

            for offset, context in zip(offsets, context_order):
                context_summary = (
                    cohort_summary.loc[cohort_summary["window_context"].eq(context)]
                    .set_index("target")
                    .reindex(target_order)
                )
                values = context_summary["median"].to_numpy(dtype=float)
                ax.bar(
                    x + offset,
                    values,
                    width=bar_width,
                    color=colors[context],
                    edgecolor="black",
                    linewidth=1.2,
                    label=context,
                    zorder=2,
                )
                q1 = context_summary["q1"].to_numpy(dtype=float)
                q3 = context_summary["q3"].to_numpy(dtype=float)
                finite = np.isfinite(values)
                if finite.any():
                    lower = np.clip(values[finite] - q1[finite], 0, None)
                    upper = np.clip(q3[finite] - values[finite], 0, None)
                    ax.errorbar(
                        (x + offset)[finite],
                        values[finite],
                        yerr=np.vstack([lower, upper]),
                        fmt="none",
                        ecolor="black",
                        elinewidth=1.8,
                        capsize=5,
                        capthick=1.8,
                        zorder=4,
                    )

            visible_summary = cohort_summary.loc[cohort_summary["target"].isin(target_order)]
            summary_values = visible_summary[["median", "q3"]].to_numpy(dtype=float).ravel()
            summary_values = summary_values[np.isfinite(summary_values)]
            y_max = float(summary_values.max()) if summary_values.size else np.nan
            y_step = 0.06 * y_max if np.isfinite(y_max) and y_max > 0 else 1.0
            annotation_top = y_max if np.isfinite(y_max) else np.nan
            for i, target in enumerate(target_order):
                if cohort_tests.empty or target not in cohort_tests.index:
                    continue
                p_lookup = cohort_tests.loc[target, "wilcoxon_log2_two_sided_p"]
                p_value = float(p_lookup.iloc[0] if isinstance(p_lookup, pd.Series) else p_lookup)
                target_summary = cohort_summary.loc[cohort_summary["target"].eq(target), ["median", "q3"]]
                target_values = target_summary.to_numpy(dtype=float).ravel()
                target_values = target_values[np.isfinite(target_values)]
                if target_values.size == 0:
                    continue
                y_text = float(target_values.max()) + y_step
                ax.text(i, y_text, p_value_stars(p_value), ha="center", va="bottom", fontsize=annotation_size)
                annotation_top = max(annotation_top, y_text)
            if np.isfinite(annotation_top):
                ax.set_ylim(bottom=0, top=annotation_top + 1.8 * y_step)

            ax.set_xticks(x)
            ax.set_xticklabels([display_label(target) for target in target_order], rotation=35, ha="right")
            ax.set_title("")
            ax.grid(axis="y", color="#d8d8d8", linewidth=0.9)
            ax.set_xlabel("")
            ax.set_ylabel("Window density", fontsize=axis_label_size, labelpad=14)
            ax.tick_params(axis="both", labelsize=tick_label_size, length=6, width=1.4)

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                [display_label(label) for label in labels],
                frameon=False,
                loc="upper center",
                ncol=2,
                bbox_to_anchor=(0.5, 0.17),
                prop={"size": legend_size},
            )
            sns.despine(fig=fig)
            bottom_margin = 0.44 if font_multiplier > 2.5 else 0.36
            left_margin = 0.17 if font_multiplier > 2.5 else 0.12
            fig.subplots_adjust(left=left_margin, right=0.985, bottom=bottom_margin, top=0.965)
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=300)
            fig.savefig(out_pdf)
            plt.close(fig)
            return out_png, out_pdf

        target = target_order[0]

        fig, ax = plt.subplots(figsize=(6.4, 7.8))
        bar_width = 0.58
        x = np.arange(len(context_order))
        context_summary = cohort_summary.set_index("window_context").reindex(context_order)
        values = context_summary["median"].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            width=bar_width,
            color=[colors[context] for context in context_order],
            edgecolor="black",
            linewidth=1.2,
            zorder=2,
        )
        q1 = context_summary["q1"].to_numpy(dtype=float)
        q3 = context_summary["q3"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            lower = np.clip(values[finite] - q1[finite], 0, None)
            upper = np.clip(q3[finite] - values[finite], 0, None)
            ax.errorbar(
                x[finite],
                values[finite],
                yerr=np.vstack([lower, upper]),
                fmt="none",
                ecolor="black",
                elinewidth=1.8,
                capsize=5,
                capthick=1.8,
                zorder=4,
            )
        summary_values = np.concatenate([values, q3])
        summary_values = summary_values[np.isfinite(summary_values)]
        y_max = float(summary_values.max()) if summary_values.size else np.nan
        y_step = 0.06 * y_max if np.isfinite(y_max) and y_max > 0 else 1.0
        annotation_top = y_max if np.isfinite(y_max) else np.nan
        if not cohort_tests.empty and target in cohort_tests.index:
            p_lookup = cohort_tests.loc[target, "wilcoxon_log2_two_sided_p"]
            p_value = float(p_lookup.iloc[0] if isinstance(p_lookup, pd.Series) else p_lookup)
            y_line = float(y_max) + y_step
            tick = 0.18 * y_step
            ax.plot([x[0], x[0], x[1], x[1]], [y_line - tick, y_line, y_line, y_line - tick], color="black", linewidth=1.6)
            ax.text(
                np.mean(x),
                y_line + 0.25 * y_step,
                p_value_stars(p_value),
                ha="center",
                va="bottom",
                fontsize=annotation_size,
            )
            annotation_top = y_line + 1.4 * y_step
        if np.isfinite(annotation_top):
            ax.set_ylim(bottom=0, top=annotation_top)

        ax.set_xticks(x)
        ax.set_xticklabels(["Background\nwindows", "Axon-associated\nwindows"])
        ax.set_title("")
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.9)
        ax.set_xlabel("")
        ax.set_ylabel("Window density", fontsize=axis_label_size, labelpad=14)
        ax.tick_params(axis="both", labelsize=tick_label_size, length=6, width=1.4)
        sns.despine(fig=fig)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        return out_png, out_pdf

    if combine_cohorts:
        fig_width = 9.5 if aggregate_only else 12.5 * min(font_multiplier, 1.45)
        fig_height = 4.8 if aggregate_only else max(7.2, 0.78 * len(target_order) + 2.2) * min(font_multiplier, 1.45)
    else:
        fig_width = max(9.5, 6.2 * len(cohorts))
        fig_height = 4.8 if aggregate_only else max(7.2, 0.78 * len(target_order) + 2.2)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(fig_width, fig_height), squeeze=False, sharex=True)
    bar_height = 0.34
    offsets = np.array([bar_height / 1.8, -bar_height / 1.8])
    y = np.arange(len(target_order))

    for ax, cohort in zip(axes[0], cohorts):
        cohort_summary = summary.loc[summary["cohort"].astype(str).eq(cohort)].copy()
        cohort_points = long.loc[long["cohort"].astype(str).eq(cohort)].copy()
        cohort_tests = (
            plot_window_tests.loc[plot_window_tests["cohort"].astype(str).eq(cohort)].set_index("target")
            if not plot_window_tests.empty
            else pd.DataFrame()
        )
        annotation_top = float(cohort_points[density_col].max()) if not cohort_points.empty else np.nan
        for offset, context in zip(offsets, context_order):
            context_summary = (
                cohort_summary.loc[cohort_summary["window_context"].eq(context)]
                .set_index("target")
                .reindex(target_order)
            )
            values = context_summary["median"].to_numpy(dtype=float)
            ax.barh(
                y + offset,
                values,
                height=bar_height,
                color=colors[context],
                edgecolor="black",
                linewidth=1.2,
                label=context,
                zorder=2,
            )
            q1 = context_summary["q1"].to_numpy(dtype=float)
            q3 = context_summary["q3"].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if finite.any():
                lower = np.clip(values[finite] - q1[finite], 0, None)
                upper = np.clip(q3[finite] - values[finite], 0, None)
                ax.errorbar(
                    values[finite],
                    (y + offset)[finite],
                    xerr=np.vstack([lower, upper]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.8,
                    capsize=5,
                    capthick=1.8,
                    zorder=4,
                )
        visible_summary = cohort_summary.loc[cohort_summary["target"].isin(target_order)]
        summary_values = visible_summary[["median", "q3"]].to_numpy(dtype=float).ravel()
        summary_values = summary_values[np.isfinite(summary_values)]
        x_max = float(summary_values.max()) if summary_values.size else np.nan
        x_step = 0.055 * x_max if np.isfinite(x_max) and x_max > 0 else 1.0
        annotation_right = float(x_max) if np.isfinite(x_max) else np.nan
        for i, target in enumerate(target_order):
            if cohort_tests.empty or target not in cohort_tests.index:
                continue
            p_lookup = cohort_tests.loc[target, "wilcoxon_log2_two_sided_p"]
            p_value = float(p_lookup.iloc[0] if isinstance(p_lookup, pd.Series) else p_lookup)
            target_summary = cohort_summary.loc[cohort_summary["target"].eq(target), ["median", "q3"]]
            target_values = target_summary.to_numpy(dtype=float).ravel()
            target_values = target_values[np.isfinite(target_values)]
            if target_values.size == 0:
                continue
            x_text = float(target_values.max()) + x_step
            label = p_value_stars(p_value)
            ax.text(x_text, y[i], label, ha="left", va="center", fontsize=annotation_size)
            annotation_right = max(annotation_right, x_text)
        if np.isfinite(annotation_right):
            ax.set_xlim(left=0, right=annotation_right + 1.8 * x_step)
        ax.set_yticks(y)
        ax.set_yticklabels([display_label(target) for target in target_order])
        ax.invert_yaxis()
        ax.set_title("")
        ax.grid(axis="x", color="#d8d8d8", linewidth=0.9)
        ax.set_xlabel("Window density", fontsize=axis_label_size, labelpad=14)
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=tick_label_size, length=6, width=1.4)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        [display_label(label) for label in labels],
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.35),
        prop={"size": legend_size},
    )
    sns.despine(fig=fig)
    bottom_margin = 0.37 if font_multiplier > 1.2 else 0.26
    fig.tight_layout(rect=[0, bottom_margin, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def write_report(
    outdir: Path,
    h5ads: list[Path],
    region: pd.DataFrame,
    patient: pd.DataFrame,
    window_region: pd.DataFrame,
    tests: pd.DataFrame,
    figures: list[Path],
    args: argparse.Namespace,
) -> Path:
    aggregate_label = args.aggregate_label
    target_kind = args.target_kind
    file_prefix = safe_name(args.file_prefix)
    aggregate_region = region.loc[region["target"].eq(aggregate_label)].copy()
    aggregate_patient = patient.loc[patient["target"].eq(aggregate_label)].copy()
    summary = (
        aggregate_region.groupby(["cohort", "radius_um"], observed=True)
        .agg(
            n_regions=("unique_region", "nunique"),
            total_axons=("n_axons", "sum"),
            total_axons_used_for_disk_union=("n_axons_used_for_disk_union", "sum"),
            total_axons_excluded_by_edge_filter=("n_axons_excluded_by_edge_filter", "sum"),
            median_density_enrichment=("density_enrichment", "median"),
            median_cooccurrence_enrichment=("cooccurrence_enrichment", "median"),
            median_area_fraction=("axon_neighborhood_area_fraction", "median"),
            median_cooccurrence_probability=("cooccurrence_probability", "median"),
        )
        .reset_index()
        if not aggregate_region.empty
        else pd.DataFrame()
    )
    patient_cols = [
        "cohort",
        "patient_id",
        "condition",
        "progression",
        "radius_um",
        "n_regions",
        "n_axons",
        "n_axons_used_for_disk_union",
        "n_axons_excluded_by_edge_filter",
        "n_target_total",
        "n_target_near_axon",
        "density_enrichment",
        "cooccurrence_enrichment",
    ]
    patient_rank = (
        aggregate_patient.sort_values(["cohort", "density_enrichment"], ascending=[True, False])
        .loc[:, [column for column in patient_cols if column in aggregate_patient.columns]]
        if not aggregate_patient.empty
        else pd.DataFrame()
    )
    top_tests = (
        tests.loc[tests["target"].eq(aggregate_label)]
        .sort_values(["unit", "metric", "cohort"])
        .loc[
            :,
            [
                "unit",
                "metric",
                "cohort",
                "radius_um",
                "n_units",
                "median_enrichment",
                "geometric_mean_enrichment",
                "wilcoxon_log2_gt0_p",
                "sign_test_gt_half_p",
            ],
        ]
        if not tests.empty and "target" in tests.columns
        else pd.DataFrame()
    )
    window_summary = (
        window_region.loc[window_region["target"].eq(aggregate_label)]
        .groupby(["cohort", "radius_um"], observed=True)
        .agg(
            n_regions=("unique_region", "nunique"),
            total_candidate_windows=("n_candidate_windows", "sum"),
            total_windows_excluded_by_edge_filter=("n_windows_excluded_by_edge_filter", "sum"),
            total_windows_excluded_by_empty_filter=("n_windows_excluded_by_empty_filter", "sum"),
            total_windows_used=("n_windows", "sum"),
            median_window_density_enrichment=("window_density_enrichment", "median"),
            median_axon_windows=("n_axon_associated_windows", "median"),
            median_background_windows=("n_background_windows", "median"),
        )
        .reset_index()
        if not window_region.empty
        else pd.DataFrame()
    )

    lines = [
        f"# {args.report_title}",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in h5ads)
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Axons are observations whose `cell_type_sync` contains `Axon`, plus truthy `is_axon` where present.",
            f"- {target_kind.capitalize()} targets use the configured labels, plus an `{aggregate_label}` aggregate.",
            "- Whole-tissue area is the convex hull of all cell coordinates in each ROI, matching the existing repository density-proxy convention.",
            "- Axon-neighborhood area is the union of radius-r axon disks clipped to that convex-hull proxy, estimated on a deterministic grid.",
            "- Axon-centered disks are included only when their estimated tissue-covered area is at least the configured fraction of a full circle.",
            f"- Co-occurrence enrichment is `fraction of {target_kind} cells within r of any axon / fraction of tissue area within r of any axon`.",
            f"- Local-window enrichment uses hex-grid radius-r windows, drops edge windows and windows with too few total observed cells, then compares area-weighted {target_kind} density in axon-associated windows against background windows within each ROI.",
            f"- Windows with zero target {target_kind} cells are retained; removing them would bias density upward. The empty-window filter is based on total observed cells, not target {target_kind} cells.",
            "- Inference is one-sample testing of log2 enrichment against 0 at the ROI or patient level; individual axons/windows are not treated as independent samples.",
            "- This remains an area-proxy analysis. Use a true tissue mask if available for final edge/hole correction.",
            "",
            "## Parameters",
            "",
            f"- radius_um: `{', '.join(str(r) for r in args.radius_um)}`",
            f"- area_grid_step_um: `{args.area_grid_step_um}`",
            f"- min_axon_disk_area_fraction: `{args.min_axon_disk_area_fraction}`",
            f"- min_window_area_fraction: `{args.min_window_area_fraction}`",
            f"- min_window_total_cells: `{args.min_window_total_cells}`",
            f"- min_window_axon_count: `{args.min_window_axon_count}`",
            "",
            f"## {aggregate_label} Region Summary",
            "",
            dataframe_to_markdown(summary, floatfmt=".6g"),
            "",
            f"## {aggregate_label} Window Summary",
            "",
            dataframe_to_markdown(window_summary, floatfmt=".6g"),
            "",
            f"## {aggregate_label} Patient Rankings",
            "",
            dataframe_to_markdown(patient_rank, floatfmt=".6g"),
            "",
            "## One-Sample Tests",
            "",
            dataframe_to_markdown(top_tests, floatfmt=".6g"),
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in figures)
    lines.extend(
        [
            "",
            "## Tables",
            "",
            f"- `tables/{file_prefix}_region_enrichment.csv`",
            f"- `tables/{file_prefix}_patient_enrichment.csv`",
            f"- `tables/{file_prefix}_window_region_summary.csv`",
            f"- `tables/{file_prefix}_window_patient_summary.csv`",
            f"- `tables/{file_prefix}_window_context_density.csv`",
            f"- `tables/{file_prefix}_window_context_tests.csv`",
            f"- `tables/{file_prefix}_enrichment_tests.csv`",
            f"- `tables/{file_prefix}_window_table.csv` if `--write-window-table` was used",
            "",
        ]
    )
    path = outdir / f"{file_prefix}_spatial_enrichment_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    tables = outdir / "tables"
    figures = outdir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    radii = [float(radius) for radius in args.radius_um]
    file_prefix = safe_name(args.file_prefix)
    aggregate_figure_label = safe_name(args.aggregate_label)

    obs = read_obs(args.h5ad)
    region_rows: list[dict[str, object]] = []
    window_summary_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    group_cols = ["cohort", "dataset", "patient_id", "unique_region"]
    for keys, group in obs.groupby(group_cols, observed=True, sort=False):
        print(f"Processing {keys[0]} {keys[3]} ({len(group):,} cells)")
        rr, wr, ww = compute_region(
            keys,
            group,
            args.target_label,
            args.aggregate_label,
            radii,
            args.area_grid_step_um,
            args.window_spacing_um,
            args.min_window_area_fraction,
            args.min_axon_disk_area_fraction,
            args.min_window_total_cells,
            args.min_window_axon_count,
            args.batch_size,
            args.density_scale,
            args.write_window_table,
        )
        region_rows.extend(rr)
        window_summary_rows.extend(wr)
        window_rows.extend(ww)

    region = pd.DataFrame(region_rows)
    window_region = pd.DataFrame(window_summary_rows)
    patient = aggregate_region_to_patient(region, args.density_scale)
    window_patient = aggregate_window_to_patient(window_region, args.density_scale)
    region = add_target_alias_columns(region)
    patient = add_target_alias_columns(patient)
    window_region = add_target_alias_columns(window_region)
    window_patient = add_target_alias_columns(window_patient)
    window_context = window_context_long(window_region, args.density_scale)
    window_context_tests = paired_window_context_tests(window_region)

    tests = pd.concat(
        [
            one_sample_tests(
                region,
                "density_enrichment",
                "log2_density_enrichment",
                "region",
                ["cohort", "radius_um", "target"],
            ),
            one_sample_tests(
                patient,
                "density_enrichment",
                "log2_density_enrichment",
                "patient",
                ["cohort", "radius_um", "target"],
            ),
            one_sample_tests(
                window_region,
                "window_density_enrichment",
                "log2_window_density_enrichment",
                "region",
                ["cohort", "radius_um", "target"],
            ),
            one_sample_tests(
                window_patient,
                "window_density_enrichment",
                "log2_window_density_enrichment",
                "patient",
                ["cohort", "radius_um", "target"],
            ),
        ],
        ignore_index=True,
    )

    region_path = tables / f"{file_prefix}_region_enrichment.csv"
    patient_path = tables / f"{file_prefix}_patient_enrichment.csv"
    window_region_path = tables / f"{file_prefix}_window_region_summary.csv"
    window_patient_path = tables / f"{file_prefix}_window_patient_summary.csv"
    window_context_path = tables / f"{file_prefix}_window_context_density.csv"
    window_context_tests_path = tables / f"{file_prefix}_window_context_tests.csv"
    tests_path = tables / f"{file_prefix}_enrichment_tests.csv"
    region.to_csv(region_path, index=False)
    patient.to_csv(patient_path, index=False)
    window_region.to_csv(window_region_path, index=False)
    window_patient.to_csv(window_patient_path, index=False)
    window_context.to_csv(window_context_path, index=False)
    window_context_tests.to_csv(window_context_tests_path, index=False)
    tests.to_csv(tests_path, index=False)
    if args.write_window_table:
        pd.DataFrame(window_rows).to_csv(tables / f"{file_prefix}_window_table.csv", index=False)

    figure_paths = [
        path
        for path in [
            plot_enrichment_points(
                region,
                "density_enrichment",
                args.aggregate_label,
                figures / f"fig_{aggregate_figure_label}_axon_disk_enrichment_by_region.png",
                f"{display_label(args.aggregate_label)} near axons versus whole tissue",
            ),
            plot_enrichment_points(
                window_region,
                "window_density_enrichment",
                args.aggregate_label,
                figures / f"fig_{aggregate_figure_label}_window_enrichment_by_region.png",
                f"{display_label(args.aggregate_label)} in axon-associated versus background windows",
            ),
            plot_target_heatmap(
                region,
                "log2_density_enrichment",
                args.aggregate_label,
                figures / f"fig_{file_prefix}_target_median_axon_disk_log2_enrichment.png",
                "Median axon-disk log2 enrichment by target",
            ),
            plot_target_heatmap(
                window_region,
                "log2_window_density_enrichment",
                args.aggregate_label,
                figures / f"fig_{file_prefix}_target_median_window_log2_enrichment.png",
                "Median window log2 enrichment by target",
            ),
        ]
        if path is not None
    ]
    for plotted in [
        plot_window_context_density(
            window_region,
            window_context_tests,
            args.aggregate_label,
            args.density_scale,
            figures / f"fig_{aggregate_figure_label}_window_density_background_vs_axon_{COMBINED_PLOT_SUFFIX}.png",
            figures / f"fig_{aggregate_figure_label}_window_density_background_vs_axon_{COMBINED_PLOT_SUFFIX}.pdf",
            f"{display_label(args.aggregate_label)}: 100 um window density, {COMBINED_COHORT_LABEL} combined",
            aggregate_only=True,
            combine_cohorts=True,
            orientation="vertical",
        ),
        plot_window_context_density(
            window_region,
            window_context_tests,
            args.aggregate_label,
            args.density_scale,
            figures / f"fig_{file_prefix}_target_window_density_background_vs_axon_{COMBINED_PLOT_SUFFIX}.png",
            figures / f"fig_{file_prefix}_target_window_density_background_vs_axon_{COMBINED_PLOT_SUFFIX}.pdf",
            f"100 um window density by target, {COMBINED_COHORT_LABEL} combined",
            aggregate_only=False,
            combine_cohorts=True,
            font_multiplier=4.0,
            orientation="vertical",
        ),
    ]:
        if plotted is not None:
            figure_paths.extend(plotted)
    report = write_report(outdir, args.h5ad, region, patient, window_region, tests, figure_paths, args)
    print(f"Wrote {region_path}")
    print(f"Wrote {window_region_path}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
