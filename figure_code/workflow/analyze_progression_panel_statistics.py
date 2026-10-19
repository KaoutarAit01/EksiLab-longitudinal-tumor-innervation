#!/usr/bin/env python3
"""Rebuild the progression statistics for Figures 5d and 6e-f from PPA H5ADs.

This script extracts region-level counts, recomputes the radial and marker progression tests, and writes the compact
table consumed by the manuscript plotting scripts.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from anndata._io.specs import read_elem

from extract_marker_metrics import AXONS as MARKER_AXONS
from extract_marker_metrics import MARKERS, extract_file, read_columns

RADII = (20, 50, 100)
GROUP_A = "Progressor"
GROUP_B = "NonProgressor"
CELL_TYPES = (
    "CD4+ T Cells",
    "FOXP3+ T Cells",
    "Macrophages",
    "CD20+",
    "CD66b+",
    "T Cells",
    "Tryp+",
)
AXON_FILES = {
    "AS_PPA_cell_type_CXCR4_CD44_FYN_axon.h5ad": (
        "CXCR4pos_CD44pos_FYNpos_Axon", "CXCR4+ / CD44+ / FYN+ Axon"
    ),
    "AS_PPA_cell_type_NRXN1_MAOA_axon.h5ad": (
        "NRXN1pos_MAOApos_Axon", "NRXN1+ / MAOA+ Axon"
    ),
    "AS_PPA_cell_type_TH_axon.h5ad": ("THpos_Axon", "TH+ Axon"),
}
MARKER_RELEASE_LABELS = {
    "CXCR4pos___CD44pos___FYNpos_Axon": "CXCR4_CD44_FYN",
    "NRXN1pos___MAOApos_Axon": "NRXN1_MAOA",
}
RADIAL_METRICS = ("proximal_fraction", "radial_centroid_um", "log_proximal_density_ratio")
MARKER_CELL_LABELS = {marker: f"{marker}+ T Cells" for marker in MARKERS}
STAT_MARKERS = ("AR", "GZMB", "Ki67", "PD1")
COMPOSITION_LABEL = "PD1+ fraction among nearby T Cells"
SEED = 20260609
N_PERMUTATIONS = 999
N_BOOTSTRAPS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("radial", "markers"), required=True)
    parser.add_argument("--cell-ppa", type=Path, nargs="*", default=[])
    parser.add_argument("--marker-ppa", type=Path, nargs="*", default=[])
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    result.loc[order] = np.clip(np.minimum.accumulate(adjusted[::-1])[::-1], 0, 1)
    return result


def h5_region_metadata(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, frame in obs.groupby("unique_region", observed=True, sort=True):
        row = {"unique_region": str(region)}
        for key, allowed in (("condition", {"Low-grade", "High-grade"}),
                             ("progression", {GROUP_A, GROUP_B})):
            raw = set(frame[key].astype(str)).difference({"nan", "NaN", "None", "none", ""})
            values = raw & allowed
            if len(values) > 1 or (key == "progression" and not values and len(raw) > 1):
                raise ValueError(f"Expected one {key} label for {region!r}; found {sorted(raw)}")
            row[key] = next(iter(values), next(iter(raw), np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def extract_cell_type_metrics(paths: list[Path]) -> pd.DataFrame:
    expected = AXON_FILES
    found: dict[str, Path] = {}
    for path in paths:
        if path.name in expected:
            found[path.name] = path
    missing = sorted(set(expected) - set(found))
    if missing:
        raise ValueError("Missing cell-type PPA H5AD inputs:\n  " + "\n  ".join(missing))

    records: list[dict[str, object]] = []
    for filename, (tag, axon) in expected.items():
        path = found[filename]
        with h5py.File(path, "r") as handle:
            obs = read_columns(
                handle, "obs", ["unique_region", "condition", "progression", "cell_type_sync"]
            )
            metadata = h5_region_metadata(obs)
            tissue = (
                obs.loc[obs["cell_type_sync"].isin(CELL_TYPES)]
                .groupby(["unique_region", "cell_type_sync"], observed=True)
                .size()
            )
            near_by_radius: dict[int, pd.Series] = {}
            for radius in RADII:
                key = f"uns/ppa_result_{radius}"
                if key not in handle:
                    raise ValueError(f"{path.name} is missing {key}")
                near = pd.DataFrame(read_elem(handle[key]))
                if "cell_type_sync" not in near or "unique_region" not in near:
                    raise ValueError(f"{path.name}:{key} lacks cell_type_sync or unique_region")
                near = near.loc[near["cell_type_sync"].astype(str).isin(CELL_TYPES)]
                near_by_radius[radius] = near.groupby(
                    ["unique_region", "cell_type_sync"], observed=True
                ).size()

            for meta in metadata.itertuples(index=False):
                for target in CELL_TYPES:
                    tissue_total = int(tissue.get((meta.unique_region, target), 0))
                    if tissue_total <= 0:
                        continue
                    for radius in RADII:
                        records.append({
                            "df": "as",
                            "adata": "immune",
                            "region_id": f"as|immune|{meta.unique_region}",
                            "unique_region": meta.unique_region,
                            "condition": meta.condition,
                            "progression": meta.progression,
                            "target": target,
                            "axon": axon,
                            "radius_um": radius,
                            "tissue_total": tissue_total,
                            "raw_n_near": int(near_by_radius[radius].get((meta.unique_region, target), 0)),
                        })
    return pd.DataFrame(records)


def radial_region_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["region_id", "df", "unique_region", "condition", "progression", "target", "axon", "tissue_total"]
    wide = metrics.pivot_table(
        index=keys, columns="radius_um", values="raw_n_near", aggfunc="first", observed=True
    ).reset_index()
    wide = wide.dropna(subset=list(RADII)).copy()
    wide["count_0_20"] = wide[20]
    wide["count_20_50"] = wide[50] - wide[20]
    wide["count_50_100"] = wide[100] - wide[50]
    count_cols = ["count_0_20", "count_20_50", "count_50_100"]
    wide = wide.loc[wide[count_cols].ge(0).all(axis=1)].copy()
    zones = []
    for zone, inner, outer in (("0_20", 0, 20), ("20_50", 20, 50), ("50_100", 50, 100)):
        part = wide[keys + [f"count_{zone}"]].copy()
        part = part.rename(columns={f"count_{zone}": "zone_count"})
        part["zone"] = zone
        part["midpoint_um"] = (inner + outer) / 2
        part["annulus_area"] = math.pi * (outer**2 - inner**2)
        zones.append(part)
    annular = pd.concat(zones, ignore_index=True)

    rows = []
    for key, frame in annular.groupby(keys, dropna=False, observed=True, sort=False):
        counts = frame.set_index("zone")["zone_count"]
        total = float(counts.sum())
        if total <= 0:
            continue
        inner_density = (float(counts.get("0_20", 0)) + 0.5) / (math.pi * 20**2)
        outer_count = float(counts.get("20_50", 0) + counts.get("50_100", 0))
        outer_density = (outer_count + 0.5) / (math.pi * (100**2 - 20**2))
        rows.append({
            **dict(zip(keys, key)),
            "proximal_fraction": float(counts.get("0_20", 0) / total),
            "radial_centroid_um": float((frame["zone_count"] * frame["midpoint_um"]).sum() / total),
            "log_proximal_density_ratio": float(math.log(inner_density / outer_density)),
        })
    return pd.DataFrame(rows)


def permutation_effect(values: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    valid = np.isfinite(values)
    values, labels = values[valid], labels[valid]
    if len(values) < 6 or len(np.unique(labels)) < 2:
        return np.nan, np.nan
    effect = float(values[labels == 1].mean() - values[labels == 0].mean())
    null = np.empty(N_PERMUTATIONS, dtype=float)
    for i in range(N_PERMUTATIONS):
        permuted = rng.permutation(labels)
        null[i] = abs(float(values[permuted == 1].mean() - values[permuted == 0].mean()))
    return effect, float((1 + np.sum(null >= abs(effect))) / (N_PERMUTATIONS + 1))


def radial_statistics(metrics: pd.DataFrame) -> pd.DataFrame:
    radial = radial_region_summary(metrics)
    radial = radial.loc[radial["progression"].isin((GROUP_A, GROUP_B))]
    rng = np.random.default_rng(SEED)
    rows = []
    for target in CELL_TYPES:
        for _, axon in AXON_FILES.values():
            data = radial.loc[radial["target"].eq(target) & radial["axon"].eq(axon)]
            labels = data["progression"].eq(GROUP_A).astype(int).to_numpy()
            for metric in RADIAL_METRICS:
                effect, p_value = permutation_effect(data[metric].to_numpy(), labels, rng)
                rows.append({
                    "analysis_family": "progression_radial_distribution",
                    "cohort": "as",
                    "axon_a": axon,
                    "cell_type": target,
                    "radius_um": 100,
                    "endpoint": metric,
                    "effect_size": effect,
                    "p_value": p_value,
                    "model": "two-sided label permutation; 999 permutations",
                })
    result = pd.DataFrame(rows)
    result["q_value"] = benjamini_hochberg(result["p_value"])
    return result


def fit_count_model(data: pd.DataFrame) -> tuple[float, float, str]:
    """Fit the NB model used for marker-positive counts, with source fallbacks."""
    data = data.copy()
    data["group_indicator"] = data["progression"].eq(GROUP_A).astype(int)
    formula = "raw_n_near ~ group_indicator"
    counts = data["progression"].value_counts()
    if counts.get(GROUP_A, 0) < 3 or counts.get(GROUP_B, 0) < 3:
        return np.nan, np.nan, "insufficient_n"
    if data.loc[data["progression"].eq(GROUP_A), "raw_n_near"].sum() == 0 or data.loc[
        data["progression"].eq(GROUP_B), "raw_n_near"
    ].sum() == 0:
        return np.nan, np.nan, "all_zero_group"

    captured: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.negativebinomial(
                formula, data=data, offset=np.log(data["tissue_total"])
            ).fit(disp=False, maxiter=500, cov_type="HC0", use_t=True)
        captured = [str(item.message) for item in caught]
        coefficient = float(fit.params.get("group_indicator", np.nan))
        p_value = float(fit.pvalues.get("group_indicator", np.nan))
        alpha = float(fit.params.get("alpha", np.nan))
        critical_warning = any(
            phrase in message.lower()
            for message in captured
            for phrase in ("failed to converge", "inverting hessian failed", "singular matrix")
        )
        valid = (
            bool(fit.mle_retvals.get("converged", False))
            and np.isfinite(coefficient)
            and np.isfinite(float(fit.bse.get("group_indicator", np.nan)))
            and float(fit.bse.get("group_indicator", np.nan)) > 0
            and np.isfinite(p_value)
            and np.isfinite(alpha)
            and 1e-4 < alpha < 1e4
            and abs(coefficient) < 10
            and not critical_warning
        )
        if valid:
            return coefficient, p_value, "negative_binomial_mle_hc0"
    except Exception as exc:
        captured.append(f"{type(exc).__name__}: {exc}")

    try:
        alpha = 1.0
        for _ in range(30):
            preliminary = smf.glm(
                formula, data=data, offset=np.log(data["tissue_total"]),
                family=sm.families.NegativeBinomial(alpha=alpha),
            ).fit()
            fitted = np.asarray(preliminary.fittedvalues, dtype=float)
            observed = data["raw_n_near"].to_numpy(dtype=float)
            df_resid = max(len(observed) - len(preliminary.params), 1)
            estimate = np.sum(((observed - fitted) ** 2 - fitted) / (fitted**2)) / df_resid
            updated = 0.5 * alpha + 0.5 * float(np.clip(estimate, 1e-4, 100.0))
            if abs(math.log(updated / alpha)) < 1e-4:
                alpha = updated
                break
            alpha = updated
        fit = smf.glm(
            formula, data=data, offset=np.log(data["tissue_total"]),
            family=sm.families.NegativeBinomial(alpha=alpha),
        ).fit(cov_type="HC0", use_t=True)
        coefficient = float(fit.params.get("group_indicator", np.nan))
        p_value = float(fit.pvalues.get("group_indicator", np.nan))
        if np.isfinite(coefficient) and np.isfinite(p_value):
            return coefficient, p_value, "negative_binomial_glm_hc0_fixed_alpha"
    except Exception:
        pass

    try:
        fit = smf.glm(
            formula, data=data, offset=np.log(data["tissue_total"]), family=sm.families.Poisson()
        ).fit(cov_type="HC0", use_t=True)
        coefficient = float(fit.params.get("group_indicator", np.nan))
        p_value = float(fit.pvalues.get("group_indicator", np.nan))
        if np.isfinite(coefficient) and np.isfinite(p_value):
            return coefficient, p_value, "poisson_glm_hc0"
    except Exception:
        pass
    return np.nan, np.nan, "fit_failed"


def marker_statistics(paths: list[Path]) -> pd.DataFrame:
    expected = {
        f"AS_PPA_T_cell_marker_{marker}_{release_label}_axon.h5ad": (marker, tag)
        for marker in STAT_MARKERS
        for tag, release_label in MARKER_RELEASE_LABELS.items()
    }
    found = {path.name: path for path in paths if path.name in expected}
    missing = sorted(set(expected) - set(found))
    if missing:
        raise ValueError("Missing marker PPA H5AD inputs:\n  " + "\n  ".join(missing))

    compositions, abundances = [], []
    for filename in sorted(expected):
        marker, tag = expected[filename]
        composition, abundance = extract_file(found[filename], marker, tag)
        compositions.append(composition)
        abundances.append(abundance)
    comp = pd.concat(compositions, ignore_index=True)
    abundance = pd.concat(abundances, ignore_index=True)
    abundance = abundance.loc[
        abundance["marker_state"].eq("positive")
        & abundance["progression"].isin((GROUP_A, GROUP_B))
        & abundance["tissue_total"].gt(0)
        & abundance["raw_n_near"].ge(0)
    ].copy()

    rows = []
    for marker in STAT_MARKERS:
        for axon in MARKER_AXONS.values():
            for radius in RADII:
                data = abundance.loc[
                    abundance["marker"].eq(marker)
                    & abundance["axon"].eq(axon)
                    & abundance["radius_um"].eq(radius)
                ].copy()
                effect, p_value, model = fit_count_model(data)
                rows.append({
                    "analysis_family": "progression_marker_abundance",
                    "cohort": "as",
                    "axon_a": axon,
                    "cell_type": MARKER_CELL_LABELS[marker],
                    "radius_um": radius,
                    "endpoint": "marker-positive nearby count with tissue-total offset",
                    "effect_size": effect,
                    "effect_scale": "log rate ratio; Progressor versus NonProgressor",
                    "p_value": p_value,
                    "model": model,
                })

    rng = np.random.default_rng(SEED)
    for marker in STAT_MARKERS:
        for axon in MARKER_AXONS.values():
            for radius in RADII:
                data = comp.loc[
                    comp["marker"].eq(marker)
                    & comp["axon"].eq(axon)
                    & comp["radius_um"].eq(radius)
                    & comp["progression"].isin((GROUP_A, GROUP_B))
                ].copy()
                labels = data["progression"].eq(GROUP_A).astype(int).to_numpy()
                effect, p_value = permutation_effect(
                    data["pct_marker_positive_near"].to_numpy(), labels, rng
                )
                rows.append({
                    "analysis_family": "progression_marker_composition",
                    "cohort": "as",
                    "axon_a": axon,
                    "cell_type": f"{marker}+ fraction among nearby T Cells",
                    "radius_um": radius,
                    "endpoint": "marker-positive percentage-point difference; Progressor minus NonProgressor",
                    "effect_size": effect,
                    "p_value": p_value,
                    "model": "two-sided label permutation; 999 permutations",
                })

    result = pd.DataFrame(rows)
    for family in result["analysis_family"].unique():
        idx = result["analysis_family"].eq(family)
        result.loc[idx, "q_value"] = benjamini_hochberg(result.loc[idx, "p_value"])
    return result


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.mode == "radial":
        result = radial_statistics(extract_cell_type_metrics(args.cell_ppa))
        name = "progression_radial_statistics.csv"
    else:
        result = marker_statistics(args.marker_ppa)
        name = "progression_marker_statistics.csv"
    result.to_csv(args.outdir / name, index=False)
    result.to_csv(args.outdir / "focused_fdr_master_analysis_table.csv", index=False)
    print(args.outdir / name)


if __name__ == "__main__":
    main()
