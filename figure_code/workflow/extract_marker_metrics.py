#!/usr/bin/env python3
"""Build region-level local T-cell marker counts from Active Surveillance cohort
PPA H5AD files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata._io.specs import read_elem
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from patient_ids import validate_patient_groups

MARKERS = ("AR", "Ki67", "GZMB", "PD1")
AXONS = {
    "CXCR4pos___CD44pos___FYNpos_Axon": "CXCR4+ / CD44+ / FYN+ Axon",
    "NRXN1pos___MAOApos_Axon": "NRXN1+ / MAOA+ Axon",
}
AXON_RELEASE_LABELS = {
    "CXCR4pos___CD44pos___FYNpos_Axon": "CXCR4_CD44_FYN",
    "NRXN1pos___MAOApos_Axon": "NRXN1_MAOA",
}
RADII = (20, 50, 100)
GROUPS = ("NonProgressor", "Progressor")


def read_columns(handle: h5py.File, group: str, columns: list[str]) -> pd.DataFrame:
    missing = set(columns).difference(handle[group].keys())
    if missing:
        raise ValueError(f"{handle.filename}: {group} missing columns {sorted(missing)}")
    data = pd.DataFrame({column: read_elem(handle[group][column]) for column in columns})
    for column in data:
        data[column] = data[column].astype(str)
    return data


def unique_metadata(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, group in obs.groupby("unique_region", sort=True, observed=True):
        row = {"unique_region": region}
        for key, allowed in [("condition", ("Low-grade", "High-grade")), ("progression", GROUPS)]:
            values = sorted(set(group[key]).intersection(allowed))
            if len(values) > 1 or (key == "progression" and not values):
                raise ValueError(f"Expected one valid {key} for {region!r}, got {values}")
            raw_values = sorted(set(group[key]).difference({"nan", "None", "none", ""}))
            row[key] = values[0] if values else (raw_values[0] if len(raw_values) == 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("unique_region")


def extract_file(path: Path, marker: str, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = f"cell_type_sync_{marker}"
    labels = {"positive": f"{marker}+ T Cells", "negative": f"{marker}- T Cells"}
    composition, abundance = [], []
    with h5py.File(path, "r") as handle:
        obs = read_columns(handle, "obs", ["unique_region", "condition", "progression", split, "cell_type_sync"])
        metadata = unique_metadata(obs)
        tissue = obs.groupby(["unique_region", split], observed=True).size()
        axon_counts = obs.loc[obs.cell_type_sync.eq(AXONS[tag])].groupby("unique_region").size()
        for radius in RADII:
            frame = read_columns(handle, f"uns/ppa_result_{radius}", ["unique_region", split])
            near = frame.groupby(["unique_region", split], observed=True).size()
            unknown = set(frame.unique_region).difference(metadata.index)
            if unknown:
                raise ValueError(f"PPA has regions absent from obs: {sorted(unknown)}")
            for region, meta in metadata.iterrows():
                base = dict(df="as", unique_region=region, condition=meta.condition, progression=meta.progression,
                            marker=marker, axon=AXONS[tag], axon_tag=tag, radius_um=radius)
                n = {state: int(near.get((region, label), 0)) for state, label in labels.items()}
                total = n["positive"] + n["negative"]
                if total:
                    composition.append(dict(base, raw_negative_near=n["negative"], raw_positive_near=n["positive"],
                                            raw_marker_split_near_total=total, pct_marker_positive_near=100 * n["positive"] / total))
                for state, label in labels.items():
                    tissue_total = int(tissue.get((region, label), 0))
                    if tissue_total:
                        abundance.append(dict(base, marker_state=state, target=label, tissue_total=tissue_total,
                                              axon_total=int(axon_counts.get(region, 0)), raw_n_near=n[state],
                                              tissue_norm=n[state] / tissue_total))
    return pd.DataFrame(composition), pd.DataFrame(abundance)


def composition_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (marker, axon, radius), group in data.groupby(["marker", "axon", "radius_um"], observed=True, sort=True):
        a = group.loc[group.progression.eq("Progressor"), "pct_marker_positive_near"].to_numpy()
        b = group.loc[group.progression.eq("NonProgressor"), "pct_marker_positive_near"].to_numpy()
        row = dict(comparison="progression", group_col="progression", adverse_group="Progressor", control_group="NonProgressor",
                   marker=marker, axon=axon, radius_um=radius, n_adverse=len(a), n_control=len(b),
                   median_pct_adverse=float(np.median(a)), median_pct_control=float(np.median(b)),
                   median_pct_diff_adverse_minus_control=float(np.median(a) - np.median(b)), method="mannwhitneyu")
        for alternative, column in [("two-sided", "p_value_two_sided"), ("greater", "p_value_greater_adverse_vs_control"), ("less", "p_value_less_adverse_vs_control")]:
            row[column] = float(mannwhitneyu(a, b, alternative=alternative).pvalue) if len(a) and len(b) else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    for pcolumn, suffix in [("p_value_two_sided", "two_sided"), ("p_value_greater_adverse_vs_control", "greater"), ("p_value_less_adverse_vs_control", "less")]:
        valid = result[pcolumn].notna()
        result.loc[valid, f"q_{suffix}_all_tests"] = multipletests(result.loc[valid, pcolumn], method="fdr_bh")[1]
        for _, group in result.groupby("marker"):
            index = group.index[group[pcolumn].notna()]
            result.loc[index, f"q_{suffix}_within_marker"] = multipletests(result.loc[index, pcolumn], method="fdr_bh")[1]
        result[f"significant_{suffix}_within_marker"] = result[f"q_{suffix}_within_marker"].lt(0.05)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("significance_axons_immune_cells"))
    parser.add_argument("--markers", nargs="+", choices=MARKERS, default=list(MARKERS))
    args = parser.parse_args()
    comps, metrics = [], []
    for marker in args.markers:
        for tag in AXONS:
            path = args.input_dir / f"AS_PPA_T_cell_marker_{marker}_{AXON_RELEASE_LABELS[tag]}_axon.h5ad"
            composition, abundance = extract_file(path, marker, tag)
            comps.append(composition)
            metrics.append(abundance)
    composition = pd.concat(comps, ignore_index=True)
    abundance = pd.concat(metrics, ignore_index=True)
    validate_patient_groups(composition)
    validate_patient_groups(abundance)
    compdir = args.outdir / "paper_ready_tcell_marker_composition"
    metricdir = args.outdir / "paper_ready_tcell_marker_positivity"
    (compdir / "progression").mkdir(parents=True, exist_ok=True)
    metricdir.mkdir(parents=True, exist_ok=True)
    composition.to_csv(compdir / "marker_positive_fraction_region_level.csv", index=False)
    abundance.to_csv(metricdir / "tcell_marker_region_level_metrics.csv", index=False)
    composition_tests(composition).to_csv(compdir / "progression/progression_marker_positive_fraction_stats.csv", index=False)
    print(f"Rebuilt {len(composition)} composition rows and {len(abundance)} abundance rows from {len(args.markers) * len(AXONS)} H5AD inputs.")


if __name__ == "__main__":
    main()
