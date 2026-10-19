#!/usr/bin/env python3
"""Analyze cellular-neighborhood enrichment during progression.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

DEFAULT_OUTDIR = Path("results/cn_neighborhood_enrichment_notebook_method")
DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTDIR / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_OUTDIR / "cache"))
(DEFAULT_OUTDIR / "mplconfig").mkdir(parents=True, exist_ok=True)
(DEFAULT_OUTDIR / "cache").mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


DEFAULT_H5ADS = [
    Path("data/AS_cellular_neighborhoods.h5ad"),
    Path("data/RP_cellular_neighborhoods.h5ad"),
]
DEFAULT_STRATA = Path("results/patient_axon_density_stratification_cn_ab_samples/tables/patient_axon_density.csv")

PREP_NEIGHBORHOODS = [0, 2, 3, 4, 5, 6, 7, 8, 9]
MODEL_NEIGHBORHOODS = [0, 2, 3, 4, 6, 7, 8, 9]
EPS = 1e-3
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the provided notebook OLS method on CN data.")
    parser.add_argument("--h5ad", type=Path, nargs="+", default=DEFAULT_H5ADS)
    parser.add_argument("--strata", type=Path, default=DEFAULT_STRATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument(
        "--group-by",
        choices=["axon_stratum", "grade", "progression"],
        default="axon_stratum",
        help="Group variable for the notebook OLS model.",
    )
    parser.add_argument("--grade-a", default="Low-grade", help="Grade coded as group 0.")
    parser.add_argument("--grade-b", default="High-grade", help="Grade coded as group 1.")
    parser.add_argument("--progression-a", default="NonProgressor", help="Progression label coded as group 0.")
    parser.add_argument("--progression-b", default="Progressor", help="Progression label coded as group 1.")
    return parser.parse_args()


def infer_cohort(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("rp_"):
        return "rp"
    if name.startswith("as_"):
        return "as"
    return path.stem


def parse_sample_id(cohort: str, unique_region: object) -> str:
    text = str(unique_region)
    if cohort == "as":
        match = re.search(r"/Batch\d+/([^_]+_[^_]+_[AB])(?:_|$)", text)
        if match:
            return match.group(1)
    return text


def normalize(X: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    arr = np.array(X.fillna(0).values)
    return pd.DataFrame(
        np.log2(eps + arr / arr.sum(axis=1, keepdims=True)),
        index=X.index.values,
        columns=X.columns,
    ).fillna(0)


def ordered_labels(labels: list[str]) -> list[str]:
    preferred = [label for label in CELL_TYPE_ORDER if label in labels]
    rest = sorted(label for label in labels if label not in preferred)
    return preferred + rest


def read_cn_obs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        cohort = infer_cohort(path)
        adata = ad.read_h5ad(path, backed="r")
        try:
            required = ["unique_region", "CN_k50_n10", "cell_type_sync"]
            optional = [column for column in ["condition", "progression"] if column in adata.obs.columns]
            obs = adata.obs[required + optional].copy()
        finally:
            adata.file.close()
        for column in obs.select_dtypes(["category"]).columns:
            obs[column] = obs[column].astype(str)
        obs["cohort"] = cohort
        obs["patients"] = obs["unique_region"].map(lambda value: parse_sample_id(cohort, value))
        obs["neighborhood10"] = pd.to_numeric(obs["CN_k50_n10"], errors="coerce").astype("Int64")
        obs["cell_type_sync"] = obs["cell_type_sync"].astype(str)
        if "condition" not in obs.columns:
            obs["condition"] = ""
        obs["condition"] = obs["condition"].astype(str)
        if "progression" not in obs.columns:
            obs["progression"] = ""
        obs["progression"] = obs["progression"].astype(str)
        frames.append(obs[["cohort", "patients", "neighborhood10", "cell_type_sync", "condition", "progression"]])
    return pd.concat(frames, ignore_index=True)


def mode_or_join(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    values = values.loc[~values.str.lower().isin(["", "nan", "none", "na"])]
    if values.empty:
        return ""
    modes = values.mode()
    if len(modes) == 1:
        return str(modes.iloc[0])
    return "|".join(sorted(values.unique()))


def build_notebook_tables(obs: pd.DataFrame, cells: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ct_freq = (
        obs.groupby(["patients", "cell_type_sync"], observed=True)
        .size()
        .unstack("cell_type_sync", fill_value=0)
        .reindex(columns=cells, fill_value=0)
        .reset_index()
    )

    all_counts = (
        obs.groupby(["patients", "neighborhood10", "cell_type_sync"], observed=True)
        .size()
        .unstack("cell_type_sync", fill_value=0)
        .reindex(columns=cells, fill_value=0)
        .reset_index()
    )
    return ct_freq, all_counts


def run_notebook_method(
    ct_freq: pd.DataFrame,
    all_freqs: pd.DataFrame,
    patients: list[str],
    group: pd.Series,
    cells: list[str],
    eps: float,
    group_one_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    neighborhood_col = "neighborhood10"

    # Normalized overall cell type frequencies.
    X_cts = normalize(ct_freq.set_index("patients").loc[patients, cells], eps=eps)

    # Normalized neighborhood-specific cell type frequencies.
    df_list = []
    for nb in PREP_NEIGHBORHOODS:
        cond_nb = (
            all_freqs.loc[all_freqs[neighborhood_col].eq(nb), ["patients"] + cells]
            .rename({col: col + "_" + str(nb) for col in cells}, axis=1)
            .set_index("patients")
        )
        cond_nb = cond_nb.reindex(patients)
        df_list.append(normalize(cond_nb, eps=eps))
    X_cond_nb = pd.concat(df_list, axis=1).loc[patients]

    # Differential enrichment for all cell subsets.
    changes = {}
    rows = []
    for col in cells:
        for nb in MODEL_NEIGHBORHOODS:
            X = pd.concat(
                [
                    X_cts[col],
                    group.astype("int"),
                    pd.Series(np.ones(len(group)), index=group.index.values),
                ],
                axis=1,
            ).values
            target = col + "_%d" % nb
            if target in X_cond_nb.columns:
                Y = X_cond_nb[target].values
                X = X[~pd.isna(Y)]
                Y = Y[~pd.isna(Y)]
                try:
                    results = sm.OLS(Y, X).fit()
                    p_value = float(results.pvalues[1])
                    coefficient = float(results.params[1])
                except Exception:
                    p_value = np.nan
                    coefficient = np.nan
                changes[(col, nb)] = (p_value, coefficient)
                rows.append(
                    {
                        "cell_type_sync": col,
                        "CN_k50_n10": nb,
                        "raw_p_value": p_value,
                        "group_coefficient": coefficient,
                        "direction": f"enriched in {group_one_label}"
                        if coefficient > 0
                        else f"depleted in {group_one_label}"
                        if coefficient < 0
                        else "no fitted difference",
                    }
                )

    # Make dataframes with coeffs and pvalues, matching notebook shape.
    dat = pd.DataFrame(changes).loc[1].unstack()
    dat = pd.DataFrame(np.nan_to_num(dat.values), index=dat.index, columns=dat.columns).T
    dat = dat.sort_index(ascending=True).loc[:, X_cts.columns]
    pvals = pd.DataFrame(changes).loc[0].unstack().T.sort_index(ascending=True).loc[:, X_cts.columns]
    long_results = pd.DataFrame(rows)
    return dat, pvals, long_results


def plot_heatmap(dat: pd.DataFrame, pvals: pd.DataFrame, cohort: str, out: Path, axon_only: bool = False) -> Path:
    plot_dat = dat.copy()
    plot_pvals = pvals.copy()
    if axon_only:
        cols = [col for col in plot_dat.columns if "Axon" in str(col)]
        plot_dat = plot_dat.loc[:, cols]
        plot_pvals = plot_pvals.loc[:, cols]
    else:
        cols = ordered_labels(list(plot_dat.columns))
        plot_dat = plot_dat.loc[:, cols]
        plot_pvals = plot_pvals.loc[:, cols]

    fig_width = max(7.0, 0.8 * len(plot_dat.columns))
    fig_height = max(4.0, 0.45 * len(plot_dat.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(plot_dat, cmap="bwr", vmin=-1, vmax=1, cbar=False, ax=ax)
    for a, b in zip(*np.where(plot_pvals < 0.05)):
        ax.text(b + 0.5, a + 0.55, "*", fontsize=16, ha="center", va="center")
    ax.set_title(
        f"{cohort.upper()} notebook-method CN enrichment"
        + (" (axon labels)" if axon_only else "")
    )
    ax.set_xlabel("")
    ax.set_ylabel("CN_k50_n10")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".6g") -> str:
    if df.empty:
        return "_No rows._"

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value).replace("\n", " ").replace("|", r"\|")

    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, all_results: pd.DataFrame, sample_counts: pd.DataFrame, figures: list[Path]) -> Path:
    top = (
        all_results.dropna(subset=["raw_p_value"])
        .sort_values("raw_p_value")
        .head(30)
    )
    nominal = (
        all_results.loc[all_results["raw_p_value"].lt(0.05)]
        .groupby(["cohort", "direction"], observed=True)
        .size()
        .reset_index(name="n_nominal_p_lt_0.05")
    )
    lines = [
        "# CN neighborhood enrichment: notebook method",
        "",
        "## Method Used",
        "",
        "This run follows the provided notebook method: `normalize = log2(1e-3 + row_fraction)`, then `sm.OLS(Y, X).fit()` with `X = [overall cell-type frequency, group, constant]` and `Y = neighborhood-specific cell-type frequency`.",
        "",
        "No BH/FDR correction is applied here. Stars in heatmaps mark raw `p < 0.05`, as in the notebook.",
        "",
        "The model neighborhoods follow the code block exactly: `[0, 2, 3, 4, 6, 7, 8, 9]`.",
        "",
        "Positive coefficients mean enrichment in the group coded as `1`.",
        "",
        "## Sample Counts",
        "",
        dataframe_to_markdown(sample_counts),
        "",
        "## Nominal p < 0.05 Counts",
        "",
        dataframe_to_markdown(nominal),
        "",
        "## Top Raw p-value Tests",
        "",
        dataframe_to_markdown(top, floatfmt=".6g"),
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{path}`" for path in figures)
    lines.extend(
        [
            "",
            "## Tables",
            "",
            "- `tables/notebook_method_coefficients_long.csv`",
            "- `tables/notebook_method_top_raw_p_values.csv`",
            "- `tables/notebook_method_coefficients_<cohort>.csv`",
            "- `tables/notebook_method_pvalues_<cohort>.csv`",
            "",
        ]
    )
    path = outdir / "cn_neighborhood_enrichment_notebook_method_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    tables = outdir / "tables"
    figures = outdir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    strata = None
    if args.group_by == "axon_stratum":
        strata = pd.read_csv(args.strata)
        strata["cohort"] = strata["cohort"].astype(str)
        strata["patient_id"] = strata["patient_id"].astype(str)
        strata = strata.loc[strata["axon_density_stratum"].isin(["axon-poor", "axon-rich"])].copy()

    obs = read_cn_obs(args.h5ad)
    all_long = []
    figure_paths = []
    sample_counts = []
    for cohort, cohort_obs in obs.groupby("cohort", observed=True):
        if args.group_by == "axon_stratum":
            assert strata is not None
            cohort_strata = strata.loc[strata["cohort"].eq(cohort)].copy()
            cohort_obs = cohort_obs.merge(
                cohort_strata[["patient_id", "axon_density_stratum"]],
                left_on="patients",
                right_on="patient_id",
                how="inner",
            )
            patients = sorted(cohort_obs["patients"].unique())
            group = (
                cohort_strata.set_index("patient_id")
                .reindex(patients)["axon_density_stratum"]
                .eq("axon-rich")
                .astype(int)
            )
            group.index = patients
            group_labels = group.map({0: "axon-poor", 1: "axon-rich"})
            coefficient_label = "axon-rich vs axon-poor"
            group_one_label = "axon-rich"
            group_zero_label = "axon-poor"
        elif args.group_by == "grade":
            patient_grade = cohort_obs.groupby("patients", observed=True)["condition"].agg(mode_or_join)
            patient_grade = patient_grade.loc[patient_grade.isin([args.grade_a, args.grade_b])]
            patients = sorted(patient_grade.index)
            cohort_obs = cohort_obs.loc[cohort_obs["patients"].isin(patients)].copy()
            group = patient_grade.reindex(patients).eq(args.grade_b).astype(int)
            group.index = patients
            group_labels = group.map({0: args.grade_a, 1: args.grade_b})
            coefficient_label = f"{args.grade_b} vs {args.grade_a}"
            group_one_label = args.grade_b
            group_zero_label = args.grade_a
        else:
            patient_progression = cohort_obs.groupby("patients", observed=True)["progression"].agg(mode_or_join)
            patient_progression = patient_progression.loc[
                patient_progression.isin([args.progression_a, args.progression_b])
            ]
            patients = sorted(patient_progression.index)
            cohort_obs = cohort_obs.loc[cohort_obs["patients"].isin(patients)].copy()
            group = patient_progression.reindex(patients).eq(args.progression_b).astype(int)
            group.index = patients
            group_labels = group.map({0: args.progression_a, 1: args.progression_b})
            coefficient_label = f"{args.progression_b} vs {args.progression_a}"
            group_one_label = args.progression_b
            group_zero_label = args.progression_a
        cells = ordered_labels(sorted(cohort_obs["cell_type_sync"].unique()))
        ct_freq, all_freqs = build_notebook_tables(cohort_obs, cells)
        dat, pvals, long_results = run_notebook_method(
            ct_freq,
            all_freqs,
            patients,
            group,
            cells,
            args.eps,
            group_one_label,
        )
        long_results.insert(0, "cohort", cohort)
        long_results["group_by"] = args.group_by
        long_results["coefficient_label"] = coefficient_label
        long_results["n_samples"] = len(patients)
        long_results[f"n_{group_zero_label}"] = int(group.eq(0).sum())
        long_results[f"n_{group_one_label}"] = int(group.eq(1).sum())
        all_long.append(long_results)

        dat.to_csv(tables / f"notebook_method_coefficients_{cohort}.csv")
        pvals.to_csv(tables / f"notebook_method_pvalues_{cohort}.csv")
        figure_paths.append(plot_heatmap(dat, pvals, cohort, figures / f"fig_notebook_method_cn_enrichment_{cohort}.png"))
        figure_paths.append(
            plot_heatmap(
                dat,
                pvals,
                cohort,
                figures / f"fig_notebook_method_cn_enrichment_{cohort}_axons.png",
                axon_only=True,
            )
        )

        counts = group_labels.value_counts().rename_axis("group_label")
        counts = counts.reset_index(name="n_samples")
        counts.insert(0, "cohort", cohort)
        sample_counts.append(counts)

    results = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    results_path = tables / "notebook_method_coefficients_long.csv"
    results.to_csv(results_path, index=False)
    results.sort_values("raw_p_value").head(100).to_csv(
        tables / "notebook_method_top_raw_p_values.csv",
        index=False,
    )
    sample_counts_df = pd.concat(sample_counts, ignore_index=True) if sample_counts else pd.DataFrame()
    sample_counts_df.to_csv(tables / "notebook_method_sample_counts.csv", index=False)
    report = write_report(outdir, results, sample_counts_df, figure_paths)
    print(f"Wrote {results_path}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
