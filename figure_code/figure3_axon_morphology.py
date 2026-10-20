"""Classify axon morphology and compare marker expression across shape groups."""

from pathlib import Path
import argparse
import json
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from scipy import sparse
from scipy.optimize import brentq
from scipy.stats import norm, mannwhitneyu, ttest_ind
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests


RP_PATH = Path("data/RP_Neuronal_data_axons.h5ad")
AS_PATH = None
# AS_PATH = Path("data/AS_Neuronal_data_axons.h5ad")
OUTPUT_DIR = Path("morphology_results")
LABELS = {
    "area": ("Small", "Large"),
    "eccentricity": ("Circular", "Elongated"),
    "solidity": ("Branched", "Unbranched"),
}
COLORS = {"Branched": "#3B4CC0", "Unbranched": "#B40426"}


def save_figure(fig, output, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def morphology(obs):
    """Calculate missing shape measurements and flag invalid values."""
    result = obs.copy()

    def column(*names):
        for name in names:
            if name in obs:
                return pd.to_numeric(obs[name], errors="coerce")
        raise ValueError(f"Missing morphology column: one of {names}")

    area = column("area")
    result["area"] = area.where(np.isfinite(area) & (area > 0))
    if "solidity" in obs:
        sol = column("solidity")
    else:
        convex = column("area_convex", "convex_area")
        sol = area.div(convex.where(convex > 0))
    if "eccentricity" in obs:
        ecc = column("eccentricity")
    else:
        major = column("axis_major_length", "major_axis_length")
        minor = column("axis_minor_length", "minor_axis_length")
        valid = (major > 0) & (minor >= 0) & (minor <= major)
        ecc = np.sqrt(1 - (minor / major.where(valid)) ** 2)
    for key, values in (("solidity", sol), ("eccentricity", ecc)):
        result[key] = values.where(np.isfinite(values) & values.between(0, 1))
    return result


def fit_threshold(values, feature):
    """Split each feature at the crossing of two fitted Gaussians."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    x = np.log1p(values) if feature == "area" else values
    if len(x) < 4 or np.unique(x).size < 2:
        raise ValueError(f"{feature}: insufficient variation for a two-component GMM")
    model = GaussianMixture(n_components=2, random_state=0).fit(x[:, None])
    if not model.converged_:
        raise ValueError(f"{feature}: GMM did not converge")
    order = np.argsort(model.means_.ravel())
    means = model.means_.ravel()[order]
    stds = np.sqrt(model.covariances_.ravel()[order])
    weights = model.weights_[order]
    if np.isclose(means[0], means[1]):
        raise ValueError(f"{feature}: GMM component means are indistinguishable")

    def difference(z):
        return weights[0] * norm.pdf(z, means[0], stds[0]) - weights[1] * norm.pdf(z, means[1], stds[1])

    grid = np.linspace(means[0], means[1], 2000)
    diff = difference(grid)
    crossings = np.flatnonzero(diff[:-1] * diff[1:] < 0)
    exact = np.flatnonzero(diff == 0)
    if len(crossings):
        i = crossings[0]
        threshold = brentq(difference, grid[i], grid[i + 1])
        method = "component_crossing"
    elif len(exact):
        threshold = grid[exact[0]]
        method = "component_crossing"
    else:
        threshold = grid[np.argmin(np.abs(diff))]
        method = "minimum_density_difference"
        warnings.warn(f"{feature}: no crossing between component means; using nearest density match")
    return {
        "feature": feature, "n": len(x), "fit_scale": "log1p" if feature == "area" else "original",
        "threshold_fit_scale": float(threshold),
        "threshold": float(np.expm1(threshold) if feature == "area" else threshold),
        "threshold_method": method, "means": means.tolist(),
        "stds": stds.tolist(), "weights": weights.tolist(),
    }


def label_morphology(obs, fits):
    result = obs.copy()
    for feature, (low, high) in LABELS.items():
        values = result[feature]
        labels = pd.Series(pd.NA, index=result.index, dtype="string")
        valid = values.notna()
        labels.loc[valid] = np.where(values[valid] < fits[feature]["threshold"], low, high)
        result[f"{feature}_label"] = pd.Categorical(labels, categories=[low, high], ordered=True)
    return result


def threshold_plot(pooled, fits, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, feature in zip(axes, LABELS):
        values = pooled[feature].dropna().to_numpy()
        x = np.log1p(values) if feature == "area" else values
        fit = fits[feature]
        grid = np.linspace(x.min(), x.max(), 1000)
        ax.hist(x, bins=70, density=True, color="0.7", alpha=0.6)
        for i in range(2):
            ax.plot(grid, fit["weights"][i] * norm.pdf(grid, fit["means"][i], fit["stds"][i]))
        ax.axvline(fit["threshold_fit_scale"], color="black", ls="--")
        ax.set(xlabel=f"log1p({feature})" if feature == "area" else feature,
               ylabel="Density", title=f"{feature}: threshold {fit['threshold']:.3g}")
    fig.tight_layout()
    save_figure(fig, output, "morphology_thresholds")


def expression(data, layer):
    key = "normalized" if layer == "auto" and "normalized" in data.layers else layer
    if key == "auto":
        key = "X"
    if key != "X" and key not in data.layers:
        raise ValueError(f"Missing expression layer: {key}")
    X = data.X if key == "X" else data.layers[key]
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    frame = pd.DataFrame(X, index=data.obs_names, columns=data.var_names)
    return frame.replace([np.inf, -np.inf], np.nan), key


def stars(q):
    if not np.isfinite(q):
        return "NA"
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns"


def marker_comparison(datasets, frames, output):
    """Compare branched and unbranched axons within each dataset."""
    markers = [m for m in next(iter(frames.values())).columns if all(m in f for f in frames.values())]
    if not markers:
        raise ValueError("No common expression markers")
    rows, means, fractions = [], {}, {}
    for name, data in datasets.items():
        labels = data.obs["solidity_label"]
        for group in LABELS["solidity"]:
            values = frames[name].loc[labels == group, markers]
            if values.empty:
                continue
            means[f"{name} {group}"] = values.mean()
            fractions[f"{name} {group}"] = values.gt(0).sum() / values.notna().sum()
        for marker in markers:
            a = frames[name].loc[labels == "Branched", marker].dropna()
            b = frames[name].loc[labels == "Unbranched", marker].dropna()
            rows.append(dict(dataset=name, marker=marker, n_branched=len(a), n_unbranched=len(b),
                             mean_branched=a.mean(), mean_unbranched=b.mean(),
                             delta=b.mean() - a.mean(),
                             p_value=mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) and len(b) else np.nan))
    stats = pd.DataFrame(rows)
    stats["q_value"] = np.nan
    for name in datasets:
        mask = (stats.dataset == name) & stats.p_value.notna()
        if mask.any():
            stats.loc[mask, "q_value"] = multipletests(stats.loc[mask, "p_value"], method="fdr_bh")[1]
    stats.to_csv(output / "marker_comparisons.csv", index=False)
    if len(datasets) > 1:
        delta = stats.pivot(index="marker", columns="dataset", values="delta")
        selected = delta.index[(delta.gt(0).all(axis=1)) | (delta.lt(0).all(axis=1))].tolist()
    else:
        selected = markers
    pd.DataFrame({"marker": selected}).to_csv(output / "dotplot_markers.csv", index=False)
    if not selected or not means:
        return "No eligible markers or morphology groups for the marker dot plot"
    mean = pd.DataFrame(means).loc[selected]
    fraction = pd.DataFrame(fractions).loc[selected]
    z = mean.sub(mean.mean(axis=1), axis=0).div(mean.std(axis=1, ddof=1).replace(0, np.nan), axis=0).fillna(0)
    if len(z) > 2:
        order = leaves_list(linkage(z.to_numpy(), method="ward"))
        z, fraction = z.iloc[order], fraction.iloc[order]
    z.to_csv(output / "dotplot_mean_zscores.csv")
    fraction.to_csv(output / "dotplot_fractions.csv")
    mean.to_csv(output / "dotplot_group_means.csv")
    fig, (ax, sig) = plt.subplots(2, 1, figsize=(max(8, len(z) * .55), 4.8),
                                  gridspec_kw={"height_ratios": [len(z.columns), len(datasets)]}, layout="constrained")
    xx, yy = np.meshgrid(np.arange(len(z)), np.arange(len(z.columns)))
    limit = max(1.0, float(np.abs(z.to_numpy()).max()))
    points = ax.scatter(xx.ravel(), yy.ravel(), s=fraction.T.to_numpy().ravel() * 180,
                        c=z.T.to_numpy().ravel(), cmap="coolwarm", vmin=-limit, vmax=limit,
                        edgecolors="black", linewidths=.25)
    ax.set(xticks=range(len(z)), yticks=range(len(z.columns)), yticklabels=z.columns,
           xlim=(-.6, len(z) - .4), ylim=(len(z.columns) - .5, -.5))
    ax.set_xticklabels(["NRXN1" if x == "rbNRXN1" else x for x in z.index], rotation=90)
    fig.colorbar(points, ax=ax, label="Z-score of group means", shrink=.8)
    handles = [ax.scatter([], [], s=f * 180, color="gray", label=str(int(f * 100))) for f in (.2, .6, 1)]
    ax.legend(handles=handles, title="Value > 0 (%)", loc="upper left", bbox_to_anchor=(1.25, 1), frameon=False)
    sig.set(xlim=ax.get_xlim(), ylim=(len(datasets) - .5, -.5), yticks=range(len(datasets)), yticklabels=list(datasets), xticks=[])
    for i, name in enumerate(datasets):
        q = stats[stats.dataset == name].set_index("marker").q_value
        for j, marker in enumerate(z.index):
            sig.text(j, i, stars(q[marker]), ha="center", va="center")
    sig.set_title("Benjamini–Hochberg adjusted q-values", fontsize=9)
    for spine in sig.spines.values():
        spine.set_visible(False)
    save_figure(fig, output, "morphology_marker_dotplot")
    return "Marker dot plot saved"


def standard_time(values):
    return values.astype("string").str.strip().replace({"A": "t1", "B": "t2", "a": "t1", "b": "t2"})


def fyn_plot(name, data, frame, output):
    if not {"progression", "time_point"}.issubset(data.obs) or "FYN" not in frame:
        return f"{name}: FYN time plot skipped (requires progression, time_point, and FYN)"
    df = data.obs[["progression", "time_point"]].copy()
    df["FYN"] = frame["FYN"]
    df["time_point"] = standard_time(df.time_point)
    df["progression"] = df.progression.astype("string").str.strip().replace({"NonProgressor": "Non-progressor"})
    df = df[df.time_point.isin(["t1", "t2"]) & df.progression.isin(["Non-progressor", "Progressor"])].dropna()
    rows = []
    for progression in ["Non-progressor", "Progressor"]:
        groups = []
        for time in ["t1", "t2"]:
            x = df.loc[(df.progression == progression) & (df.time_point == time), "FYN"].to_numpy()
            if len(x):
                q1, q3 = np.quantile(x, [.25, .75])
                x = x[(x >= q1 - 1.5 * (q3 - q1)) & (x <= q3 + 1.5 * (q3 - q1))]
            groups.append(x)
        if min(map(len, groups)) < 2:
            continue
        fig, ax = plt.subplots(figsize=(3.5, 4))
        for i, x in enumerate(groups):
            if np.ptp(x) > 0:
                parts = ax.violinplot([x], positions=[i], showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(["#D9D9D9", "#4A4A4A"][i])
                    body.set_alpha(1)
        ax.boxplot(groups, positions=[0, 1], widths=.2, showfliers=False)
        p = ttest_ind(*groups, equal_var=False).pvalue
        rows.append(dict(progression=progression, n_t1=len(groups[0]), n_t2=len(groups[1]), p_value=p))
        ax.set(xticks=[0, 1], xticklabels=["t1", "t2"], ylabel="FYN",
               title=f"{name}: {progression}\nWelch p = {p:.3g}")
        save_figure(fig, output, f"{name}_FYN_{progression}")
    if rows:
        pd.DataFrame(rows).to_csv(output / f"{name}_FYN_tests.csv", index=False)
    return f"{name}: {len(rows)} FYN time plots saved"


def composition_plot(name, data, output):
    """Connect group proportions across time, not individual axons."""
    if not {"time_point", "neighborhood"}.issubset(data.obs):
        return f"{name}: composition plot skipped (requires time_point and neighborhood)"
    df = data.obs[["time_point", "neighborhood", "solidity_label"]].dropna().copy()
    df["time_point"] = standard_time(df.time_point)
    df["neighborhood"] = df.neighborhood.astype("string").str.strip()
    df = df[df.time_point.isin(["t1", "t2"])]
    rows = []
    for neighborhood, sub in df.groupby("neighborhood", observed=True):
        counts = pd.crosstab(sub.time_point, sub.solidity_label).reindex(index=["t1", "t2"], columns=LABELS["solidity"], fill_value=0)
        if (counts.sum(axis=1) == 0).any():
            continue
        proportions = counts.div(counts.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(4, 3.6))
        bottom = np.zeros(2)
        for group in LABELS["solidity"]:
            vals = proportions[group].to_numpy()
            ax.add_patch(Polygon([(0.1, bottom[0]), (.9, bottom[1]), (.9, bottom[1]+vals[1]), (.1, bottom[0]+vals[0])], color=COLORS[group], alpha=.4, linewidth=0))
            ax.bar([0, 1], vals, bottom=bottom, width=.2, color=COLORS[group])
            for i, time in enumerate(["t1", "t2"]):
                rows.append(dict(neighborhood=neighborhood, time_point=time, morphology=group, count=int(counts.loc[time, group]), percentage=vals[i]))
                if vals[i] > 8:
                    ax.text(i, bottom[i]+vals[i]/2, f"{vals[i]:.1f}%", ha="center", va="center", color="white", fontsize=8)
            bottom += vals
        ax.set(xticks=[0, 1], xticklabels=["t1", "t2"], ylabel="Axons (%)", ylim=(0, 100), title=f"{name}: {neighborhood}")
        ax.legend(handles=[Patch(color=COLORS[g], label=g) for g in LABELS["solidity"]], loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, frameon=False)
        safe = "".join(c if c.isalnum() else "_" for c in str(neighborhood))
        save_figure(fig, output, f"{name}_composition_{safe}")
    if rows:
        pd.DataFrame(rows).to_csv(output / f"{name}_composition.csv", index=False)
    return f"{name}: {len(rows)//4} neighborhood composition plots saved"


def run(rp, as_path, output, layer="auto"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datasets, frames, layers, status = {}, {}, {}, []
    for name, path in [("RP", rp), ("AS", as_path)]:
        if path is None:
            continue
        data = ad.read_h5ad(path)
        if not data.var_names.is_unique:
            raise ValueError(f"{name}: marker names must be unique")
        data.obs_names_make_unique()
        data.obs = morphology(data.obs)
        datasets[name] = data
        frames[name], layers[name] = expression(data, layer)
    pooled = pd.concat([data.obs[list(LABELS)] for data in datasets.values()])
    fits = {feature: fit_threshold(pooled[feature], feature) for feature in LABELS}
    (output / "morphology_thresholds.json").write_text(json.dumps(fits, indent=2))
    threshold_plot(pooled, fits, output)
    quality = []
    for name, data in datasets.items():
        data.obs = label_morphology(data.obs, fits)
        data.obs.to_csv(output / f"{name}_morphology.csv", index_label="axon_id")
        for feature in LABELS:
            quality.append(dict(dataset=name, feature=feature, total=len(data), valid=int(data.obs[feature].notna().sum()), invalid=int(data.obs[feature].isna().sum())))
        status.extend([fyn_plot(name, data, frames[name], output), composition_plot(name, data, output)])
    pd.DataFrame(quality).to_csv(output / "measurement_quality.csv", index=False)
    status.append(marker_comparison(datasets, frames, output))
    (output / "run_summary.json").write_text(json.dumps(dict(inputs={"RP": str(rp), "AS": str(as_path) if as_path else None}, expression_layers=layers, threshold_datasets=list(datasets), status=status), indent=2))
    print(f"Results saved in {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rp", type=Path, default=RP_PATH)
    parser.add_argument("--as", dest="as_path", type=Path, default=AS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--expression-layer", default="auto", help="auto, X, or a layer name")
    args = parser.parse_args()
    run(args.rp, args.as_path, args.output, args.expression_layer)


if __name__ == "__main__":
    main()
