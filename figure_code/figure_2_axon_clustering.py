"""Axon clustering across marker transformations and Leiden resolutions."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse
import scanpy as sc


DATA_PATH = Path("data/RP_Neuronal_data_axons.h5ad")
# DATA_PATH = Path("data/AS_Neuronal_data_axons.h5ad")
OUTPUT_DIR = Path("axon_figures") / DATA_PATH.stem

RESOLUTIONS = [round(i / 10, 1) for i in range(1, 11)]
SELECTED_RESOLUTION = 0.3
SELECTED_TRANSFORMATION = "Raw"  # Raw, Z-score, Log1p, Log1p + Z-score
LOG_POLICY = "shift_nonnegative"  # or "direct"
USE_EXISTING_LABELS = False
SAVE_RESOLUTION_COMPARISON = True
SEED = 42
N_NEIGHBORS = 15

CLUSTER_NAMES = {}  # {"0": "marker-based label"}

CLUSTER_MARKERS = [
    "PSANCAM", "NOTCH1", "MAOA", "TH", "VIM", "CXCR4", "CMYC",
    "CD90", "CD44", "msNLGN1", "rbNRXN1", "INA", "FYN",
]
DOT_MARKERS = ["rbNRXN1", "CD44", "CXCR4", "FYN", "TH", "MAOA", "CD90", "INA"]
DOT_LABELS = ["NRXN1", "CD44", "CXCR4", "FYN", "TH", "MAOA", "CD90", "INA"]


COLORS = {
    "CD90+ / INA+": "#FFD166", "CD90+": "#FFD166",
    "CXCR4+ / CD44+ / FYN+": "#26547C",
    "CD44+ / FYN+": "#26547C", "CXCR4+ / FYN+": "#759AB8",
    "NRXN1+ / MAOA+": "#06D6A0", "TH+": "#EF476F",
}
ORDER = list(COLORS)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def load_data(path):
    source = sc.read_h5ad(path)
    missing = sorted(set(CLUSTER_MARKERS + DOT_MARKERS) - set(source.var_names))
    if missing or not source.var_names.is_unique:
        raise ValueError(f"Marker names must be unique and include: {missing}")
    data = sc.AnnData(
        X=source[:, CLUSTER_MARKERS].X.copy(), obs=source.obs.copy(),
        var=pd.DataFrame(index=CLUSTER_MARKERS),
    )
    data.obs_names_make_unique()
    values = data.X.data if sparse.issparse(data.X) else data.X
    if data.n_obs < 3 or not np.isfinite(values).all():
        raise ValueError("Expected at least three axons with finite marker values")
    return data


def marker_zscore(X):
    """Standardize each marker; constant markers become zero."""
    std = X.std(axis=0, ddof=0)
    return (X - X.mean(axis=0)) / np.where(std == 0, 1.0, std)

def transform_data(data):
    """Build all four versions from the same input matrix."""
    raw = data.X.toarray() if sparse.issparse(data.X) else np.asarray(data.X)
    raw = np.array(raw, dtype=float, copy=True)
    if LOG_POLICY == "shift_nonnegative":
        offsets = np.maximum(0, -raw.min(axis=0))
    elif LOG_POLICY == "direct":
        if np.any(raw <= -1):
            raise ValueError("Direct log1p requires values > -1; use shift_nonnegative for this input")
        offsets = np.zeros(raw.shape[1])
    else:
        raise ValueError("LOG_POLICY must be 'shift_nonnegative' or 'direct'")
    logged = np.log1p(raw + offsets)
    matrices = {"Raw": raw, "Z-score": marker_zscore(raw),
                "Log1p": logged, "Log1p + Z-score": marker_zscore(logged)}
    versions = {}
    for name, values in matrices.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid values after {name}")
        versions[name] = sc.AnnData(values.copy(), obs=data.obs.copy(), var=data.var.copy())
    return versions, offsets


def cluster_versions(versions):
    """Use one graph and UMAP per transformation, shared across resolutions."""
    counts = []
    for name, data in versions.items():
        sc.pp.neighbors(data, use_rep="X", n_neighbors=min(N_NEIGHBORS, data.n_obs - 1), random_state=SEED)
        sc.tl.umap(data, random_state=SEED)
        for resolution in RESOLUTIONS:
            key = f"leiden_new_{resolution:.1f}"
            sc.tl.leiden(data, resolution=resolution, key_added=key, random_state=SEED,
                         directed=False, n_iterations=2, flavor="igraph")
            counts.append({"transformation": name, "resolution": resolution,
                           "n_clusters": data.obs[key].nunique()})
    return pd.DataFrame(counts)


def umap_figure(data, groupby, title=""):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    categories = list(data.obs[groupby].cat.categories)
    palette = [COLORS.get(name, plt.get_cmap("tab20")(i % 20))
               for i, name in enumerate(categories)]
    sc.pl.umap(
        data, color=groupby, palette=palette, size=6, frameon=False,
        title=title, ax=ax, show=False, legend_loc="right margin",
    )
    ax.set_aspect("equal", adjustable="box")
    for endpoint in [(0.20, 0.04), (0.04, 0.20)]:
        ax.annotate("", xy=endpoint, xytext=(0.04, 0.04), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 1})
    ax.text(0.12, 0.00, "UMAP1", transform=ax.transAxes, ha="center", fontsize=8)
    ax.text(0.00, 0.12, "UMAP2", transform=ax.transAxes, va="center", rotation=90, fontsize=8)
    return fig

def dotplot_figure(data, groupby, title=""):
    dp = sc.pl.dotplot(
        data, DOT_MARKERS, groupby=groupby, use_raw=False,
        standard_scale="var", expression_cutoff=0, mean_only_expressed=False,
        cmap="coolwarm", dot_min=0, dot_max=1, smallest_dot=0,
        figsize=(9, max(3.5, 0.30 * data.obs[groupby].nunique() + 1.8)),
        colorbar_title="Scaled mean expression\nin group",
        size_title="Fraction of axons\nwith value > 0 (%)",
        show=False, return_fig=True,
    )
    dp.style(cmap="coolwarm", largest_dot=180, dot_edge_color="black", dot_edge_lw=0.25)
    dp.make_figure()
    ax = dp.get_axes()["mainplot_ax"]
    ax.set_xticklabels(DOT_LABELS, rotation=90)
    ax.set_title(title)
    return dp.fig

def save_figure(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{name}.{extension}", dpi=300, bbox_inches="tight")

def select_groups(data):
    """Apply saved annotations or names for the selected clustering."""
    if USE_EXISTING_LABELS:
        if "axontype" not in data.obs:
            raise ValueError("No axontype labels found. Set USE_EXISTING_LABELS = False to plot new clusters.")
        labels = data.obs["axontype"].astype("string").str.strip().fillna("Unannotated")
        labels = labels.replace({"CD90+ / INA": "CD90+ / INA+"})
        present = labels.unique().tolist()
        order = [x for x in ORDER if x in present] + sorted(set(present) - set(ORDER))
    else:
        key = f"leiden_new_{SELECTED_RESOLUTION:.1f}"
        if key not in data.obs:
            raise ValueError(f"{key} was not computed; choose one of {RESOLUTIONS}.")
        ids = data.obs[key].astype(str)
        labels = ids.map(lambda x: CLUSTER_NAMES.get(x, f"Cluster {x}"))
        order = list(dict.fromkeys(CLUSTER_NAMES.get(x, f"Cluster {x}")
                                  for x in data.obs[key].cat.categories))

    data.obs["figure_group"] = pd.Categorical(labels, categories=order, ordered=True)


VERSION_FILES = {"Raw": "raw", "Z-score": "zscore", "Log1p": "log1p", "Log1p + Z-score": "log1p_zscore"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(DATA_PATH)
    versions, offsets = transform_data(data)
    pd.DataFrame({"marker": data.var_names, "offset_before_log1p": offsets}).to_csv(
        OUTPUT_DIR / "log1p_offsets.csv", index=False)
    cluster_versions(versions).to_csv(OUTPUT_DIR / "cluster_counts.csv", index=False)
    log_name = "shifted log1p" if np.any(offsets > 0) else "log1p"
    titles = {"Raw": "Raw", "Z-score": "Z-score", "Log1p": log_name,
              "Log1p + Z-score": log_name + " + Z-score"}

    if SAVE_RESOLUTION_COMPARISON:
        for name, data in versions.items():
            with PdfPages(OUTPUT_DIR / f"{VERSION_FILES[name]}_resolution_comparison.pdf") as pdf:
                for resolution in RESOLUTIONS:
                    key = f"leiden_new_{resolution:.1f}"
                    title = f"{titles[name]} | Leiden resolution {resolution:.1f}"
                    for plot in (umap_figure, dotplot_figure):
                        fig = plot(data, key, title)
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

    if SELECTED_TRANSFORMATION not in versions:
        raise ValueError(f"Choose one of {list(versions)}")
    selected = versions[SELECTED_TRANSFORMATION].copy()
    select_groups(selected)
    grouping = "Existing axon types" if USE_EXISTING_LABELS else f"Leiden {SELECTED_RESOLUTION:.1f}"
    suffix = "existing_labels" if USE_EXISTING_LABELS else f"leiden_{SELECTED_RESOLUTION:.1f}"
    prefix = f"{VERSION_FILES[SELECTED_TRANSFORMATION]}_{suffix}"
    for plot, name in [(umap_figure, "a_umap"), (dotplot_figure, "b_marker_dotplot")]:
        fig = plot(selected, "figure_group", f"{titles[SELECTED_TRANSFORMATION]} | {grouping}")
        save_figure(fig, f"{prefix}_{name}")
        plt.show()
        plt.close(fig)
    print(f"Figures saved in {OUTPUT_DIR.resolve()}")
    return versions


if __name__ == "__main__":
    versions = main()
