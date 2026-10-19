from __future__ import annotations

import os
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
WORKFLOW = ROOT / "workflow"
H5AD_FILES = {
    "cn_as": "AS_cellular_neighborhoods.h5ad",
    "cn_rp": "RP_cellular_neighborhoods.h5ad",
    "marker_as": "AS_T_cell_marker_cell_types.h5ad",
    "marker_rp": "RP_T_cell_marker_cell_types.h5ad",
}
AXON_TAGS = (
    "CXCR4pos___CD44pos___FYNpos_Axon",
    "NRXN1pos___MAOApos_Axon",
)
AXON_RELEASE_LABELS = {
    "CXCR4pos___CD44pos___FYNpos_Axon": "CXCR4_CD44_FYN",
    "NRXN1pos___MAOApos_Axon": "NRXN1_MAOA",
}
PPA_FILES = {
    f"{marker}_{axon}": f"AS_PPA_T_cell_marker_{marker}_{AXON_RELEASE_LABELS[axon]}_axon.h5ad"
    for marker in ("AR", "Ki67", "GZMB", "PD1")
    for axon in AXON_TAGS
}
CELL_PPA_FILES = {
    f"radial_{tag}": f"AS_PPA_cell_type_{release_label}_axon.h5ad"
    for tag, release_label in (
        ("CXCR4pos_CD44pos_FYNpos_Axon", "CXCR4_CD44_FYN"),
        ("NRXN1pos_MAOApos_Axon", "NRXN1_MAOA"),
        ("THpos_Axon", "TH"),
    )
}
INPUT_FILES = {**H5AD_FILES, **PPA_FILES, **CELL_PPA_FILES}
PANEL_FILES = {
    "figure_4a_cn_heatmap": ("cn_as", "cn_rp"),
    "figure_4bc_whole_tissue": ("cn_as", "cn_rp"),
    "figure_4e_window_density": ("cn_as", "cn_rp"),
    "figure_5b_cn_progression": ("cn_as",),
    "figure_5d_local_redistribution": tuple(CELL_PPA_FILES),
    "figure_6a_pd1_composition": tuple(f"PD1_{axon}" for axon in AXON_TAGS),
    "figure_6b_whole_tissue_markers": ("marker_as", "marker_rp"),
    "figure_6cd_axon_marker_fractions": tuple(PPA_FILES),
    "figure_6ef_progression_effects": tuple(PPA_FILES),
}
FIGURE_OUTPUTS = {
    "figure_4a_cn_heatmap": ("fig_cn_cell_type_heatmap_combined.pdf",),
    "figure_4bc_whole_tissue": (
        "fig_whole_tissue_grade_composition_separate_panel.pdf",
        "fig_whole_tissue_grade_region_immune_fraction_separate_panel.pdf",
        "fig_whole_tissue_grade_immune_density_separate_panel.pdf",
    ),
    "figure_4e_window_density": ("fig_axon_immune_target_window_density_background_vs_axon_as_rp_combined.pdf",),
    "figure_5b_cn_progression": ("AS_progression_paper_ready.pdf",),
    "figure_5d_local_redistribution": ("fig_progression_local_redistribution.pdf",),
    "figure_6a_pd1_composition": ("PD1_raw_composition_donuts.pdf",),
    "figure_6b_whole_tissue_markers": (
        "fig_whole_tissue_tcell_marker_grade_whole_sample_fraction_separate_panel.pdf",
        "fig_whole_tissue_tcell_marker_grade_region_fraction_separate_panel.pdf",
        "fig_whole_tissue_tcell_marker_grade_positive_tcell_density_separate_panel.pdf",
        "fig_whole_tissue_tcell_marker_progression_whole_sample_fraction_separate_panel.pdf",
        "fig_whole_tissue_tcell_marker_progression_region_fraction_separate_panel.pdf",
        "fig_whole_tissue_tcell_marker_progression_positive_tcell_density_separate_panel.pdf",
    ),
    "figure_6cd_axon_marker_fractions": (
        "PD1_positive_fraction_barplot.pdf",
        "GZMB_positive_fraction_barplot.pdf",
    ),
    "figure_6ef_progression_effects": (
        "fig_progression_marker_specific_changes_pd1_fraction_separate_panel.pdf",
        "fig_progression_marker_specific_changes_abundance_separate_panel.pdf",
    ),
}
PANEL_TABLE_OUTPUTS = {
    "figure_5d_local_redistribution": ("progression_radial_statistics.csv",),
    "figure_6ef_progression_effects": ("progression_marker_statistics.csv",),
}


def locate_input(data_dir: Path, name: str) -> Path | None:
    path = data_dir / name
    return path if path.is_file() else None


def panel_commands(panel: str, inputs: dict[str, Path], statistics_target: Path) -> list[list[str]]:
    final = "manuscript final figures/figures"
    tables = "manuscript final figures/tables"
    grade_tables = "manuscript_panels_results/whole_tissue_grade/combined"
    marker_grade_tables = "manuscript_panels_results/whole_tissue_tcell_marker_grade/combined"
    marker_progression_tables = "manuscript_panels_results/whole_tissue_tcell_marker_progression"
    cn_tables = "results/cn_neighborhood_enrichment_progression_as"
    cn_as = str(inputs.get(H5AD_FILES["cn_as"], ""))
    cn_rp = str(inputs.get(H5AD_FILES["cn_rp"], ""))
    marker_as = str(inputs.get(H5AD_FILES["marker_as"], ""))
    marker_rp = str(inputs.get(H5AD_FILES["marker_rp"], ""))
    ppa_files = [inputs[INPUT_FILES[name]] for name in PPA_FILES if INPUT_FILES[name] in inputs]
    ppa_dir = str(ppa_files[0].parent) if ppa_files else ""
    radial_files = [inputs[INPUT_FILES[name]] for name in CELL_PPA_FILES if INPUT_FILES[name] in inputs]
    if panel == "figure_4a_cn_heatmap":
        return [["plot_cn_cell_type_heatmap.py", "--h5ad", cn_as, cn_rp, "--outdir", final, "--table-dir", tables]]
    if panel == "figure_4bc_whole_tissue":
        return [
            ["analyze_whole_tissue_progression.py", "--analysis", "grade", "--h5ad", cn_as, cn_rp,
             "--outdir", grade_tables],
            ["plot_whole_tissue_grade.py", "--table-dir", f"{grade_tables}/tables", "--figure-dir", final],
        ]
    if panel == "figure_4e_window_density":
        return [["analyze_axon_window_enrichment.py", "--h5ad", cn_as, cn_rp,
                 "--outdir", "results/axon_window_enrichment"]]
    if panel == "figure_5b_cn_progression":
        return [
            ["analyze_cn_progression.py", "--group-by", "progression", "--h5ad", cn_as,
             "--outdir", cn_tables],
            ["plot_cn_progression.py", "--coefficients", f"{cn_tables}/tables/notebook_method_coefficients_as.csv",
             "--pvalues", f"{cn_tables}/tables/notebook_method_pvalues_as.csv", "--figdir", final],
        ]
    if panel == "figure_5d_local_redistribution":
        return [
            ["analyze_progression_panel_statistics.py", "--mode", "radial", "--cell-ppa",
             *[str(path) for path in radial_files], "--outdir", str(statistics_target.parent)],
            ["plot_local_redistribution.py", "--master", str(statistics_target), "--outdir", final],
        ]
    if panel == "figure_6a_pd1_composition":
        return [
            ["extract_marker_metrics.py", "--input-dir", ppa_dir, "--outdir", "significance_axons_immune_cells",
             "--markers", "PD1"],
            ["plot_pd1_composition.py"],
        ]
    if panel == "figure_6b_whole_tissue_markers":
        return [
            ["analyze_tcell_marker_positivity.py", "--analysis", "grade", "--h5ad", marker_as, marker_rp,
             "--outdir", marker_grade_tables],
            ["analyze_tcell_marker_positivity.py", "--analysis", "progression", "--h5ad", marker_as,
             "--outdir", marker_progression_tables],
            ["plot_tcell_marker_grade.py", "--table-dir", f"{marker_grade_tables}/tables", "--figure-dir", final],
            ["plot_tcell_marker_progression.py", "--table-dir", f"{marker_progression_tables}/tables", "--figure-dir", final],
        ]
    if panel == "figure_6cd_axon_marker_fractions":
        return [
            ["extract_marker_metrics.py", "--input-dir", ppa_dir, "--outdir", "significance_axons_immune_cells",
             "--markers", "PD1", "GZMB"],
            ["plot_pd1_progression_barplot.py", "--marker", "PD1"],
            ["plot_pd1_progression_barplot.py", "--marker", "GZMB"],
        ]
    if panel == "figure_6ef_progression_effects":
        return [
            ["analyze_progression_panel_statistics.py", "--mode", "markers", "--marker-ppa",
             *[str(path) for path in ppa_files], "--outdir", str(statistics_target.parent)],
            ["plot_progression_marker_effects.py", "--master", str(statistics_target), "--outdir", final],
        ]
    raise ValueError(f"Unknown panel: {panel}")


def run_panel(panel: str, data_dir: Path, output_dir: Path) -> None:
    if panel not in PANEL_FILES:
        raise ValueError(f"Unknown panel: {panel}")
    data_dir = data_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    required_names = [
        INPUT_FILES[name] for name in PANEL_FILES[panel]
    ]
    inputs = {name: locate_input(data_dir, name) for name in required_names}
    missing = [name for name, path in inputs.items() if path is None]
    if missing:
        raise ValueError("Missing required inputs in data directory:\n  " + "\n  ".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{panel}_") as temp:
        work = Path(temp)
        environment = os.environ.copy()
        environment.update(
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(work / "matplotlib"),
            PYTHONHASHSEED="0",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
        )
        statistics_target = work / "manuscript_panels_results/tables/focused_fdr_master_analysis_table.csv"
        statistics_target.parent.mkdir(parents=True, exist_ok=True)
        selected_commands = []

        selected_commands.extend(panel_commands(panel, inputs, statistics_target))

        for command in selected_commands:
            log = work / f"{Path(command[0]).stem}.log"
            with log.open("w") as handle:
                result = subprocess.run(
                    [sys.executable, str(WORKFLOW / command[0]), *command[1:]],
                    cwd=work,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
                raise RuntimeError(f"{command[0]} failed:\n{tail}")

        for filename in FIGURE_OUTPUTS[panel]:
            matches = list(work.rglob(filename))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {filename}, found {len(matches)}")
            source = matches[0]
            for suffix in (".pdf", ".png", ".svg"):
                asset = source.with_suffix(suffix)
                if asset.is_file():
                    shutil.copyfile(asset, output_dir / asset.name)
        for filename in PANEL_TABLE_OUTPUTS.get(panel, ()):
            matches = list(work.rglob(filename))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {filename}, found {len(matches)}")
            shutil.copyfile(matches[0], output_dir / filename)
    print(f"Created {len(FIGURE_OUTPUTS[panel])} figure file(s) in {output_dir}")


def main(panel: str, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Flat directory of downloaded Zenodo inputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    run_panel(panel, args.data_dir, args.output_dir)
