from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from figure_runner import PANEL_FILES, run_panel

AXON_PANELS = ("figure_2_axon_clustering", "figure_3_axon_morphology")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the manuscript figure panels in this repository.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the H5AD inputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=(*AXON_PANELS, *PANEL_FILES),
        default=(*AXON_PANELS, *PANEL_FILES),
        help="Panel script names to run; defaults to every panel listed in README.md.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, out = args.data_dir, args.output_dir
    axon_arguments = {
        "figure_2_axon_clustering": ["--input", data / "AS_Neuronal_data_axons.h5ad",
                                     "--output-dir", out / "axon_figures"],
        "figure_3_axon_morphology": ["--rp", data / "RP_Neuronal_data_axons.h5ad",
                                     "--as", data / "AS_Neuronal_data_axons.h5ad",
                                     "--as-immune", data / "AS_Immune_data_axons.h5ad",
                                     "--neighborhoods", data / "AS_cellular_neighborhoods.h5ad",
                                     "--output", out / "morphology_results"],
    }
    for panel in args.panels:
        print(f"Running {panel}...", flush=True)
        if panel in AXON_PANELS:
            script = Path(__file__).resolve().parent / f"{panel}.py"
            subprocess.run([sys.executable, str(script), *map(str, axon_arguments[panel])], check=True)
        else:
            run_panel(panel, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
