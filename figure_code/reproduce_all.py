from __future__ import annotations

import argparse
from pathlib import Path

from figure_runner import PANEL_FILES, run_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the manuscript figure panels in this repository.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the H5AD inputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=tuple(PANEL_FILES),
        default=tuple(PANEL_FILES),
        help="Panel script names to run; defaults to every panel listed in README.md.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for panel in args.panels:
        print(f"Running {panel}...", flush=True)
        run_panel(panel, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
