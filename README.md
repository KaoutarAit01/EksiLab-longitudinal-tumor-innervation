# Longitudinal Tumor Innervation

Spatial analysis pipeline for quantifying axon morphology, molecular phenotypes, and neuroimmune interactions during longitudinal prostate cancer progression.

This repository includes code accompanying Ak, Ait-Ahmad et al., “Longitudinal Analysis of Tumor Innervation Identifies Dynamic Neuroimmune Niches in Prostate Cancer,” submitted to *Nature Cancer*. The manuscript figure reproduction pipeline is organized under `figure_code/`. Additional analysis scripts and documentation can be added alongside it.

## Reproduce manuscript figure panels

The H5AD inputs needed for the figure panels are listed in
[zenodo_data_files.md](zenodo_data_files.md). Download the listed files into one
directory before running the pipeline.

Use Python 3.12. Create the environment once, then run all panels from the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python figure_code/reproduce_all.py --data-dir /path/to/h5ad-files
```

Figures and regenerated progression statistics are written to `figures/`. To
reproduce one panel, run its script directly, for example:

```bash
python figure_code/figure_2_axon_clustering.py \
  --input /path/to/h5ad-files/AS_Neuronal_data_axons.h5ad \
  --output-dir figures/axon_figures

python figure_code/figure_3_axon_morphology.py \
  --rp /path/to/h5ad-files/RP_Neuronal_data_axons.h5ad \
  --as /path/to/h5ad-files/AS_Neuronal_data_axons.h5ad \
  --as-immune /path/to/h5ad-files/AS_Immune_data_axons.h5ad \
  --neighborhoods /path/to/h5ad-files/AS_cellular_neighborhoods.h5ad \
  --output figures/morphology_results

python figure_code/figure_5d_local_redistribution.py \
  --data-dir /path/to/h5ad-files \
  --output-dir figures
```

## Figure panels included
- `figure_2_axon_clustering.py` (Figure 2a-b)
- `figure_3_axon_morphology.py` (Figure 3b-d)
- `figure_4a_cn_heatmap.py` (Figure 4a)
- `figure_4bc_whole_tissue.py` (Figure 4b–c)
- `figure_4e_window_density.py` (Figure 4e)
- `figure_5b_cn_progression.py` (Figure 5b)
- `figure_5d_local_redistribution.py` (Figure 5d)
- `figure_6a_pd1_composition.py` (Figure 6a)
- `figure_6b_whole_tissue_markers.py` (Figure 6b)
- `figure_6cd_axon_marker_fractions.py` (Figure 6c–d)
- `figure_6ef_progression_effects.py` (Figure 6e–f)

See [zenodo_data_files.md](zenodo_data_files.md) for the complete H5AD input
list. Install dependencies from `requirements.txt` before running.

## System requirements
- Operating system: Figures 2–3 were tested on Linux (Red Hat Enterprise Linux 9.6, x86_64). Other operating systems with Python 3.12 may work but have not been tested for those figures. macOS 15.8.1 (x86_64) was used to run Figures 4–6.
- Software:  Figures 2–3 were generated with Python 3.12.5; Figures 4–6 with Python 3.12.11. Dependencies are listed in `requirements.txt`.
- Hardware: For Figures 2–3, a standard desktop computer with at least 4 CPU cores and 16 GB RAM is recommended. Figures 4–6 ran on a MacBook Pro with an Intel Core i5 at 1.4 GHz, 4 cores, and 8 GB RAM. 
- Disk space: About 6 GB for the H5AD inputs.
- Time: Installation takes a few minutes. Figures 4–6 took about 6 minutes 35 seconds on the system above. Reproducing all panels may take 30–90 minutes.
- Demo: Run the Figure 3 command above with only `--rp /path/to/h5ad-files/RP_Neuronal_data_axons.h5ad --as "" --as-immune ""`. On the 2.7 MB dataset, it finishes in under a minute and writes `morphology_thresholds.pdf` and `morphology_marker_dotplot.pdf`.

## License
This code is released under the MIT License; see `LICENSE`.
