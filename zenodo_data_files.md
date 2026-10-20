Zenodo H5AD data files for reproducing the figure panels in this repository are available at https://doi.org/10.5281/zenodo.23482995.

Use the descriptive filenames below in the Zenodo record. 

To perform axon clustering and morphology analysis, run 'figure_2_axon_clustering.py' and 'figure_3_axon_morphology.py', respectively, on RP_Neuronal_data_axons.h5ad and AS_Neuronal_data_axons.h5ad.

For the simplest workflow, download all files into one directory and run
`figure_code/reproduce_all.py --data-dir /path/to/downloaded-files` from the
code repository root.

Cellular-neighborhood datasets:
  AS_cellular_neighborhoods.h5ad
  RP_cellular_neighborhoods.h5ad
  AS_T_cell_marker_cell_types.h5ad
  RP_T_cell_marker_cell_types.h5ad

PPA cell-type datasets for Figure 5d:
  AS_PPA_cell_type_CXCR4_CD44_FYN_axon.h5ad
  AS_PPA_cell_type_NRXN1_MAOA_axon.h5ad
  AS_PPA_cell_type_TH_axon.h5ad

PPA T-cell marker datasets for Figures 6a, 6c-d, and 6e-f:
  AS_PPA_T_cell_marker_AR_CXCR4_CD44_FYN_axon.h5ad
  AS_PPA_T_cell_marker_AR_NRXN1_MAOA_axon.h5ad
  AS_PPA_T_cell_marker_Ki67_CXCR4_CD44_FYN_axon.h5ad
  AS_PPA_T_cell_marker_Ki67_NRXN1_MAOA_axon.h5ad
  AS_PPA_T_cell_marker_GZMB_CXCR4_CD44_FYN_axon.h5ad
  AS_PPA_T_cell_marker_GZMB_NRXN1_MAOA_axon.h5ad
  AS_PPA_T_cell_marker_PD1_CXCR4_CD44_FYN_axon.h5ad
  AS_PPA_T_cell_marker_PD1_NRXN1_MAOA_axon.h5ad
