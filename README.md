# Replication Code for "Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments."

This code is separated into six parts:

- run_data_construction_1.py can be run using Python 3
- run_data_construction_2.py is run through SLURM using run_data_construction_2.sh, and is optimized to be parallelized on a cluster. Instructions to run it on the MIT cluster are included below.
- run_data_construction_3.py can be run using Python 3
- run_analysis.py can be run using Python 3
- analysis/main.do can be run using Stata
- run_bootstrap.py can be run through SLURM using run_bootstrap.sh and is optimized to be parallelized on a cluster

## Instructions for how to access MIT cluster:
1. Transfer necessary files to the cluster via SCP
`
scp -r -C "vbperal@eofe9.mit.edu:/nfs/home2/vbperal/research/ystar-dynamics/data/original/file_name" "/Users/vbp/Dropbox (Personal)/research/research-data/ystar-data/data/original/file_name"
`
The following files must be transferred:
    - clean/leasehold_flats_lw.p
    - working/merged_hmlr_hedonics.p

2. Run sbatch run_data_construction_2.sh

3. Transfer the necessary files back to main computer. The following files must be transferred:
    - working/residuals*.p
    - working/rsi.p
    - working/rsi_flip.p
    - working/rsi_hedonics.p
    - working/rsi_full.p
    - working/rsi_bmn.p
    - working/rsi_yearly.p
    - working/rsi_postcode.p
    - working/hedonics_variations/*
    - working/bayes_bootstrap/*

