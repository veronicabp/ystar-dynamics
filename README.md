# Replication Code for "Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments."

## Data Availability and Provenance Statements

All newly assembled data on UK leasehold extensions (2000–2024) are publicly available and updated monthly via our project website:
- **Lease extension dataset**: https://sites.google.com/view/ystar-dynamics/home  

Provenance:
1. **Closed lease titles** (pre-extension) purchased from His Majesty’s Land Registry (HMLR) archives.  
2. **Open lease titles** and transaction prices matched via HMLR’s public Lease Register and Price Paid Data.  
3. **Rightmove & Zoopla listings** (2006–present) accessed under academic license from Rightmove, Inc. and Urban Big Data Centre.  
4. **Geospatial and climate risk layers** obtained from UK Environment Agency and British Geological Survey.  

For each data source, see Appendix A of the paper for full download and preprocessing steps.



### Statement about Rights
- **Proprietary inputs**: Rightmove and Zoopla data are used under license; users must obtain their own access to reproduce those portions of the analysis.  



## Dataset list

1. **HMLR Closed Leases**  
   - **What**: Historical “closed” lease titles overwritten by extensions.  
   - **How**: Bulk purchase from HMLR; matched to open leases via identifier and fuzzy address rules.  
   - **Coverage**: All UK lease extensions recorded before May 2023; subsequently updated monthly.  

2. **HMLR Open Leases & Price Paid Data**  
   - **What**: Current remaining-term lease titles and all flat transaction prices since 1995.  
   - **How**: Public downloads from the HMLR website; merged on address/postcode.  
   - **Coverage**: All conveyances in England & Wales.

3. **Rightmove Listing Data**  
   - **What**: Detailed hedonic attributes (bedrooms, area, condition) and rental quotes.  
   - **How**: Academic license access; scraped and timestamped by property ID.  
   - **Coverage**: 2006–present; matched to ~80% of transactions.

4. **Zoopla Listing Data**  
   - **What**: Complimentary listing data on bedrooms, receptions, floors, rent.  
   - **How**: Provided by Urban Big Data Centre (UBDC) under research-use agreement.  
   - **Coverage**: 2010–present; matched to ~75% of transactions.

5. **Geospatial & Risk Data**  
   - **Flood risk**: UK Environment Agency flood-zone shapefiles.  
   - **Subsidence risk**: BGS GeoClimate projections.  
   - **How**: Aggregated to Local Authority level via GIS union and area-weighting.



### Details on monthly updates to the data

Updates to HMLR data can be found at the following links: 
1. Lease Data: https://use-land-property-data.service.gov.uk/datasets/leases
2. Price Data: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads

## Computation requirements

All Python packages can be found in the anaconda environemnt, `environment.yml`

## Description of programs and code

### 1. `data_construction_pt1.py`  
**Purpose:** Ingest and clean basic input data.  
**Main Steps:**
- **`get_boe_interest_rates(data_folder)`**  
  Downloads and formats Bank of England interest‐rate series.  
- **`clean_price_paid(data_folder)`**  
  Cleans and standardizes the Price Paid dataset.  
- **`clean_leases(data_folder)`**  
  Cleans lease transaction records.  
- **`convert_hedonics_data(data_folder)`**  
  Converts raw hedonic‐price inputs into analysis-ready form.  
- **`merge_hmlr(data_folder)`**  
  Merges Land Registry (HMLR) data into the cleaned dataset.  

### 2. `data_construction_pt2.py`  
**Purpose:** Construct the core Repeat Sales Index (RSI) and related controls.  
**Main Steps:**
- **`get_residuals(data_folder)`**  
  Calculates model residuals used in RSI weights.  
- **`construct_rsi(data_folder)`**  
  Builds the main RSI series from cleaned data.  
- **`get_rsi_hedonic_variations(data_folder)`**  
  Computes hedonic‐variation adjustments to the RSI.  
- **`construct_restrictive_controls(data_folder)`**  
  Generates control measures with same transaction times as treated variable.  

### 3. `data_construction_pt3.py`  
**Purpose:** Finalize experimental samples, assemble additional datasets, and export final outputs.  
**Main Steps:**
- **`calculate_hazard_rate(data_folder)`**  
  Estimates hazard-rate measures for lease extensions.  
- **`run_create_experiments(data_folder)`**  
  Defines and constructs the experimental samples for analysis.  
- **`make_additional_datasets(data_folder)`**  
  Builds ancillary datasets.  
- **`construct_rent_rsi(data_folder)`**  
  Constructs a rent-based RSI series.  
- **`combine_ashe_data(data_folder)`**  
  Merges ASHE (Annual Survey of Hours and Earnings) data.  
- **`expand_hilber_data(data_folder)`**  
  Extendes Hilber et al. data to present for cross-sectional use.  
- **`get_cross_sectional_estimates(data_folder)`**  
  Computes cross-sectional estimates of y-star.  
- **`get_hedonics_variations(data_folder)`**  
  Runs variations of y-star measure with different hedonic controls.  
- **`output_dta(data_folder)`**  
  Exports all cleaned and merged datasets to Stata `.dta` files.  

## Instructions to replicators

This code is separated into six parts:

- run_data_construction_1.py can be run using Python 3
- run_data_construction_2.py is run through SLURM using run_data_construction_2.sh, and is optimized to be parallelized on a cluster. Instructions to run it on the MIT cluster are included below.
- run_data_construction_3.py can be run using Python 3
- run_analysis.py can be run using Python 3
- analysis/main.do can be run using Stata
- run_bootstrap.py can be run through SLURM using run_bootstrap.sh and is optimized to be parallelized on a cluster

Note: This project uses data from Rightmove and Zoopla that is already cleaned. The code to clean this data can be found in this Github repository: https://github.com/veronicabp/uk-property-data. The cleaned data is added to the folder `/working/hedonics/`.

### Instructions for how to access MIT cluster:
1. Transfer necessary files to the cluster via SCP
`
scp -r -C "USERNAME@eofe9.mit.edu:/{CLUSTER_PATH}/ystar-dynamics/data/original/{FILE_NAME}" "{LOCAL_PATH}/{FILE_NAME}"
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



**Contact & Support**  
If you encounter any issues reproducing the results or have questions about the data, please open an issue or contact Verónica Bäcker-Peral (vbperal@mit.edu).