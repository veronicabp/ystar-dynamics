# %%
from clean.price_paid import clean_price_paid
from clean.leases import clean_leases
from clean.merge_hmlr import merge_hmlr, convert_hedonics_data
from clean.rsi import *
from clean.restrictive_controls import construct_restrictive_controls
from clean.bootstrap_rsi import bootstrap_rsi
from clean.finalize_experiments import run_create_experiments
from clean.additional_datasets import make_additional_datasets
from clean.output_final_data import output_dta
from analysis.compile_bootstrap import compile_bootstrap
from utils import *

if __name__ == "__main__":
    print("MAIN:")

    start = time.time()
    data_folder = "../data/original"

    # Clean price
    # clean_price_paid(data_folder)

    # Clean new leases
    # clean_leases(data_folder)

    # Merge new data
    # convert_hedonics_data(data_folder)
    merge_hmlr(data_folder)

    # Create RSI
    # construct_rsi_no_parallel(data_folder)
    # get_residuals(data_folder)
    # construct_rsi(data_folder)
    # get_rsi_hedonic_variations(data_folder)
    # bootstrap_rsi(data_folder)

    # construct_restrictive_controls(data_folder)

    # Finalize experiments
    # run_create_experiments(data_folder)

    # Additional datasets
    # make_additional_datasets(data_folder)
    # construct_rent_rsi(data_folder)
    # construct_hazard_rate(data_folder)
    # get_cross_sectional_estimates(data_folder)
    # compile_bootstrap(data_folder)

    # output_dta(data_folder)

    end = time.time()
    print(f"Time elapsed: {end-start}")
# %%
