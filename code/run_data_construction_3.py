from clean.finalize_experiments import run_create_experiments
from clean.additional_datasets import make_additional_datasets
from clean.output_final_data import output_dta
from clean.rent_rsi import construct_rent_rsi
from clean.combine_ashe import combine_ashe_data
from clean.hazard_rate import calculate_hazard_rate
from clean.hedonics_variations import get_hedonics_variations
from clean.cross_sectional_estimate import *
import time
from utils import log_time_elapsed


if __name__ == "__main__":
    print("DATA CONSTRUCTION PT3:")

    start = time.time()
    data_folder = "../data/data/original"

    # Finalize experiments
    calculate_hazard_rate(data_folder)
    run_create_experiments(data_folder)

    # Additional datasets
    make_additional_datasets(data_folder)
    construct_rent_rsi(data_folder)

    combine_ashe_data(data_folder)
    expand_hilber_data(data_folder)
    get_cross_sectional_estimates(data_folder)
    get_hedonics_variations(data_folder)

    output_dta(data_folder)

    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed: {time_elapsed/60}\n\n")
    log_time_elapsed(time_elapsed, "DC3", data_folder)
# %%
