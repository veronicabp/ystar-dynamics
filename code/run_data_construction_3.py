from utils import *

if __name__ == "__main__":
    print("DATA CONSTRUCTION PT3:")

    start = time.time()

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
    print(f"Time elapsed: {(end-start)/60}\n\n")
# %%
