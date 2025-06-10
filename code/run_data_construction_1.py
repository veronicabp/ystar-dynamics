from utils import *

if __name__ == "__main__":
    print("DATA CONSTRUCTION PT 1:")

    start = time.time()

    get_boe_interest_rates(data_folder)

    # Clean price
    clean_price_paid(data_folder)

    # Clean new leases
    clean_leases(data_folder)

    # Merge new data
    convert_hedonics_data(data_folder)
    merge_hmlr(data_folder)
    end = time.time()
    print(f"Time elapsed: {(end-start)/60}\n\n")
# %%
