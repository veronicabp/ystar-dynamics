# %%
from clean.clean_interest_rates import get_boe_interest_rates
from clean.price_paid import clean_price_paid
from clean.leases import clean_leases
from clean.merge_hmlr import merge_hmlr, convert_hedonics_data
from clean.rsi import *
import time
from utils import log_time_elapsed

# %%
if __name__ == "__main__":

    print("DATA CONSTRUCTION PT 1:")

    start = time.time()
    data_folder = "../data/data/original"

    get_boe_interest_rates(data_folder)

    # Clean price
    clean_price_paid(data_folder)

    # Clean new leases
    clean_leases(data_folder)

    # Merge new data
    convert_hedonics_data(data_folder)
    merge_hmlr(data_folder)
    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed: {time_elapsed/60}\n\n")
    log_time_elapsed(time_elapsed, "DC1", data_folder)
# # %%
