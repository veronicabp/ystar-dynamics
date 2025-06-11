from clean.restrictive_controls import construct_restrictive_controls
from clean.bootstrap_rsi import bootstrap_rsi
from clean.hedonics_variations import get_rsi_hedonic_variations
from clean.rsi import *
import time
from utils import log_time_elapsed

if __name__ == "__main__":
    print("DATA CONSTRUCTION PT 2:")

    start = time.time()
    data_folder = "../data/data/original"

    # Create RSI
    get_residuals(data_folder)
    construct_rsi(data_folder)
    get_rsi_hedonic_variations(data_folder)
    bootstrap_rsi(data_folder)
    construct_restrictive_controls(data_folder)

    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed: {time_elapsed/60}\n\n")
    log_time_elapsed(time_elapsed, "DC2", data_folder)
# %%
