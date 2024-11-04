# %%
from clean.price_paid import clean_price_paid
from clean.leases import clean_leases
from clean.merge_hmlr import merge_hmlr
from clean.rsi import construct_rsi
from clean.bootstrap_rsi import bootstrap_rsi
from clean.finalize_experiments import run_create_experiments
from utils import *

if __name__ == "__main__":
    start = time.time()
    data_folder = "../data/original"

    # Clean price
    # clean_price_paid(data_folder)

    # Clean new leases
    # clean_leases(data_folder)

    # Merge new data
    # merge_hmlr(data_folder)

    # Finalize

    # Create RSI
    # construct_rsi(data_folder)
    bootstrap_rsi(data_folder)

    # Finalize experiments
    # run_create_experiments(data_folder)

    # Create timeseries

    end = time.time()
    print(f'Time elapsed: {end-start}')
# %%
