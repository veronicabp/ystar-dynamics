# %%
print("In program.")

from clean.price_paid import clean_price_paid
from clean.leases import clean_leases
from clean.merge_hmlr import merge_hmlr
# from clean.rsi_dask import construct_rsi
from clean.rsi import construct_rsi, get_residuals
from clean.bootstrap import bootstrap_rsi
from clean.finalize_experiments import run_create_experiments
from clean.hedonics_variations import get_rsi_hedonic_variations
from analysis.compile_bootstrap import compile_bootstrap
from utils import *

if __name__ == "__main__":
    start = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data_folder = "../data/original"

    if rank==0:
        print('Starting MAIN.')

    # Clean price
    # clean_price_paid(data_folder)

    # Clean new leases
    # clean_leases(data_folder)

    # Merge new data
    # merge_hmlr(data_folder)

    # Finalize

    # Create RSI
    get_residuals(data_folder)
    construct_rsi(data_folder)
    # get_rsi_hedonic_variations(data_folder)

    # Finalize experiments
    # run_create_experiments(data_folder)

    # Create timeseries
    # compile_bootstrap(data_folder)

    end = time.time()
    print(f"Time elapsed: {end-start}")
# %%
