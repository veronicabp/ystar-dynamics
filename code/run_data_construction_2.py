from utils import *

if __name__ == "__main__":
    print("DATA CONSTRUCTION PT 2:")

    start = time.time()

    # Create RSI
    get_residuals(data_folder)
    construct_rsi(data_folder)
    get_rsi_hedonic_variations(data_folder)
    bootstrap_rsi(data_folder)
    construct_restrictive_controls(data_folder)

    end = time.time()
    print(f"Time elapsed: {(end-start)/60}\n\n")
# %%
