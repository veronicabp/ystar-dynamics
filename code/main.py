# %%
from clean.price_paid import clean_price_paid
from clean.leases import clean_leases
from clean.merge_hmlr import merge_hmlr

if __name__ == "__main__":
    data_folder = "/Users/vbp/Princeton Dropbox/Veronica Backer Peral/Research/natural-rate/ystar-dynamics/data/original"

    # Clean price
    clean_price_paid(data_folder)

    # Clean new leases
    # clean_leases(data_folder)

    # Merge new data
    merge_hmlr(data_folder)

    # Finalize

    # Create RSI

    # Finalize experiments

    # Create timeseries

# %%
