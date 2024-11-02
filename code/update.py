from clean.price_paid import *
from clean.leases import *
from clean.merge_hmlr import *
from clean.rsi import *

if __name__=='__main__':
    data_folder = '/Users/vbp/Princeton Dropbox/Veronica Backer Peral/Research/natural-rate/ystar-dynamics/data/update_21_08_24'
    original_data_folder = '/Users/vbp/Princeton Dropbox/Veronica Backer Peral/Research/natural-rate/ystar-dynamics/data/original'
    prev_data_folder = original_data_folder

    # Create new folder

    # Download new data

    # Clean price
    # update_price_paid(data_folder)

    # Clean new leases

    # start = time.time()
    # clean_leases(original_data_folder)
    # end = time.time()
    # print(f'>> Time elapsed to clean leases: {round((end-start)/60,2)} minutes.')

    # start = time.time()
    # update_leases(data_folder, prev_data_folder)
    # end = time.time()
    # print(f'>> Time elapsed to update leases: {round((end-start)/60,2)} minutes.')

    # Merge new data
    # start = time.time()
    # merge_hmlr(original_data_folder)
    # end = time.time()
    # print(f'>> Time elapsed to merge data: {round((end-start)/60,2)} minutes.')

    # start = time.time()
    # update_hmlr_merge(data_folder, prev_data_folder, original_data_folder)
    # end = time.time()
    # print(f'>> Time elapsed to merge updated data: {round((end-start)/60,2)} minutes.')

    # Create RSI
    start = time.time()
    construct_rsi(original_data_folder)
    end = time.time()
    print(f'>> Time elapsed to create RSI: {round((end-start)/60,2)} minutes.')

    # start = time.time()
    # update_rsi(data_folder, prev_data_folder, original_data_folder)
    # end = time.time()
    # print(f'>> Time elapsed to update RSI: {round((end-start)/60,2)} minutes.')

    # Finalize experiments

    # Create timeseries