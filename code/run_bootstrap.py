from clean.bootstrap_rsi import bootstrap_rsi
from utils import *

if __name__ == "__main__":

    print('Running')

    start = time.time()
    bootstrap_rsi(data_folder="../data/original", bootstrap_iter=500)
    end = time.time()
    print(f'Time elapsed: {end-start}')