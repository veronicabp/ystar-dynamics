from analysis.section4 import *
from analysis.compile_bootstrap import compile_bootstrap
from analysis.appendix import *
import time
from utils import log_time_elapsed

if __name__ == "__main__":
    start = time.time()
    print("Analysis:")
    data_folder = "../data/data/original"
    create_yield_curve_figures(data_folder, figures_folder)
    create_differencing_out_figure(figures_folder)
    residual_plots(data_folder, figures_folder)
    lpa_map(data_folder, figures_folder)
    construct_alpha_table(data_folder, tables_folder)
    compile_bootstrap(data_folder)

    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed: {(time_elapsed)/60}\n\n")
    log_time_elapsed(time_elapsed, "A", data_folder)
# %%
