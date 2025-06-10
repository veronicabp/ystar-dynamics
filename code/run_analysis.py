from utils import *

if __name__ == "__main__":
    start = time.time()
    print("Analysis:")
    create_yield_curve_figures(data_folder, figures_folder)
    create_differencing_out_figure(figures_folder)
    residual_plots(data_folder, figures_folder)
    lpa_map(data_folder, figures_folder)
    construct_alpha_table(data_folder, tables_folder)
    compile_bootstrap(data_folder)

    end = time.time()
    print(f"Time elapsed: {(end-start)/60}\n\n")
# %%
