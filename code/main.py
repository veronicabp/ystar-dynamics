# %%
from clean.clean_interest_rates import get_boe_interest_rates
from clean.price_paid import clean_price_paid
from clean.leases import clean_leases
from clean.merge_hmlr import merge_hmlr, convert_hedonics_data
from clean.rsi import *
from clean.restrictive_controls import construct_restrictive_controls
from clean.bootstrap_rsi import bootstrap_rsi
from clean.finalize_experiments import run_create_experiments
from clean.additional_datasets import make_additional_datasets
from clean.output_final_data import output_dta
from clean.rent_rsi import construct_rent_rsi
from clean.combine_ashe import combine_ashe_data
from clean.hedonics_variations import *
from clean.hazard_rate import calculate_hazard_rate
from clean.cross_sectional_estimate import *
from analysis.section4 import *
from analysis.compile_bootstrap import compile_bootstrap
from analysis.appendix import *

from utils import *

if __name__ == "__main__":
    print("MAIN:")

    start = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data_folder = "../data/original"
    output_folder = (
        "/Users/vbp/Princeton Dropbox/Veronica Backer Peral/Apps/Overleaf/UK Duration"
    )
    figures_folder = os.path.join(output_folder, "Figures")
    tables_folder = os.path.join(output_folder, "Tables")

    ################# Create dataset #################
    # get_boe_interest_rates(data_folder)

    # # Clean price
    # clean_price_paid(data_folder)

    # # Clean new leases
    # clean_leases(data_folder)

    # # Merge new data
    # convert_hedonics_data(data_folder)
    # merge_hmlr(data_folder)

    # # Create RSI
    # construct_rsi_no_parallel(data_folder)
    # get_residuals(data_folder)
    # construct_rsi(data_folder)
    # get_rsi_hedonic_variations(data_folder)
    # bootstrap_rsi(data_folder)

    # construct_restrictive_controls(data_folder)

    # # Finalize experiments
    # calculate_hazard_rate(data_folder)
    # run_create_experiments(data_folder)

    # # Additional datasets
    # make_additional_datasets(data_folder)
    # construct_rent_rsi(data_folder)

    # combine_ashe_data(data_folder)
    # expand_hilber_data(data_folder)
    # get_cross_sectional_estimates(data_folder)
    # get_hedonics_variations(data_folder)

    # output_dta(data_folder)

    ################# Analysis #################
    # print(">> Analysis:")
    # create_yield_curve_figures(data_folder, figures_folder)
    # create_differencing_out_figure(figures_folder)
    # residual_plots(data_folder, figures_folder)
    # lpa_map(data_folder, figures_folder)
    # construct_alpha_table(data_folder, tables_folder)
    compile_bootstrap(data_folder)

    # end = time.time()
    # print(f"Time elapsed: {(end-start)/60}")
# %%
