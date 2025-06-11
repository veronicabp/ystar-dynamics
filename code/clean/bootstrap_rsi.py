# %%

from utils import *
from clean.rsi import *
from clean.finalize_experiments import *


def process_row(args):
    b, i, row, controls_sub, start_date, end_date = args
    if len(controls_sub) <= 2:
        return [row.property_id, row.date_trans, np.nan]

    # Check that we still have sufficient observations to produce an RSI
    uf = get_union(controls_sub)
    connected_dates = uf.build_groups()

    if not uf.are_connected(row.date, row.L_date):
        return [row.property_id, row.date_trans, np.nan]

    controls_sub = get_dummies(controls_sub, start_date=start_date, end_date=end_date)
    dates = list(uf.get_group(row.date))
    dummy_vars = [f"d_{int(date)}" for i, date in enumerate(dates) if i != 0]

    params, constant, summary = rsi(
        controls_sub, dummy_vars=dummy_vars, weight_col="weight"
    )
    # print(f"{b}:\n\n{summary}")

    d_rsi = (
        params[dates.index(row["date"])] - params[dates.index(row["L_date"])] + constant
    )

    return [row.property_id, row.date_trans, d_rsi]


def run_single_boot_iteration(
    b, boot_sample, extensions, data_folder, start_date, end_date
):

    groups = {
        key: group
        for key, group in boot_sample.groupby(["experiment_pid", "experiment_date"])
    }

    # Get args
    args = []
    for i, row in extensions.iterrows():
        if (row.property_id, row.date_trans) in groups:
            arg = (
                b,
                i,
                row,
                groups[(row.property_id, row.date_trans)],
                start_date,
                end_date,
            )
            args.append(arg)

    pool = Pool(16)
    results = pool.map(process_row, args)
    result_df = pd.DataFrame(
        results, columns=["property_id", "date_trans", f"d_rsi_boot{b}"]
    )

    result_df.to_pickle(
        os.path.join(data_folder, "working", output_folder, f"boot{b}.p")
    )


def bootstrap_rsi(
    data_folder,
    bootstrap_iter=100,
    start_year=1995,
    start_quarter=1,
    end_year=2024,
    end_quarter=1,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Number of times each process (node) runs the function
    num_runs_per_process = bootstrap_iter // size

    start_date = start_year * 4 + start_quarter
    end_date = end_year * 4 + end_quarter

    df = load_data(data_folder, restrict_cols=False)[
        [
            "property_id",
            "date_trans",
            "L_date_trans",
            "date",
            "L_date",
            "d_log_price",
        ]
    ].copy()

    extensions = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))
    controls = pd.read_pickle(os.path.join(data_folder, "working", "control_pids.p"))
    controls.rename(columns={"date": "date_trans"}, inplace=True)

    controls = controls[
        ["experiment_pid", "experiment_date", "property_id", "date_trans"]
    ]
    controls = controls.merge(df, on=["property_id", "date_trans"], how="inner")

    extensions["date"] = extensions["year"] * 4 + extensions["quarter"]
    extensions["L_date"] = extensions["L_year"] * 4 + extensions["L_quarter"]
    extensions.reset_index(inplace=True, drop=True)

    # Calculate the global iteration number for unique filenames
    for i in range(num_runs_per_process):
        b = rank * num_runs_per_process + i

        # Generate random weights from Dirichlet distribution for Bayesian bootstrap
        np.random.seed(b)
        w = np.random.dirichlet(np.ones(len(controls)), 1)[0]
        controls["weight"] = w

        if os.path.exists(
            os.path.join(data_folder, "working", "bayes_bootstrap", f"boot{b}.p")
        ):
            continue

        print(
            f"\n\nRunning iteration {b} with total weight {w.sum()}...\n"
            + "-" * 20
            + "\n"
        )
        run_single_boot_iteration(
            b, controls, extensions, data_folder, start_date, end_date
        )
        print(f"Finished iteration {b}.")
