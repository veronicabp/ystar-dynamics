# %%
from utils import *
from clean.rsi import *


def run_rsi_for_bootstrap(args):
    i, row, controls_sub, start_date, end_date = args
    # print(f"\n\n{row.property_id} {row.date_trans}: num controls = {len(controls_sub)}")

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

    params, constant, summary = rsi(controls_sub, dummy_vars=dummy_vars)

    d_rsi = (
        params[dates.index(row["date"])] - params[dates.index(row["L_date"])] + constant
    )

    return [row.property_id, row.date_trans, d_rsi]


if __name__ == "__main__":
    start_year = 1995
    start_quarter = 1
    end_year = 2024
    end_quarter = 1

    start_date = start_year * 4 + start_quarter
    end_date = end_year * 4 + end_quarter

    data_folder = "../data/original"

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

    # Add weights for Case-Shiller RSI
    residuals = pd.read_pickle(os.path.join(data_folder, "working", "residuals.p"))
    controls = add_weights(controls, residuals)
    controls = controls.merge(
        residuals,
        left_on=["experiment_pid", "experiment_date", "property_id", "date_trans"],
        right_on=[
            "pid_treated",
            "date_trans_treated",
            "pid_control",
            "date_trans_control",
        ],
        how="inner",
    )
    controls["weight"] = 1 / (
        controls["b_cons"]
        + controls["b_years_held"] * controls["years_held"]
        + controls["b_distance"] * controls["distance"]
    )
    controls = controls[
        ["experiment_pid", "experiment_date", "property_id", "date_trans", "weight"]
    ]

    controls = controls.merge(df, on=["property_id", "date_trans"], how="inner")

    extensions["date"] = extensions["year"] * 4 + extensions["quarter"]
    extensions["L_date"] = extensions["L_year"] * 4 + extensions["L_quarter"]
    extensions.reset_index(inplace=True, drop=True)

    b = 1
    boot_sample = resample(controls, random_state=b)
    groups = {
        key: group
        for key, group in boot_sample.groupby(["experiment_pid", "experiment_date"])
    }

    # Get args
    args = []
    for i, row in extensions.iterrows():
        if (row.property_id, row.date_trans) in groups:
            arg = (
                i,
                row,
                groups[(row.property_id, row.date_trans)],
                start_date,
                end_date,
            )
            args.append(arg)

    # print("Running in serial:")
    # results = []
    # for arg in tqdm(args):
    #     results.append(run_rsi_for_bootstrap(arg))

    args = args[:5000]

    print("Running in parallel:")
    # results = pqdm(args, run_rsi_for_bootstrap, n_jobs=4)
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_rsi_for_bootstrap, args))

    result_df = pd.DataFrame(
        results, columns=["property_id", "date_trans", f"d_rsi_boot{b}"]
    )

    result_df.to_pickle(os.path.join(data_folder, "working", "boostrap", f"boot{b}.p"))
