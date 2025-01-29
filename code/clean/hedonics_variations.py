from utils import *
from clean.rsi import *


def get_rsi_hedonic_variations(
    data_folder,
    start_year=1995,
    start_quarter=1,
    end_year=2024,
    end_quarter=1,
    n_jobs=16,
):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_date = start_year * 4 + start_quarter
    end_date = end_year * 4 + end_quarter

    df = load_data(data_folder)
    hedonics = pd.read_pickle(
        os.path.join(data_folder, "working", "merged_hmlr_hedonics.p")
    )

    hedonics_cols = [
        col
        for col in hedonics.columns
        if col.startswith("pres") or col.startswith("tpres")
    ]
    hedonics.drop(
        columns=[
            col
            for col in hedonics.columns
            if col not in hedonics_cols + ["property_id", "date_trans"]
        ],
        inplace=True,
    )
    hedonics = hedonics.merge(
        df[["property_id"]].drop_duplicates(),
        on=["property_id"],
        how="inner",
    )

    for col in hedonics_cols:

        output_file = os.path.join(
            data_folder, "working", "hedonic_variations", f"rsi_{col}.p"
        )

        # If we've already run this one, continue
        if os.path.exists(output_file):
            continue

        print(f"\n\n{col}:\n" + "=" * 20)

        this_hedonic = (
            hedonics[["property_id", "date_trans", col]].dropna(subset=[col]).copy()
        )

        # Merge on first date
        df_hedonic = df.merge(
            this_hedonic, on=["property_id", "date_trans"], how="inner"
        )

        # Merge on second date
        df_hedonic = df_hedonic.merge(
            this_hedonic.rename(
                columns={"date_trans": "L_date_trans", col: f"L_{col}"}
            ),
            on=["property_id", "L_date_trans"],
            how="inner",
        )

        df_hedonic[f"d_pres"] = df_hedonic[col] - df_hedonic[f"L_{col}"]

        # Get RSI for this hedonic
        local_extensions, local_controls = get_local_extensions_controls(
            df_hedonic, rank, size
        )
        print(
            f"[{rank}/{size}]:\n\nNum Ext: {len(local_extensions)}\nNum Ctrl: {len(local_controls)}\n Local DF areas: {sorted(local_extensions.area.unique())}\n"
        )

        rsi = get_rsi(
            local_extensions,
            local_controls,
            start_date=start_date,
            end_date=end_date,
            case_shiller=False,
            price_var="d_pres",
        )
        rsi_gather = comm.gather(rsi, root=0)

        if rank == 0:
            combined_rsi = pd.concat(rsi_gather)
            combined_rsi.to_pickle(output_file)


def estimate_cs_stability(data_folder):
    hedonics = pd.read_pickle(
        os.path.join(data_folder, "working", "merged_hmlr_hedonics.p")
    )

    # Drop freeholds that switch between leasehold and freehold, because these may be transactions of the underlying freehold
    hedonics["freehold"] = hedonics["freehold"].astype(int)
    hedonics["tot_freehold"] = hedonics.groupby("property_id")["freehold"].transform(
        "sum"
    )
    hedonics["num_obs"] = hedonics.groupby("property_id")["freehold"].transform("count")

    hedonics.drop(
        hedonics[
            (hedonics.freehold == 1) & (hedonics.tot_freehold != hedonics.num_obs)
        ].index,
        inplace=True,
    )

    hedonics.drop(hedonics[hedonics.year < 2000].index, inplace=True)

    # We need at least one freehold and one non freehold by quarter-postcode group
    group_vars = ["outcode", "year", "quarter"]
    hedonics["pct_freehold"] = hedonics.groupby(group_vars)["freehold"].transform(
        "mean"
    )
    hedonics.drop(
        hedonics[(hedonics.pct_freehold == 0) | (hedonics.pct_freehold == 1)].index,
        inplace=True,
    )

    hedonics_cols = ["log_price"] + [
        col
        for col in hedonics.columns
        if col.startswith("pres") or col.startswith("tpres")
    ]

    leaseholds = hedonics.drop(hedonics[hedonics.freehold == 1].index)
    freeholds = hedonics.drop(hedonics[hedonics.freehold == 0].index)

    fh_means = (
        freeholds.groupby(group_vars)[
            hedonics_cols
        ]  # group by your group_vars on the hedonic cols
        .mean()
        .reset_index()  # so you can merge
    )

    leaseholds = pd.merge(leaseholds, fh_means, on=group_vars, suffixes=("", "_fh"))

    new_columns = {
        f"{col}_discount": leaseholds[col] - leaseholds[f"{col}_fh"]
        for col in hedonics_cols
    }

    # Create all new columns at once
    leaseholds = leaseholds.assign(**new_columns)

    leaseholds = leaseholds[
        ["property_id", "log_price", "duration"]
        + group_vars
        + [c for c in leaseholds.columns if c.endswith("_discount")]
    ]

    leaseholds["T"] = leaseholds["duration"]
    leaseholds["k"] = 1000

    def fh_discount_function(ystar, T, k):
        exponent = (ystar / 100) * T
        return np.log(np.clip(1 - np.exp(-exponent), 1e-15, None))

    ystars = []
    ses = []
    variations = []
    time_interaction = []

    ystar, se = estimate_ystar(
        leaseholds.copy(),
        lhs_var=f"log_price_discount",
        model_function=fh_discount_function,
        get_se="boot",
    )
    ystars.append(ystar)
    ses.append(se)
    variations.append("None")
    time_interaction.append(False)

    for col in tqdm(hedonics_cols):
        ystar, se = estimate_ystar(
            leaseholds.copy(),
            lhs_var=f"{col}_discount",
            model_function=fh_discount_function,
            get_se="boot",
        )

        ystars.append(ystar)
        ses.append(se)
        variations.append(
            col.replace("_discount", "").replace("tpres_", "").replace("pres_", "")
        )
        time_interaction.append(col.startswith("tpres"))

    results = pd.DataFrame(
        {
            "ystar": ystars,
            "se": ses,
            "variation": variations,
            "time_interaction": time_interaction,
        }
    )

    results.to_pickle(
        os.path.join(data_folder, "working", "cross_sectional_stability.p")
    )


def estimate_qe_stability(data_folder):

    df = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))[
        ["property_id", "date_trans", "T", "k", "did_rsi"]
    ]

    hedonics_folder = os.path.join(data_folder, "working", "hedonics_variations")

    ystars = []
    ses = []
    variations = []
    time_interaction = []

    # Baseline
    ystar, se = estimate_ystar(
        df,
        get_se="boot",
    )

    ystars.append(ystar)
    ses.append(se)
    variations.append("None")
    time_interaction.append(False)

    for rsi_file in tqdm(sorted(os.listdir(hedonics_folder))):

        if not rsi_file.endswith(".p"):
            continue

        rsi = pd.read_pickle(os.path.join(hedonics_folder, rsi_file))
        rsi["did"] = rsi["d_pres"] - rsi["d_rsi"]

        df_ = df.merge(rsi, on=["property_id", "date_trans"])

        ystar, se = estimate_ystar(
            df_,
            lhs_var=f"did",
            get_se="boot",
        )

        ystars.append(ystar)
        ses.append(se)
        variations.append(
            rsi_file.replace(".p", "")
            .replace("rsi_", "")
            .replace("tpres_", "")
            .replace("pres_", "")
        )
        time_interaction.append(rsi_file.startswith("rsi_tpres"))

    results = pd.DataFrame(
        {
            "ystar": ystars,
            "se": ses,
            "variation": variations,
            "time_interaction": time_interaction,
        }
    )

    results.to_pickle(
        os.path.join(data_folder, "working", "quasi_experimental_stability.p")
    )


def get_hedonics_variations(data_folder):
    print("Quasi Experimental Stability:")
    estimate_qe_stability(data_folder)

    print("Cross Sectional Stability:")
    estimate_cs_stability(data_folder)
