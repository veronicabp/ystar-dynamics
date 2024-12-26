from utils import *


def keep_cols(df):

    if "date_from" in df.columns:
        df["year_from"] = df["date_from"].dt.year
        df["year_registered"] = df["date_registered"].dt.year

    if "date_extended" in df.columns:
        df["year_extended"] = df["date_extended"].dt.year

    hedonics = ["bedrooms", "bathrooms", "floorarea", "livingrooms", "yearbuilt"]
    cols_to_keep = [
        "property_id",
        "date_trans",
        "L_date_trans",
        "year",
        "L_year",
        "quarter",
        "L_quarter",
        "month",
        "L_month",
        "extension",
        "extension_amount",
        "has_been_extended",
        "closed_lease",
        "duration",
        "L_duration",
        "whb_duration",
        "duration2023",
        "number_years",
        "year_from",
        "year_extended",
        "year_registered",
        "area",
        "sector",
        "postcode",
        "lpa_code",
        "years_held",
        "log_rent",
        "L_log_rent",
        "rent",
        "date_rent",
        "L_date_rent",
        "time_on_market",
        "price_change_pct",
        "time_from_listing",
        "age",
        "T",
        "T5",
        "T10",
        "k",
        "k90",
        "k700p",
        "T_at_ext",
        "Pi",
    ]

    cols_to_keep += [col for col in df if ("price" in col or "pres" in col)]
    cols_to_keep += [
        col
        for col in df
        if col.startswith("did_rsi") or col == "did_rc" or col.startswith("radius")
    ]
    cols_to_keep += (
        [col for col in hedonics]
        + [f"L_{col}" for col in hedonics]
        + [f"date_{col}" for col in hedonics]
        + [f"L_date_{col}" for col in hedonics]
    )

    cols_to_keep = list(set(cols_to_keep))

    df = df[[col for col in cols_to_keep if col in df.columns]].copy()

    # Set missing to false
    if "has_been_extended" in df.columns:
        df.loc[df.has_been_extended.isna(), "has_been_extended"] = False
        df["has_been_extended"] = df.has_been_extended.astype(int)

        df["closed_lease"] = df.closed_lease.astype(int)

    return df


def output_rent_rsi(data_folder):
    # Combine and output to stata
    rsi_resid = pd.read_pickle(
        os.path.join(data_folder, "working", "rsi_rent_resid.p")
    )[
        [
            "property_id",
            "date_rm",
            "L_date_rm",
            "year_rm",
            "L_year_rm",
            "d_log_rent_res",
            "d_rsi",
        ]
    ].rename(
        columns={"d_rsi": "d_rsi_rent_resid"}
    )

    rsi = pd.read_pickle(os.path.join(data_folder, "working", "rsi_rent.p"))[
        [
            "property_id",
            "date_rm",
            "L_date_rm",
            "year_rm",
            "L_year_rm",
            "d_log_rent",
            "d_rsi",
        ]
    ].rename(columns={"d_rsi": "d_rsi_rent"})

    rsi = rsi.merge(
        rsi_resid, on=["property_id", "date_rm", "L_date_rm", "year_rm", "L_year_rm"]
    )

    experiments = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))[
        ["property_id", "date_trans", "L_date_trans", "year", "L_year"]
    ]

    df = rsi.merge(experiments, on=["property_id"], how="inner")
    df.to_stata(os.path.join(data_folder, "clean", "rent_rsi.dta"), write_index=False)


def clean_for_dta(file_path, data_folder, restrict_cols=False):
    # Experiments
    df = pd.read_pickle(os.path.join(data_folder, file_path)).copy()

    if restrict_cols:
        df = keep_cols(df)

    csv_path = os.path.join(
        data_folder, file_path.replace(".p", ".csv").replace("working/", "clean/")
    )
    df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)

    # Convert dates to stata format
    for date in [
        "date_trans",
        "L_date_trans",
        "experiment_date",
        "date_extended",
        "date_rm",
        "L_date_rm",
        "control_date",
        "date_rent",
    ]:
        if date in df.columns:
            stata_base_date = pd.Timestamp("1960-01-01")
            df[date] = pd.to_datetime(df[date])
            df[date] = (df[date] - stata_base_date).dt.days

    stata_path = os.path.join(
        data_folder, file_path.replace(".p", ".dta").replace("working/", "clean/")
    )
    df.to_stata(stata_path, write_index=False)


def output_ystar_timeseries(data_folder):
    ystar = pd.read_pickle(os.path.join(data_folder, "clean", "ystar_estimates.p"))
    df = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))

    ystar["date"] = ystar.year + (ystar.month - 1) / 12

    ystar_pre2003, se_pre2003 = estimate_ystar(df[df.year <= 2003].copy())

    df_pre2003 = pd.DataFrame(
        {
            "date": list(range(2000, 2004)),
            "ystar": [ystar_pre2003 for _ in range(4)],
            "se": [se_pre2003 for _ in range(4)],
            "freq": "2000-2003",
        }
    )

    df_2004t2023 = (
        ystar[(ystar.year > 2003) & (ystar.year < 2024)][
            ["year", "ystar_yearly", "se_yearly"]
        ]
        .drop_duplicates()
        .rename(columns={"year": "date", "ystar_yearly": "ystar", "se_yearly": "se"})
    )
    df_2004t2023["freq"] = "annual"

    df_post2024 = (
        ystar[ystar.year >= 2024][["date", "ystar_monthly", "se_monthly"]]
        .drop_duplicates()
        .rename(columns={"ystar_monthly": "ystar", "se_monthly": "se"})
    )
    df_post2024["freq"] = "monthly"

    df = pd.concat([df_pre2003, df_2004t2023, df_post2024])
    df.dropna(subset="ystar", inplace=True)

    df["ub"] = df.ystar + 1.96 * df.se
    df["lb"] = df.ystar - 1.96 * df.se

    df.to_stata(
        os.path.join(data_folder, "clean", "ystar_estimates.dta"), write_index=False
    )

    # Monthly series
    ystar_monthly = ystar[
        ["year", "month", "ystar_monthly", "ub_monthly", "lb_monthly"]
    ].copy()

    ystar_monthly = ystar_monthly[
        (ystar_monthly.year >= 2016) & (ystar_monthly.ystar_monthly.notna())
    ].copy()

    df.to_stata(
        os.path.join(data_folder, "clean", "ystar_monthly_estimates.dta"),
        write_index=False,
    )

    # Quarterly series
    ystar_quarterly = ystar[
        [
            "year",
            "quarter",
            "ystar_quarterly",
            "ub_quarterly",
            "lb_quarterly",
            "se_quarterly",
        ]
    ].copy()

    ystar_quarterly.dropna(subset="ystar_quarterly", inplace=True)
    ystar_quarterly.to_stata(
        os.path.join(data_folder, "clean", "ystar_quarterly_estimates.dta"),
        write_index=False,
    )


def output_hedonics_variations(data_folder):
    qe = (
        pd.read_pickle(
            os.path.join(data_folder, "working", "quasi_experimental_stability.p")
        )
        .rename(columns={"ystar": "ystar_qe"})
        .drop(columns="se")
    )

    cs = (
        pd.read_pickle(
            os.path.join(data_folder, "working", "cross_sectional_stability.p")
        )
        .rename(columns={"ystar": "ystar_cs"})
        .drop(columns="se")
    )

    df = qe.merge(cs, on=["variation", "time_interaction"])
    df.to_stata(
        os.path.join(data_folder, "clean", "hedonics_variations.dta"), write_index=False
    )


def output_rent_to_price(data_folder):
    df = pd.read_pickle(os.path.join(data_folder, "working", "merged_hmlr_hedonics.p"))
    df = df[np.abs((df.date_trans - df.date_rent).dt.days) <= 365].copy()
    df["rtp"] = 100 * df["rent"] / df["price"]

    lh = pd.read_pickle(os.path.join(data_folder, "clean", "leasehold_flats.p"))
    df = df.merge(
        lh[
            [
                "property_id",
                "date_trans",
                "extension",
                "has_extension",
                "has_been_extended",
            ]
        ],
        on=["property_id", "date_trans"],
        indicator=True,
        how="left",
        suffixes=("_w_fh", ""),
    )
    df.drop(df[(df.leasehold) & (df._merge != "both")].index, inplace=True)

    df = df[
        [
            "property_id",
            "date_trans",
            "year",
            "date_rent",
            "price",
            "rent",
            "rtp",
            "extension",
            "has_extension",
            "has_been_extended",
            "leasehold",
            "freehold",
            "area",
        ]
    ].copy()

    for col in ["extension", "has_been_extended"]:
        df[col] = df[col].fillna(False)
        df[col] = df[col].astype(int)
    df["has_extension"] = df["has_extension"].fillna(0)

    df.to_stata(
        os.path.join(data_folder, "clean", "rent_to_price.dta"), write_index=False
    )


def output_residuals(data_folder):
    residuals = load_residuals(data_folder)
    residuals = residuals[["residuals", "years_held", "distance"]]
    residuals.to_stata(
        os.path.join(data_folder, "clean", "rsi_residuals.dta"), write_index=False
    )


def output_dta(data_folder):

    # output_rent_rsi(data_folder)
    output_ystar_timeseries(data_folder)
    # output_hedonics_variations(data_folder)
    # output_rent_to_price(data_folder)
    # output_residuals(data_folder)

    for data in [
        {"file": "clean/experiments.p", "restrict_cols": True},
        # {"file": "clean/experiments_flip.p", "restrict_cols": True},
        # {"file": "clean/experiments_public.p", "restrict_cols": True},
        # {"file": "clean/leasehold_flats.p", "restrict_cols": True},
        # {"file": "working/experiment_rent_panel.p", "restrict_cols": False},
        # {"file": "working/experiment_pids.p", "restrict_cols": False},
        # {"file": "working/renovations.p", "restrict_cols": False},
        # {"file": "working/for_event_study.p", "restrict_cols": False},
        # {"file": "working/global_forward.p", "restrict_cols": False},
        # {"file": "working/global_rtp.p", "restrict_cols": False},
        # {"file": "working/uk_rtp.p", "restrict_cols": False},
        # {"file": "clean/leasehold_panel.p", "restrict_cols": False},
        {"file": "clean/ystar_by_lpas_2009-2022.p", "restrict_cols": False},
    ]:
        file = data["file"]
        restrict_cols = data["restrict_cols"]

        if not os.path.exists(f"{data_folder}/{file}"):
            continue

        print(f"Saving {file}")
        clean_for_dta(file, data_folder, restrict_cols=restrict_cols)
