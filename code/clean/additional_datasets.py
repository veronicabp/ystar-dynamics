from utils import *


def build_rent_panel(data_folder):
    leasehold_flats = pd.read_pickle(
        os.path.join(data_folder, "clean", "leasehold_flats.p")
    )

    # Rightmove
    rightmove_rents_flats = pd.read_pickle(
        os.path.join(data_folder, "working", "rightmove_rents_flats.p")
    )
    rightmove_merge_keys = pd.read_pickle(
        os.path.join(data_folder, "working", "rightmove_merge_keys.p")
    )

    rightmove = rightmove_rents_flats.merge(
        rightmove_merge_keys[["property_id", "property_id_rm"]], on="property_id_rm"
    )

    rightmove.rename(columns={"listingprice": "rent"}, inplace=True)
    rightmove["log_rent"] = np.log(rightmove["rent"])
    rightmove.loc[rightmove.log_rent == -np.inf, "log_rent"] = np.nan

    # Collapse by property_id and date_rm
    rightmove = (
        rightmove.groupby(["property_id", "date_rm"])
        .agg({"log_rent": "mean"})
        .reset_index()
    )

    # Zoopla
    zoopla_rents_flats = pd.read_pickle(
        os.path.join(data_folder, "working", "zoopla_rents_flats.p")
    )
    zoopla_merge_keys = pd.read_pickle(
        os.path.join(data_folder, "working", "zoopla_merge_keys.p")
    )

    zoopla = zoopla_rents_flats.merge(
        zoopla_merge_keys[["property_id", "property_id_zoop"]], on="property_id_zoop"
    )

    zoopla.rename(columns={"price_zoopla": "rent"}, inplace=True)
    zoopla = zoopla[zoopla["date_zoopla"].notna()]

    zoopla["rent"] *= 52  # Annualize rent
    zoopla["log_rent"] = np.log(zoopla["rent"])
    zoopla.loc[zoopla.log_rent == -np.inf, "log_rent"] = np.nan

    # Collapse by property_id and date_zoop
    zoopla = (
        zoopla.groupby(["property_id", "date_zoopla"])
        .agg({"log_rent": "mean"})
        .reset_index()
    )
    zoopla.rename(columns={"date_zoopla": "date_rm"}, inplace=True)

    # Combine Zoopla and Rightmove
    combined_rent = pd.concat([zoopla, rightmove], ignore_index=True)
    combined_rent = combined_rent[combined_rent["log_rent"].notna()]

    # Collapse again if duplicates exist
    combined_rent = (
        combined_rent.groupby(["property_id", "date_rm"])
        .agg({"log_rent": "mean"})
        .reset_index()
    )

    # Sort and generate year_rm
    combined_rent = combined_rent.sort_values(["property_id", "date_rm"])
    combined_rent["year_rm"] = combined_rent["date_rm"].dt.year

    # Filter for properties with more than one listing
    combined_rent["count"] = combined_rent.groupby("property_id")[
        "property_id"
    ].transform("size")
    combined_rent = combined_rent[combined_rent["count"] > 1].drop(columns="count")

    # Generate lagged values and differences
    combined_rent["L_date_rm"] = combined_rent.groupby("property_id")["date_rm"].shift(
        1
    )
    combined_rent["L_year_rm"] = combined_rent.groupby("property_id")["year_rm"].shift(
        1
    )
    combined_rent["L_log_rent"] = combined_rent.groupby("property_id")[
        "log_rent"
    ].shift(1)

    combined_rent["d_log_rent"] = (
        combined_rent["log_rent"] - combined_rent["L_log_rent"]
    )
    combined_rent["d_log_rent_ann"] = combined_rent["d_log_rent"] / (
        (combined_rent["date_rm"] - combined_rent["L_date_rm"]).dt.days / 365
    )

    # Drop pairs within 6 months
    combined_rent = combined_rent[
        ((combined_rent["date_rm"] - combined_rent["L_date_rm"]).dt.days >= 180)
        | (combined_rent.L_date_rm.isna())
    ]

    # Merge transaction data
    combined_rent = combined_rent.merge(
        leasehold_flats[["property_id", "date_trans", "L_date_trans"]],
        on="property_id",
        how="left",
    )

    # Generate has_trans indicator
    combined_rent["has_trans"] = (
        (combined_rent["L_date_rm"] <= combined_rent["date_trans"])
        & (combined_rent["date_rm"] >= combined_rent["date_trans"])
    ).astype(int)

    # Remove duplicate property_id, date_rm entries
    combined_rent = combined_rent.drop_duplicates(subset=["property_id", "date_rm"])

    # Regression with fixed effects and residuals (using statsmodels for OLS)
    combined_rent["interaction_term"] = (
        combined_rent["year_rm"].astype(str)
        + "_"
        + combined_rent["L_year_rm"].astype(str)
    )
    combined_rent.loc[combined_rent.L_year_rm.isna(), "interaction_term"] = np.nan
    combined_rent = residualize(
        combined_rent.copy(),
        "d_log_rent",
        [],
        ["interaction_term", "has_trans"],
        "d_log_rent_res",
    )

    # Save results
    combined_rent.to_pickle(os.path.join(data_folder, "working", "log_rent_panel.p"))

    # Save data for RSI
    df = leasehold_flats.drop(
        leasehold_flats[
            (leasehold_flats.has_extension) & (~leasehold_flats.extension)
        ].index
    )
    df = df[
        [
            "property_id",
            "duration2023",
            "latitude",
            "longitude",
            "area",
            "extension",
            "date_extended",
        ]
    ]

    df.drop_duplicates(subset="property_id", inplace=True)
    df = df.merge(combined_rent, on="property_id", how="inner")
    df.to_pickle(os.path.join(data_folder, "working", "for_rent_rsi.p"))


def get_experiment_ids(data_folder):
    # Load experiments dataset
    experiments = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))

    experiments["experiment_pid"] = experiments["property_id"]
    experiments["experiment_date"] = experiments["date_trans"]
    experiments["date"] = experiments["date_trans"]

    experiments = experiments[
        ["experiment_pid", "experiment_date", "property_id", "date"]
    ]
    experiments["type"] = "extension"
    experiments.to_pickle(os.path.join(data_folder, "working", "extension_pids.p"))

    controls = pd.read_pickle(os.path.join(data_folder, "working", "residuals.p"))
    controls["experiment_pid"] = controls["pid_treated"]
    controls["experiment_date"] = controls["date_trans_treated"]
    controls["property_id"] = controls["pid_control"]
    controls["date"] = controls["date_trans_control"]

    controls = controls[["experiment_pid", "experiment_date", "property_id", "date"]]
    controls["type"] = "control"
    controls.to_pickle(os.path.join(data_folder, "working", "control_pids.p"))

    # Append extension and control datasets
    experiment_pids = pd.concat([experiments, controls])
    experiment_pids.sort_values(
        by=["experiment_pid", "experiment_date", "type"], inplace=True
    )
    experiment_pids.to_pickle(os.path.join(data_folder, "working", "experiment_pids.p"))


def get_experiment_rent_panel(data_folder):
    extension_pids = pd.read_pickle(
        os.path.join(data_folder, "working", "extension_pids.p")
    )
    log_rent_panel = pd.read_pickle(
        os.path.join(data_folder, "working", "log_rent_panel.p")
    )

    merged_df = extension_pids.merge(log_rent_panel, on="property_id", how="inner")
    merged_df = merged_df[["experiment_pid", "experiment_date"]].drop_duplicates()

    # Load data for rent repeat sales index
    experiment_pids = pd.read_pickle(
        os.path.join(data_folder, "working", "experiment_pids.p")
    )

    # Merge with matched_pids on 'experiment_pid' and 'experiment_date' with matched rows only
    experiment_matched_df = experiment_pids.merge(
        merged_df, on=["experiment_pid", "experiment_date"], how="inner"
    )

    # Keep one instance of each property per experiment
    experiment_matched_df["experiment"] = experiment_matched_df.groupby(
        ["experiment_pid", "experiment_date"]
    ).ngroup()
    experiment_matched_df = experiment_matched_df.drop_duplicates(
        subset=["experiment", "property_id", "type"], keep="first"
    )

    # Merge with rent data
    experiment_rent_panel = experiment_matched_df.merge(
        log_rent_panel, on="property_id", how="inner"
    )

    # Merge in experiment data
    experiments = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))
    experiment_rent_panel = experiment_rent_panel.rename(
        columns={
            "property_id": "pid",
            "date_trans": "control_date",
            "experiment_pid": "property_id",
            "experiment_date": "date_trans",
        }
    )

    # Final merge with experiment data
    final_panel = experiment_rent_panel.merge(
        experiments[
            [
                "property_id",
                "date_trans",
                "year",
                "L_year",
                "L_date_trans",
                "date_extended",
            ]
        ],
        on=["property_id", "date_trans"],
        how="inner",
    )

    # Save the final dataset
    final_panel.to_pickle(
        os.path.join(data_folder, "working", "experiment_rent_panel.p")
    )


def make_timeseries(data_folder, year0=2003, year1=2023):
    df = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))

    # Initialize columns for yearly, quarterly, and monthly values
    data_yearly = []
    data_quarterly = []
    data_monthly = []

    # Loop over each year and quarter
    for year in range(year0, year1 + 1):
        print(year)
        print("-" * 10)
        estimate, se = estimate_ystar(df[df.year == year].copy())
        data_yearly.append([year, estimate, se])

        # Loop over each quarter
        for quarter in range(1, 5):
            print(f">{quarter}")
            estimate, se = estimate_ystar(
                df[(df.year == year) & (df.quarter == quarter)].copy()
            )
            if estimate > 10:
                estimate, se = None, None
            data_quarterly.append([year, quarter, estimate, se])

            # Loop over each month
            if estimate:
                for m in range(1, 4):
                    month = (quarter - 1) * 3 + m
                    print(f">>{month}")
                    estimate, se = estimate_ystar(
                        df[(df.year == year) & (df.month == month)].copy()
                    )
                    if estimate > 10:
                        estimate, se = None, None
                    data_monthly.append([year, quarter, month, estimate, se])

    df_yearly = pd.DataFrame(data_yearly, columns=["year", "ystar_yearly", "se_yearly"])
    df_quarterly = pd.DataFrame(
        data_quarterly, columns=["year", "quarter", "ystar_quarterly", "se_quarterly"]
    )
    df_monthly = pd.DataFrame(
        data_monthly,
        columns=["year", "quarter", "month", "ystar_monthly", "se_monthly"],
    )

    df = df_yearly.merge(df_quarterly, on="year", how="outer")
    df = df.merge(df_monthly, on=["year", "quarter"], how="outer")

    for freq in ["yearly", "quarterly", "monthly"]:
        df[f"ub_{freq}"] = df[f"ystar_{freq}"] + 1.96 * df[f"se_{freq}"]
        df[f"lb_{freq}"] = df[f"ystar_{freq}"] - 1.96 * df[f"se_{freq}"]

    df.to_pickle(os.path.join(data_folder, "clean", "ystar_estimates.p"))


def build_event_study(data_folder):
    rsi_full = pd.read_pickle(os.path.join(data_folder, "working", "rsi_full.p"))
    # rsi_full["date_trans_ext"] = pd.to_datetime(rsi_full["date_trans"], format="%m/%d/%Y")
    # rsi_full.drop(columns=["date_trans"], inplace=True)
    # rsi_full.drop_duplicates(subset=["property_id", "date"], inplace=True)
    # rsi_full_nd = f"{working}/rsi_full_nd.csv"
    # rsi_full.to_csv(rsi_full_nd, index=False)

    # # Load and clean experiments data
    # experiments = pd.read_stata(f"{clean}/experiments.dta")
    # experiments = experiments[experiments["did_rsi"].notna()]
    # experiments = experiments[["property_id", "year", "whb_duration"]]
    # experiments.rename(columns={"year": "experiment_year", "whb_duration": "experiment_duration"}, inplace=True)
    # experiments.drop_duplicates(subset=["property_id"], inplace=True)
    # experiments.to_stata(f"{working}/pids.dta", write_index=False)

    # # Merge leasehold flats with experiments
    # leasehold_flats = pd.read_stata(f"{clean}/leasehold_flats.dta")
    # merged = leasehold_flats.merge(experiments, on="property_id", how="inner")
    # merged = merged[~((merged["year"] == merged["L_year"]) & (merged["quarter"] == merged["L_quarter"]))]

    # # Drop properties with multiple extensions
    # if "multiple_extensions" in merged.columns:
    #     merged = merged[~merged["multiple_extensions"]]

    # # Create date variable
    # merged["date"] = merged["year"] * 4 + merged["quarter"]

    # # Merge with rsi_full_nd
    # rsi_full_nd_df = pd.read_csv(rsi_full_nd)
    # merged = merged.merge(rsi_full_nd_df, on=["property_id", "date"], how="inner")

    # # Replace extension_amount
    # merged["extension_amount"] = merged.apply(
    #     lambda row: None if not row["extension"] else row["extension_amount"],
    #     axis=1,
    # )
    # merged["extension_amount"] = merged.groupby("property_id")["extension_amount"].transform(
    #     lambda x: x.fillna(x.mean())
    # )

    # # Save final dataset
    # merged.to_stata(f"{working}/for_event_study.dta", write_index=False)


def calculate_housing_risk_premium(data_folder):
    uk_gdp = pd.read_csv(os.path.join(data_folder, "raw", "fred", "UKNGDP.csv"))
    uk_gdp.rename(columns={"DATE": "date", "UKNGDP": "gdp"}, inplace=True)
    uk_gdp["date"] = pd.to_datetime(uk_gdp["date"])
    uk_gdp["year"] = uk_gdp["date"].dt.year
    uk_gdp["quarter"] = uk_gdp["date"].dt.quarter

    # Load leasehold flats data and calculate price and rent for 2022 Q4
    # leasehold_flats = pd.read_pickle(
    #     os.path.join(data_folder, "clean", "leasehold_flats.p")
    # )
    # price2022Q4 = leasehold_flats.loc[
    #     (leasehold_flats["year"] == 2022) & (leasehold_flats["quarter"] == 4), "price"
    # ].mean()
    # rent2022Q4 = leasehold_flats.loc[
    #     (leasehold_flats["date_rent"].dt.year == 2022)
    #     & (leasehold_flats["date_rent"].dt.quarter == 4),
    #     "rent",
    # ].mean()

    price2022Q4 = 311819.6
    rent2022Q4 = 14245.41

    # Load OECD house prices
    house_prices = pd.read_csv(
        os.path.join(data_folder, "raw", "oecd", "house_prices.csv")
    )
    house_prices.columns = house_prices.columns.str.lower()
    house_prices = house_prices[
        (house_prices["location"] == "GBR") & (house_prices["frequency"] == "Q")
    ]
    house_prices["date"] = pd.PeriodIndex(house_prices["time"], freq="Q").to_timestamp()
    house_prices["year"] = house_prices["date"].dt.year
    house_prices["quarter"] = house_prices["date"].dt.quarter
    house_prices = house_prices[
        (house_prices["date"] >= "1970-01-01") & (house_prices["date"] <= "2022-09-30")
    ]

    house_prices = house_prices.pivot_table(
        values="value", index=["year", "quarter", "date"], columns="subject"
    ).reset_index()
    house_prices.rename(
        columns={"NOMINAL": "price_index", "RENT": "rent_index"}, inplace=True
    )
    house_prices.sort_values(by="date", ascending=False, inplace=True)
    house_prices = house_prices[["year", "quarter", "price_index", "rent_index"]]

    # Generate price and rent levels
    house_prices["price"] = price2022Q4
    house_prices["rent"] = rent2022Q4
    for i in range(1, len(house_prices)):
        house_prices.loc[i, "price"] = (
            house_prices.loc[i - 1, "price"]
            * house_prices.loc[i, "price_index"]
            / house_prices.loc[i - 1, "price_index"]
        )
        house_prices.loc[i, "rent"] = (
            house_prices.loc[i - 1, "rent"]
            * house_prices.loc[i, "rent_index"]
            / house_prices.loc[i - 1, "rent_index"]
        )
    house_prices["rent_price"] = house_prices["rent"] / house_prices["price"]

    # Merge with GDP data
    merged = pd.merge(house_prices, uk_gdp, on=["year", "quarter"], how="inner")

    # Load interest rates data
    interest_rates = pd.read_stata(
        os.path.join(data_folder, "clean", "uk_interest_rates.dta")
    )
    interest_rates["quarter"] = interest_rates["date"].dt.quarter
    interest_rates = (
        interest_rates.groupby(["year", "quarter"])
        .agg({"uk10y15": "mean", "uk10y": "mean"})
        .reset_index()
    )

    # Merge all datasets
    final_data = pd.merge(interest_rates, merged, on=["year", "quarter"], how="inner")

    # Define additional variables
    final_data.sort_values(by="date", inplace=True)
    final_data["g"] = final_data["rent"].pct_change()
    final_data["r"] = final_data["uk10y15"] / 100
    final_data["d_gdp"] = final_data["gdp"].pct_change()

    # VAR model
    model = VAR(final_data[["g", "rent_price", "d_gdp"]])
    results = model.fit(maxlags=2)

    # Forecast computation and balanced growth
    T = 30 * 4
    for y in range(160, 252):
        # Add forecast logic using statsmodels or another library here
        pass

    final_data["g_balanced"] = np.nan
    final_data["rtp_balanced"] = np.nan
    for y in range(160, 252):
        # Compute g_balanced and rtp_balanced
        pass

    final_data["rtp_balanced"] *= 100
    final_data["g_balanced"] *= 100
    final_data["r"] *= 100
    final_data["risk_premium"] = (
        final_data["rtp_balanced"] + final_data["g_balanced"] - final_data["r"]
    )

    final_data["date_q"] = final_data["date"]
    final_data[["risk_premium", "g_balanced", "year", "quarter"]].dropna(inplace=True)

    # Save the cleaned dataset
    final_data.to_stata("$clean/risk_premium.dta", write_index=False)


def global_forward_rates(data_folder):
    return


def global_rtp(data_folder):
    return


def rightmove_desriptions(data_folder):
    return
