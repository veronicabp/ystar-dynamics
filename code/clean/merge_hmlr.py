# %%
from utils import *
from clean.fuzzy_merge import *


# %%
def get_lease_for_merge(lease_data):
    print("Restricting lease data for merge.")
    lease_data["merge_key_1"] = lease_data["merge_key"]
    lease_data["merge_key_2"] = lease_data["merge_key"]
    lease_data["address"] = lease_data.progress_apply(
        lambda row: row.merge_key.replace(row.postcode, "").strip(), axis=1
    )
    return lease_data[
        ["merge_key", "merge_key_1", "merge_key_2", "uprn", "postcode", "address"]
    ].copy()


def get_price_for_merge(price_data):
    print("Restricting price data for merge.")
    price_data.drop_duplicates(subset="property_id", inplace=True)
    price_data["address"] = price_data.progress_apply(
        lambda row: row.property_id.replace(row.postcode, "").strip(), axis=1
    )
    return price_data[
        ["property_id", "postcode", "address", "merge_key_1", "merge_key_2"]
    ].copy()


def merge_wrapper(price_data, lease_data, pid1="property_id", pid2="merge_key"):
    match, _, _ = fuzzy_merge(
        price_data,
        lease_data,
        pid1=pid1,
        pid2=pid2,
        to_tokenize1="address",
        to_tokenize2="address",
        exact_ids=["merge_key_1", "merge_key_2"],
        output_vars=["property_id", "merge_key", "merged_on"],
    )
    match = pick_best_match(match, pid1=pid1, pid2=pid2)
    return match


def additional_cleaning(df, original_data_folder):
    print("\nNum obs:", len(df))

    # Make number of years an integer
    df["number_years"] = df["number_years"].round()

    df["leasehold"] = df.tenure == "L"
    df["freehold"] = df.tenure == "F"

    # Drop if the lease started a year after the transaction or there is no lease
    len_df = len(df)
    df.drop(
        df[
            (df.leasehold)
            & (
                (df["date_from"].isnull())
                | (df["date_from"] > df["date_trans"].dt.to_period("D") + 365)
            )
        ].index,
        inplace=True,
    )
    print("Dropped", len_df - len(df), "with lease discrepancies.")

    df["years_elapsed"] = years_between_dates(
        df.date_trans.dt.to_period("D") - df.date_from
    )
    df["duration"] = df.number_years - df.years_elapsed

    df["incode"] = df["postcode"].str.extract(r"^([A-Z]{1,2}[0-9R][0-9A-Z]?)")
    df["outcode"] = df["postcode"].str.extract(r"([0-9][A-Z]{2})$")
    df["area"] = df["postcode"].str.extract(r"^([A-Z]{1,2})")
    df["sector"] = df["postcode"].str.extract(r"^([A-Z]{1,2}[0-9R][0-9A-Z]?\s[0-9])")

    df["flat"] = df["type"] == "F"

    postcode_areas = pd.read_stata(
        os.path.join(
            original_data_folder, "raw", "geography", "postcode_area_regions.dta"
        )
    )
    df = df.merge(postcode_areas, on="area", how="left")

    df["log_price"] = np.log(df.price)
    df["log_price100"] = np.log(df.price) * 100

    df["year"] = df.date_trans.dt.year
    df["month"] = df.date_trans.dt.month
    df["quarter"] = df.date_trans.dt.quarter

    # Drop incoherent data
    len_df = len(df)
    df.drop(df[(df.leasehold) & (df.duration <= 0)].index, inplace=True)
    print("Dropped", len_df - len(df), "with negative duration.")

    # Keep only relevant columns
    cols_to_keep = [
        "property_id",
        "date_trans",
        "price",
        "log_price",
        "log_price100",
        "date_from",
        "date_registered",
        "date_expired",
        "number_years",
        "duration",
        "duration2023",
        "leasehold",
        "freehold",
        "type",
        "flat",
        "year",
        "quarter",
        "month",
        "postcode",
        "incode",
        "outcode",
        "area",
        "sector",
        "city",
        "county",
        "district",
        "has_been_extended",
        "extension",
        "extension_amount",
        "years_elapsed",
        "id",
        "unique_id",
        "uprn",
        "term",
        "new",
        "closed_lease",
        "purchased_lease",
    ]
    df = df[list(set(cols_to_keep) & (set(df.columns)))]
    df.rename(columns={"id": "lease_id", "unique_id": "transaction_id"}, inplace=True)

    return df


def merge_purchased_leases(price_data, purchased_leases):
    df = purchased_leases.merge(price_data, on="unique_id", how="right", indicator=True)

    df["purchased_lease"] = df._merge == "both"
    df["closed_lease"] = df["closed_lease"].fillna(False)
    df.drop(columns="_merge", inplace=True)

    return df


def merge_open_leases(df, merge_keys, leases):
    leases.drop(columns="postcode", inplace=True)

    df = df.merge(merge_keys, on="property_id", how="left")
    df = df.merge(leases, on="merge_key", suffixes=("_purch", ""), how="left")

    # For leases identified as extended, check if transaction is pre or post extension
    df.loc[
        (df.extension == True) & (df.date_trans.dt.to_period("D") < df.date_registered),
        "number_years",
    ] = df["number_years (pre_extension)"]
    df.loc[
        (df.extension == True) & (df.date_trans.dt.to_period("D") < df.date_registered),
        "date_from",
    ] = df["date_from (pre_extension)"]
    df.loc[
        (df.extension == True) & (df.date_trans.dt.to_period("D") < df.date_registered),
        "date_registered",
    ] = df["date_registered (pre_extension)"]

    return df


def merge_rightmove(hmlr_data, data_folder):
    """
    Fuzzy merge of HMLR data and Rightmove data
    """
    print("Number of rows (HMLR):", len(hmlr_data.index))

    rightmove_data = pd.read_pickle(
        os.path.join(data_folder, "working", "rightmove_for_merge_flats.p")
    )
    print("Number of rows (Rightmove):", len(rightmove_data.index))

    match, _, _ = fuzzy_merge(
        hmlr_data,
        rightmove_data,
        pid1="property_id_x",
        pid2="property_id_y",
        to_tokenize1="address",
        to_tokenize2="address1",
        exact_ids=["property_id", "uprn"],
        output_vars=["property_id_x", "property_id_y", "uprn_x", "uprn_y", "merged_on"],
    )
    match = match.rename(
        columns={"property_id_x": "property_id", "property_id_y": "property_id_rm"}
    )
    match = match[
        (match.uprn_x == match.uprn_y) | (match.uprn_x.isna()) | (match.uprn_y.isna())
    ]
    match["uprn"] = np.where(
        match["uprn_x"].notnull(), match["uprn_x"], match["uprn_y"]
    )
    match = pick_best_match(match, "property_id_rm", "property_id", only_pid1=True)
    return match


def merge_zoopla(hmlr_data, data_folder):
    """
    Fuzzy merge of HMLR data and Zoopla data
    """
    print("Number of rows (HMLR):", len(hmlr_data.index))

    zoopla_data = pd.read_pickle(
        os.path.join(data_folder, "working", "zoopla_for_merge.p")
    )
    print("Number of rows (Zoopla):", len(zoopla_data.index))

    match, _, _ = fuzzy_merge(
        hmlr_data,
        zoopla_data,
        pid1="property_id_x",
        pid2="property_id_y",
        to_tokenize1="address",
        to_tokenize2="property_number",
        exact_ids=["property_id"],
        output_vars=["property_id_x", "property_id_y", "merged_on"],
    )
    match = match.rename(
        columns={"property_id_x": "property_id", "property_id_y": "property_id_zoop"}
    )
    match = pick_best_match(match, "property_id_zoop", "property_id", only_pid1=True)
    return match


# Function to get first non-missing value
def first_non_missing(series):
    return series.dropna().iloc[0] if not series.dropna().empty else np.nan


# Function to process closest variables
def process_closest_variable(df, vars_to_process, date_var_name):
    for var in vars_to_process:
        print(f"Processing variable: {var}")
        # Calculate absolute difference in dates
        df["diff"] = (df["date_trans"] - df[date_var_name]).abs()
        # Get minimum difference per group
        df["mindiff"] = df.groupby(["property_id", "date_trans"])["diff"].transform(
            "min"
        )
        # Conditions where diff equals mindiff
        condition = (df["diff"] == df["mindiff"]) & df["mindiff"].notna()
        # Assign closest dates and variables
        df.loc[condition, f"date_{var}_closest"] = df.loc[condition, date_var_name]
        df.loc[condition, f"{var}_closest"] = df.loc[condition, var]
        # Get first non-missing value per group
        df[f"date_{var}_closest"] = df.groupby(["property_id", "date_trans"])[
            f"date_{var}_closest"
        ].transform(first_non_missing)
        df[f"{var}_closest"] = df.groupby(["property_id", "date_trans"])[
            f"{var}_closest"
        ].transform(first_non_missing)
        # Drop temporary columns
        df.drop(["diff", "mindiff"], axis=1, inplace=True)
    return df


# Function to rename variables with a suffix
def rename_variables_with_suffix(df, vars_to_rename, suffix):
    rename_dict = {}
    for var in vars_to_rename:
        rename_dict[var] = f"{var}{suffix}"
        rename_dict[f"date_{var}"] = f"date_{var}{suffix}"
    df.rename(columns=rename_dict, inplace=True)
    return df


def merge_with_hedonics(df, data_folder):

    # Get hedonics merge keys
    df["address"] = df.progress_apply(
        lambda row: row.property_id.replace(row.postcode, "").strip(), axis=1
    )
    hmlr_for_merge = df[["uprn", "property_id", "postcode", "address"]].copy()
    hmlr_for_merge.drop_duplicates(inplace=True)

    rm_keys = merge_rightmove(hmlr_for_merge, data_folder)
    zoopla_keys = merge_zoopla(hmlr_for_merge, data_folder)

    ###############################
    # Merge hedonics with HMLR data
    ###############################
    hedonics_rm_full = [
        "bedrooms",
        "bathrooms",
        "floorarea",
        "yearbuilt",
        "livingrooms",
        "parking",
        "heatingtype",
        "condition",
    ]
    hedonics_rm = [
        "bedrooms",
        "bathrooms",
        "floorarea",
        "yearbuilt",
    ]
    hedonics_zoop = [
        "bedrooms",
        "bathrooms",
        "floorarea",
        "yearbuilt",
        "livingrooms",
        "parking",
        "heatingtype",
        "condition",
    ]

    # Merge Rightmove rents
    rm_rents = pd.read_pickle(
        os.path.join(data_folder, "working", "rightmove_rents_flats.p")
    )

    df = df.merge(
        rm_keys[["property_id", "property_id_rm"]], on="property_id", how="left"
    )
    df = df.merge(rm_rents, on="property_id_rm", how="left", suffixes=["", "_rmrent"])

    # Drop and rename columns
    df.drop(
        ["listprice0", "listprice1", "datelist0", "datelist1"], axis=1, inplace=True
    )
    df.rename(columns={"listingid": "rentid", "listingprice": "rent"}, inplace=True)

    # Process closest variables for Rightmove rents
    df = process_closest_variable(df, hedonics_rm_full + ["rent"], "date_rm")

    # Drop original variables and rename closest variables
    df.drop(hedonics_rm_full + ["rent", "date_rm"], axis=1, inplace=True)
    df.rename(
        columns=lambda x: x.replace("_closest", "") if "_closest" in x else x,
        inplace=True,
    )

    # Remove duplicates
    df.drop_duplicates(
        subset=["property_id", "date_trans", "property_id_rm"], inplace=True
    )

    # Rename variables to prevent overwriting
    df = rename_variables_with_suffix(df, hedonics_rm_full, "_rent")

    return df


def finalize_data(df, original_data_folder):

    # Merge in interest rates
    interest_rates = pd.read_stata(
        os.path.join(original_data_folder, "clean", "uk_interest_rates.dta"),
        columns=[
            "year",
            "month",
            "uk1y",
            "uk5y",
            "uk10y",
            "uk25y",
            "uk30y",
            "uk10y20",
            "uk10y15",
            "uk10y15_real",
            "uk10y20_real",
        ],
    )
    df = df.merge(interest_rates, on=["year", "month"], how="left")

    # Merge in LPA codes
    lpa_codes = pd.read_stata(
        os.path.join(original_data_folder, "raw", "geography", "lpa_codes.dta")
    )
    df = df.merge(lpa_codes, on="postcode", how="left")

    # Merge in lat/lon
    df.rename(columns={"latitude": "lat_rm", "longitude": "lon_rm"}, inplace=True)
    lat_lon = pd.read_stata(
        os.path.join(original_data_folder, "raw", "geography", "ukpostcodes.dta")
    )
    df = df.merge(lat_lon, on="postcode", how="left")

    if "lat_rm" in df.columns:
        df.loc[
            (~df.lat_rm.isna())
            & (np.abs(df.latitude - df.lat_rm) < 0.01)
            & (np.abs(df.longitude - df.lon_rm) < 0.01),
            "latitude",
        ] = df.lat_rm
        df.loc[
            (~df.lon_rm.isna())
            & (np.abs(df.latitude - df.lat_rm) < 0.01)
            & (np.abs(df.longitude - df.lon_rm) < 0.01),
            "longitude",
        ] = df.lon_rm

    # Take differences
    df.sort_values(by=["property_id", "date_trans"], inplace=True)
    for var in (
        [
            "year",
            "quarter",
            "month",
            "price",
            "rent",
            "duration",
            "number_years",
            "tenure",
            "closed_lease",
        ]
        + [
            col
            for col in df.columns
            if col.startswith("date")
            or col.startswith("log_price")
            or col.startswith("pres")
            or col.startswith("tpres")
            or col.startswith("log_rent")
            or col.startswith("uk")
        ]
        + hedonics_rm
    ):
        if var in df.columns:
            df[f"L_{var}"] = df.groupby("property_id")[var].shift(1)
            df[f"d_{var}"] = df[var] - df[f"L_{var}"]

    df.rename(columns={"d_date_trans": "days_held"}, inplace=True)
    df["years_held"] = df.days_held.dt.days / 365.25
    df["years_held_n"] = df["years_held"].round()

    df.drop(
        columns=["d_year", "d_quarter", "d_month", "d_number_years", "d_duration"],
        inplace=True,
    )

    # Identify extensions
    df["extension"] = (
        (df["duration"] - df["L_duration"] + df["years_held"] > 5)
        & df["L_duration"].notna()
        & df["duration"].notna()
    )
    df["extension_amount"] = np.where(
        df["extension"], df["duration"] - df["L_duration"] + df["years_held"], np.nan
    )

    df["not_valid_ext"] = (
        (df["extension_amount"] <= 30)
        | (df["duration"] - df["extension_amount"] > 150)
        | (
            (
                years_between_dates(
                    (df["L_date_expired"] - df["date_trans"].dt.to_period("D"))
                )
                > 1
            )
            & df["L_date_expired"].notna()
        )
    ) & df["extension"]

    df.loc[df["not_valid_ext"], "extension"] = False
    df.loc[df["not_valid_ext"], "extension_amount"] = np.nan

    df["L_extension"] = df.groupby("property_id")["extension"].shift(1)

    df["has_extension"] = df.groupby("property_id")["extension"].transform("sum")
    df["multiple_extensions"] = df["has_extension"] > 1

    # Maturity bins
    df["short_lease"] = df["leasehold"] & (df["duration"] <= 100) & ~df["extension"]
    df["med_lease"] = (
        df["leasehold"]
        & (df["duration"] > 100)
        & (df["duration"] <= 300)
        & ~df["extension"]
    )
    df["long_lease"] = df["leasehold"] & (df["duration"] > 300)

    # Extension date
    df["date_extended"] = df["date_registered"]
    df.loc[
        df["date_registered"] > df["date_trans"].dt.to_period("D"), "date_extended"
    ] = df["L_date_expired"]
    df.loc[
        df["date_extended"] > df["date_trans"].dt.to_period("D"), "date_extended"
    ] = df["date_trans"].dt.to_period("D")
    df.loc[df["has_been_extended"] & df["date_expired"].notna(), "date_extended"] = df[
        "date_expired"
    ]

    # Drop impossible values
    df = df[
        ~((df["L_duration"] < 0) | (df["duration"] < 0) | (df["extension_amount"] < 0))
    ]
    df["whb_duration"] = np.where(
        df["extension"], df["L_duration"] - df["years_held"], df["duration"]
    )

    # Create more useful variables
    for var in ["duration", "L_duration", "whb_duration"]:
        df[f"{var}5yr"] = df[var].round(5)
        df[f"{var}10yr"] = df[var].round(10)
        df[f"{var}20yr"] = df[var].round(20)
        df[f"{var}p1000"] = df[var] / 1000

    df["duration2023"] = df["number_years"] - years_between_dates(
        pd.Period("2023-01-01", freq="D") - df.date_from
    )
    df.loc[df.extension, "duration2023"] = df.duration2023 - df.extension_amount
    df["duration2023"] = df["duration2023"].round()

    df["num_trans"] = df.groupby("property_id")["property_id"].transform("count")

    return df


def merge_hmlr(data_folder):

    # Merge open leases
    price_data = pd.read_pickle(
        os.path.join(data_folder, "working", "price_data_leaseholds.p")
    )
    lease_data = pd.read_pickle(os.path.join(data_folder, "working", "lease_data.p"))
    purchased_leases = pd.read_pickle(
        os.path.join(data_folder, "working", "purchased_lease_data.p")
    )
    merge_keys = merge_wrapper(
        get_price_for_merge(price_data.copy()), get_lease_for_merge(lease_data.copy())
    )

    # Merge closed leases
    merged = merge_purchased_leases(price_data, purchased_leases)

    # Merge with open leases
    merged = merge_open_leases(merged, merge_keys, lease_data)

    # Identify extended leases
    merged["has_been_extended"] = (
        (merged.closed_lease)
        & (
            (merged.date_from.dt.year + merged.number_years)
            - (merged.date_from_purch.dt.year + merged.number_years_purch)
            > 30
        )
        & (~merged.date_from.isna())
    )
    merged.loc[merged.has_been_extended, "extension_amount"] = (
        merged.date_from.dt.year + merged.number_years
    ) - (merged.date_from_purch.dt.year + merged.number_years_purch)

    # Use purchased leases when available
    for var in ["date_from", "number_years", "date_registered"]:
        merged.loc[merged.purchased_lease, var] = merged[f"{var}_purch"]
    merged.loc[merged.closed_lease, "date_registered"] = np.nan

    # If we purchased a non-closed lease title for a transaction, use that lease for transactions for which the lease is missing
    for var in ["date_from", "number_years"]:
        merged["temp"] = np.where(
            (merged["purchased_lease"]) & (~merged["closed_lease"]), merged[var], np.nan
        )
        merged.loc[merged[var].isna(), var] = merged.groupby("property_id")[
            "temp"
        ].transform("first")
        merged.drop(columns="temp", inplace=True)

    merged.rename(columns={"date_registered_purch": "date_expired"}, inplace=True)
    merged.loc[merged.closed_lease == False, "date_expired"] = np.nan

    # Append freeholds
    freeholds = pd.read_pickle(
        os.path.join(data_folder, "working", "price_data_freeholds.p")
    )
    merged = pd.concat([merged, freeholds])

    merged = additional_cleaning(merged, data_folder)
    merged.to_pickle(os.path.join(data_folder, "working", "merged_hmlr_all.p"))

    # Keep only flats for main data
    merged.drop(merged[~merged.flat].index, inplace=True)
    merged.to_pickle(os.path.join(data_folder, "working", "merged_hmlr.p"))

    # Merge in hedonics data
    # merged = merge_with_hedonics(merged, data_folder)

    # Finalize
    merged = pd.read_pickle(os.path.join(data_folder, "working", "merged_hmlr.p"))
    final = finalize_data(merged, data_folder)
    final.to_pickle(os.path.join(data_folder, "clean", "flats.p"))

    final.drop(final[~final.leasehold].index, inplace=True)
    final.to_pickle(os.path.join(data_folder, "clean", "leasehold_flats.p"))


def update_hmlr_merge(data_folder, prev_data_folder, original_data_folder):
    # Get new transactions
    price_data = pd.read_pickle(os.path.join(data_folder, "working", "price_data.p"))
    old_price_data = pd.read_pickle(
        os.path.join(prev_data_folder, "working", "price_data_leaseholds.p")
    )

    new_price_data = price_data.merge(
        old_price_data, on=["property_id", "date_trans"], how="left", indicator=True
    )
    new_price_data = new_price_data[new_price_data._merge == "left_only"]
    new_price_data = (
        new_price_data[
            ["property_id", "date_trans"]
            + [col for col in new_price_data.columns if col.endswith("_x")]
        ]
        .rename(columns={col: col.replace("_x", "") for col in new_price_data.columns})
        .copy()
    )

    # Load lease data
    lease_data = pd.read_pickle(os.path.join(data_folder, "working", "lease_data.p"))

    # Merge
    merge_keys = merge_wrapper(
        get_price_for_merge(new_price_data.copy()),
        get_lease_for_merge(lease_data.copy()),
    )

    # 1. For all new transactions, match them to a lease in the new lease data set
    print("Merging new transactions.")
    merged = merge_open_leases(new_price_data, merge_keys, lease_data)
    merged = additional_cleaning(merged, original_data_folder)
    print(len(merged), "new obs.")

    # 2. For all old transactions, make sure there's not a new lease that matches better
    print("Updating old transactions.")
    merged_old = pd.read_pickle(
        os.path.join(prev_data_folder, "working", "merged_hmlr.p")
    )
    merged_old.drop(
        merged_old[(merged_old.freehold) | (~merged_old.flat)].index, inplace=True
    )
    print(len(merged_old), "old obs.")
    merged_old = merged_old.merge(
        lease_data,
        left_on="lease_id",
        right_on="id",
        suffixes=("", "_new"),
        how="left",
        indicator=True,
    )

    for var in ["date_from", "number_years", "date_registered"]:
        merged_old.loc[
            (merged_old._merge == "both")
            & (
                merged_old.date_trans.dt.to_period("D") > merged_old.date_registered_new
            ),
            var,
        ] = merged_old[f"{var}_new"]

    merged_old.drop(
        columns=[
            col for col in merged_old.columns if col.endswith("_new") or col == "_merge"
        ],
        inplace=True,
    )

    # 3. Combine
    combined = pd.concat([merged, merged_old])
    combined.to_pickle(os.path.join(data_folder, "working", "merged_hmlr.p"))

    # Finalize
    combined = pd.read_pickle(os.path.join(data_folder, "working", "merged_hmlr.p"))
    final = finalize_data(combined, original_data_folder)
    final.to_pickle(os.path.join(data_folder, "clean", "leasehold_flats.p"))


# %%
