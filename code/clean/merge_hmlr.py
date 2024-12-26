# %%
from utils import *
from clean.fuzzy_merge import *

lightweight_cols = [
    "property_id",
    "date_trans",
    "L_date_trans",
    "year",
    "L_year",
    "quarter",
    "L_quarter",
    "years_held",
    "d_log_price",
    "d_pres_linear",
    "d_pres_main",
    "area",
    "postcode",
    "extension",
    "latitude",
    "longitude",
    "duration2023",
    "whb_duration",
    "date_extended",
    "extension_amount",
    "duration",
    "L_duration",
    "number_years",
    "date_from",
    "not_valid_ext",
]


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

    df["outcode"] = df["postcode"].str.extract(r"^([A-Z]{1,2}[0-9R][0-9A-Z]?)")
    df["incode"] = df["postcode"].str.extract(r"([0-9][A-Z]{2})$")
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
        "region",
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
        os.path.join(data_folder, "working", "rightmove_for_merge.p")
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
    match.rename(
        columns={"property_id_x": "property_id", "property_id_y": "property_id_rm"},
        inplace=True,
    )
    match.drop(
        match[
            (match.uprn_x != match.uprn_y) & match.uprn_x.notna() & match.uprn_y.notna()
        ].index,
        inplace=True,
    )
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


def get_closest_match(df, hedonic, hedonics_date_var):
    merged = df[["date_trans", "property_id"]].merge(hedonic, on="property_id")

    merged["date_diff"] = np.abs(
        (merged["date_trans"] - merged[hedonics_date_var]).dt.total_seconds()
    )
    merged = merged.sort_values(["property_id", "date_trans", "date_diff"])
    merged = merged.drop_duplicates(subset=["date_trans", "property_id"], keep="first")
    merged.drop(["date_diff"], axis=1, inplace=True)
    return merged


def process_closest_variable(
    df,
    df_hedonics,
    vars_to_process,
    hedonics_date_var="date_rm",
):
    df.sort_values(by="date_trans", inplace=True)

    for var in vars_to_process:
        print(f"Processing variable: {var}")

        hedonic = df_hedonics.drop(df_hedonics[df_hedonics[var].isna()].index)[
            ["property_id", hedonics_date_var, var]
        ]

        merged = get_closest_match(df, hedonic, hedonics_date_var)
        merged.rename(
            columns={hedonics_date_var: f"{hedonics_date_var}_{var}"}, inplace=True
        )

        df = df.merge(merged, on=["property_id", "date_trans"], how="left")

    return df


def merge_hedonics_datasets(
    data_folder,
    df,
    merge_keys,
    hedonics_file="rightmove_rents_flats.p",
    tag="rm",
    hedonics_date_var="date_rm_rent",
    rename_dict={
        "listingid": "rentid",
        "listingprice": "rent",
        "date_rm": "date_rm_rent",
    },
    hedonics_vars=["bedrooms", "bathrooms"],
):
    hedonics = pd.read_pickle(os.path.join(data_folder, "working", hedonics_file))

    hedonics = hedonics.merge(
        merge_keys[["property_id", f"property_id_{tag}"]],
        on=f"property_id_{tag}",
        how="inner",
    )

    hedonics.rename(
        columns=rename_dict,
        inplace=True,
    )

    df = process_closest_variable(df, hedonics, hedonics_vars, hedonics_date_var)

    if "rent" in hedonics_file:
        df.rename(columns={var: f"{var}_rent" for var in hedonics_vars}, inplace=True)

    return df


def get_time_on_market(df, merge_keys, data_folder):
    hedonics = pd.read_pickle(
        os.path.join(data_folder, "working", "rightmove_sales_flats.p")
    )

    hedonics = hedonics.merge(
        merge_keys[["property_id", f"property_id_rm"]],
        on=f"property_id_rm",
        how="inner",
    )

    merged = get_closest_match(df, hedonics, "date_rm")
    merged["time_on_market"] = merged.datelist1 - merged.datelist0
    merged["price_change_pct"] = (
        100 * (merged.listprice1 - merged.listprice0) / merged.listprice0
    )
    merged["time_from_listing"] = np.where(
        merged.date_trans > merged.date_rm,
        (merged.date_trans - merged.date_rm).dt.days,
        np.nan,
    )
    df = df.merge(
        merged[
            [
                "property_id",
                "date_trans",
                "time_on_market",
                "time_from_listing",
                "price_change_pct",
            ]
        ],
        on=["property_id", "date_trans"],
        how="left",
    )

    return df


def create_interactions(df, vars_list, interaction_var):
    """
    Creates interaction terms between each variable in vars_list and interaction_var.

    Parameters:
    - df: pandas DataFrame containing the data.
    - vars_list: list of strings, names of variables to interact.
    - interaction_var: string, name of the variable to interact with.

    Returns:
    - df: updated DataFrame with new interaction variables.
    - new_vars: list of names of the new interaction variables.
    """
    new_vars = []
    for var in vars_list:
        new_var = f"{var}_{interaction_var}"
        df[new_var] = df[var].astype(str) + "_" + df[interaction_var].astype(str)
        df[new_var] = df[new_var].astype("category")
        df.loc[(df[var].isna()) | (df[interaction_var].isna()), new_var] = np.nan
        new_vars.append(new_var)
    return df, new_vars


def get_latitude_longitude(df, merge_keys, data_folder):
    hedonics = pd.read_pickle(
        os.path.join(data_folder, "working", "rightmove_sales_flats.p")
    )

    hedonics = hedonics.merge(
        merge_keys[["property_id", f"property_id_rm"]],
        on=f"property_id_rm",
        how="inner",
    )

    merged = get_closest_match(df, hedonics, "date_rm")
    df = df.merge(
        merged[["property_id", "date_trans", "latitude", "longitude"]],
        on=["property_id", "date_trans"],
        how="left",
    )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    return df


def merge_with_hedonics(df, data_folder):

    # Get hedonics merge keys
    hmlr_for_merge = df[["uprn", "property_id", "postcode"]].copy()
    hmlr_for_merge.drop_duplicates(inplace=True)
    hmlr_for_merge["address"] = hmlr_for_merge.swifter.apply(
        lambda row: row.property_id.replace(row.postcode, "").strip(), axis=1
    )

    rm_keys = merge_rightmove(hmlr_for_merge, data_folder)
    zoopla_keys = merge_zoopla(hmlr_for_merge, data_folder)

    # Save
    rm_keys.to_pickle(os.path.join(data_folder, "working", "rightmove_merge_keys.p"))
    zoopla_keys.to_pickle(os.path.join(data_folder, "working", "zoopla_merge_keys.p"))

    ###############################
    # Merge hedonics with HMLR data
    ###############################

    ####### Merge Rightmove rents
    df = merge_hedonics_datasets(
        data_folder,
        df,
        rm_keys,
        hedonics_file="rightmove_rents_flats.p",
        tag="rm",
        hedonics_date_var="date_rm_rent",
        rename_dict={
            "listingid": "rentid",
            "listingprice": "rent",
            "date_rm": "date_rm_rent",
        },
        hedonics_vars=hedonics_rm_full + ["rent"],
    )

    ###### Merge Rightmove hedonics
    df = merge_hedonics_datasets(
        data_folder,
        df,
        rm_keys,
        hedonics_file="rightmove_sales_flats.p",
        tag="rm",
        hedonics_date_var="date_rm",
        rename_dict=dict(),
        hedonics_vars=hedonics_rm_full,
    )

    for var in hedonics_rm:
        df.loc[df[var].isna(), f"date_rm_{var}"] = df[f"date_rm_rent_{var}"]
        df.loc[df[var].isna(), var] = df[f"{var}_rent"]

    ###### Merge Zoopla rents
    df = merge_hedonics_datasets(
        data_folder,
        df,
        zoopla_keys,
        hedonics_file="zoopla_rents_flats.p",
        tag="zoop",
        hedonics_date_var="date_zoopla_rent",
        rename_dict={
            "price_zoopla": "rent_zoopla",
            "date_zoopla": "date_zoopla_rent",
        },
        hedonics_vars=hedonics_zoop + ["rent_zoopla"],
    )

    ##### Merge Zoopla hedonics
    df = merge_hedonics_datasets(
        data_folder,
        df,
        zoopla_keys,
        hedonics_file="zoopla_sales_flats.p",
        tag="zoop",
        hedonics_date_var="date_zoopla",
        rename_dict=dict(),
        hedonics_vars=hedonics_zoop,
    )

    for var in hedonics_zoop:
        df.loc[df[var].isna(), f"date_zoopla_{var}"] = df[f"date_zoopla_rent_{var}"]
        df.loc[df[var].isna(), var] = df[f"{var}_rent"]

    ##### Get time on market
    df = get_time_on_market(df, rm_keys, data_folder)

    ##### Get latitude/longitude
    df = get_latitude_longitude(df, rm_keys, data_folder)

    ##### Other edits
    df.rename(
        columns={
            "rent_rent": "rent",
            "date_rm_rent_rent": "date_rm_rent",
            "rent_zoopla_rent": "rent_zoop",
            "date_zoopla_rent_rent_zoopla": "date_zoopla_rent_zoop",
        },
        inplace=True,
    )
    df["rent_zoop"] *= 52  # Annualize rent

    # Use zoopla if rightmove data is missing
    df.rename(
        columns={
            var: var.replace("date_rm", "date")
            for var in df.columns
            if "date_rm" in var
        },
        inplace=True,
    )
    for var in ["bedrooms", "bathrooms", "rent"]:
        df.loc[(df[var].isna()) | (df[var] == 0), f"date_{var}"] = df[
            f"date_zoopla_{var}_zoop"
        ]
        df.loc[(df[var].isna()) | (df[var] == 0), var] = df[f"{var}_zoop"]

    # Get property age
    df["age"] = df.date_trans.dt.year - df.yearbuilt
    df.loc[df.age < 0, "yearbuilt"] = np.nan
    df.loc[df.age < 0, "age"] = np.nan

    # Remove erroneous values
    df.loc[(df.rent <= 100) | (df.rent > 0.7 * df.price), "rent"] = np.nan

    df["bedrooms_z"] = df["bedrooms"].where(df["bedrooms"] <= 10)
    df["bathrooms_z"] = df["bathrooms"].where(df["bathrooms"] <= 5)
    df["log_rent"] = np.log(df["rent"])
    df["log_rent100"] = 100 * df["log_rent"]

    df["floorarea_50"] = pd.qcut(df["floorarea"], q=50, labels=False, duplicates="drop")
    df["yearbuilt_50"] = pd.qcut(df["yearbuilt"], q=50, labels=False, duplicates="drop")
    df["age_50"] = pd.qcut(df["age"], q=50, labels=False, duplicates="drop")

    # Grouping with missing values treated as a separate category
    for col in ["condition", "heatingtype", "parking"]:
        df[f"{col}_n"] = df[col].astype("string").fillna("MISSING").astype("category")

    # GMS-style hedonics
    for col in ["bedrooms", "bathrooms", "livingrooms"]:
        df[f"{col}_n"] = df[col]
        df.loc[df[col].isnull() | (df[col] > 8), f"{col}_n"] = 99

    df["floorarea_n"] = pd.qcut(
        df["floorarea"], q=50, labels=False, duplicates="drop"
    ).fillna(99)
    df["age_n"] = pd.qcut(df["age"], q=50, labels=False, duplicates="drop").fillna(99)

    # Residualize price on hedonics
    for var in ["bedrooms", "bathrooms", "floorarea_50", "yearbuilt_50"]:
        df = residualize(df, "log_price", [], [var], f"pres_{var}")

    df = residualize(
        df,
        "log_price",
        [],
        ["bedrooms", "bathrooms", "floorarea_50", "yearbuilt_50"],
        "pres_main",
    )
    df = residualize(
        df,
        "log_price",
        [],
        [
            "bedrooms",
            "bathrooms",
            "floorarea_50",
            "yearbuilt_50",
            "condition",
            "heatingtype",
            "parking",
        ],
        "pres_all",
    )
    df = residualize(
        df,
        "log_price",
        [],
        [
            "bedrooms_n",
            "bathrooms_n",
            "floorarea_n",
            "age_n",
            "condition_n",
            "heatingtype_n",
            "parking_n",
        ],
        "pres_all_gms",
    )

    # Linear controls
    df = residualize(
        df,
        "log_price",
        ["bedrooms", "floorarea"],
        [],
        "pres_linear",
    )

    # Quadratic controls
    df["bedrooms2"] = df["bedrooms"] ** 2
    df["floorarea2"] = df["floorarea"] ** 2
    df = residualize(
        df,
        "log_price",
        ["bedrooms", "floorarea", "bedrooms2", "floorarea2"],
        [],
        "pres_quad",
    )
    df.drop(["bedrooms2", "floorarea2"], axis=1, inplace=True)

    # Interact with year FE
    vars_list = ["bedrooms", "bathrooms", "floorarea_50", "yearbuilt_50"]
    df, interaction_vars = create_interactions(df, vars_list, "year")
    df = residualize(df, "log_price", [], interaction_vars, "tpres_main")

    vars_list.extend(["condition", "heatingtype", "parking"])
    df, interaction_vars = create_interactions(df, vars_list, "year")
    df = residualize(df, "log_price", [], interaction_vars, "tpres_all")

    vars_list = [
        "bedrooms_n",
        "bathrooms_n",
        "floorarea_n",
        "age_n",
        "condition_n",
        "heatingtype_n",
        "parking_n",
    ]
    df, interaction_vars = create_interactions(df, vars_list, "year")
    df = residualize(df, "log_price", [], interaction_vars, "tpres_all_gms")

    # Generate all combinations of hedonic characteristics
    varlist = [
        "bedrooms",
        "bathrooms",
        "floorarea_50",
        "yearbuilt_50",
        "condition",
        "heatingtype",
        "parking",
    ]

    combs = get_combinations(varlist)
    combs_time_interaction = get_combinations([f"{var}_year" for var in varlist])

    for count, comb in enumerate(combs):
        df = process_combination(df, comb, count).copy()

    for count, comb in enumerate(combs_time_interaction):
        df = process_combination(df, comb, count, name_prefix="tpres").copy()

    return df


def get_combinations(varlist):
    combinations_list = [
        comb
        for idx, comb in enumerate(
            comb
            for r in range(1, len(varlist) + 1)
            for comb in combinations(varlist, r)
        )
    ]
    return combinations_list


def process_combination(df, comb, count, name_prefix="pres"):
    residual_name = f"{name_prefix}{count}"
    print(f"Processing combination {count}: {comb}")
    return residualize(df, "log_price", [], list(comb), residual_name)


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
    df = df.merge(lat_lon, on="postcode", how="inner")

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

    # Drop unnecessary dates
    df.drop(
        columns=[
            col
            for col in df.columns
            if "date" in col and ("zoopla" in col or "_rent_" in col)
        ],
        inplace=True,
    )

    # Take differences
    print("Taking differences")
    df.sort_values(by=["property_id", "date_trans"], inplace=True)
    vars_to_process = (
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
        + ["age"]
    )

    # Create all new columns in one go
    new_columns = []
    for var in tqdm(vars_to_process):
        if var in df.columns:
            new_columns.append(
                df.groupby("property_id")[var].shift(1).rename(f"L_{var}")
            )
            new_columns.append(
                (df[var] - df.groupby("property_id")[var].shift(1)).rename(f"d_{var}")
            )

    # Concatenate all new columns at once
    df = pd.concat([df] + new_columns, axis=1)
    df = df.copy()

    df.rename(columns={"d_date_trans": "days_held"}, inplace=True)
    df["years_held"] = df.days_held.dt.days / 365.25
    df["years_held_n"] = df["years_held"].round()

    df.drop(
        columns=["d_year", "d_quarter", "d_month", "d_number_years", "d_duration"],
        inplace=True,
    )

    # Identify extensions
    print("Identifying Extensions")
    df["extension"] = (
        (df["duration"] - df["L_duration"] + df["years_held"] > 5)
        & (~df["L_duration"].isna())
        & (~df["duration"].isna())
    )
    df["extension_amount"] = np.where(
        df["extension"],
        df["duration"] - df["L_duration"] + df["years_held"],
        df.extension_amount,
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
            & (~df["L_date_expired"].isna())
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

    ##### Extension date
    # If we have it, set extension date as date of new lease
    df.loc[
        (df.extension)
        & (df["date_registered"] < df["date_trans"].dt.to_period("D"))
        & (df["date_registered"] > df["L_date_trans"].dt.to_period("D")),
        "date_extended",
    ] = df["date_registered"]

    # If date of new lease is missing, use end of last lease
    df.loc[
        (df.extension)
        & (df.date_extended.isna())
        & (df["L_date_expired"] < df["date_trans"].dt.to_period("D"))
        & (df["L_date_expired"] > df["L_date_trans"].dt.to_period("D")),
        "date_extended",
    ] = df["L_date_expired"]

    # If neither is valid, use sale date
    df.loc[
        (df.extension) & (df.date_extended.isna()),
        "date_extended",
    ] = df[
        "date_trans"
    ].dt.to_period("D")

    # Also set extension dates for properties that have extended but we stil don't have a post extension transaction:
    df.loc[
        (df["has_been_extended"]) & (~df["date_expired"].isna()) & (~df.extension),
        "date_extended",
    ] = df["date_expired"]

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

    # Get duration on Jan 1, 2023, so that we can compare duration across properties transacted in different years
    jan2023 = pd.Period("2023-01-01", freq="D")
    df["duration2023"] = df["number_years"] - df.apply(
        lambda row: num_full_years(row.date_from, jan2023), axis=1
    )
    df.loc[df.extension, "duration2023"] = df.duration2023 - df.extension_amount
    df["duration2023"] = df["duration2023"].round()

    df["num_trans"] = df.groupby("property_id")["property_id"].transform("count")

    return df


def num_full_years(start, end):
    years_diff = end.year - start.year
    if end.month < start.month or (end.month == start.month and end.day < start.day):
        years_diff -= 1
    return years_diff


def convert_hedonics_data(data_folder):
    hedonics_folder = os.path.join(data_folder, "working", "hedonics")
    for file in sorted(os.listdir(hedonics_folder)):
        print(f">Converting {file} to pickle.")
        df = pd.read_stata(os.path.join(hedonics_folder, file))
        df.to_pickle(os.path.join(data_folder, "working", file.replace(".dta", ".p")))

    # Rightmove
    rightmove_sales = pd.read_pickle(
        os.path.join(data_folder, "working", "rightmove_sales_flats.p")
    )
    rightmove_rent = pd.read_pickle(
        os.path.join(data_folder, "working", "rightmove_rents_flats.p")
    )
    rightmove = pd.concat([rightmove_sales, rightmove_rent])
    rightmove_for_merge = (
        rightmove[["property_id_rm", "postcode_rm", "uprn", "address1"]]
        .rename(columns={"property_id_rm": "property_id", "postcode_rm": "postcode"})
        .drop_duplicates()
    )
    rightmove_for_merge.to_pickle(
        os.path.join(data_folder, "working", "rightmove_for_merge.p")
    )

    # Zoopla
    zoopla_sales = pd.read_pickle(
        os.path.join(data_folder, "working", "zoopla_sales_flats.p")
    )
    zoopla_rent = pd.read_pickle(
        os.path.join(data_folder, "working", "zoopla_rents_flats.p")
    )
    zoopla = pd.concat([zoopla_sales, zoopla_rent])
    zoopla_for_merge = (
        zoopla[["property_id_zoop", "postcode", "property_number"]]
        .rename(columns={"property_id_zoop": "property_id"})
        .drop_duplicates()
    )
    zoopla_for_merge.to_pickle(
        os.path.join(data_folder, "working", "zoopla_for_merge.p")
    )


def merge_hmlr(data_folder):
    print("Merging HMLR data.")

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
    merged = merge_with_hedonics(merged, data_folder)
    merged.to_pickle(os.path.join(data_folder, "working", "merged_hmlr_hedonics.p"))

    merged = pd.read_pickle(
        os.path.join(data_folder, "working", "merged_hmlr_hedonics.p")
    )

    # Finalize
    merged.drop(merged[~merged.leasehold].index, inplace=True)
    merged.drop(
        columns=[
            col
            for col in merged.columns
            if ("pres" in col or "tpres" in col)
            and not ("pres_main" in col or "pres_linear" in col)
        ],
        inplace=True,
    )

    final = finalize_data(merged, data_folder)
    final.to_pickle(os.path.join(data_folder, "clean", "leasehold_flats.p"))

    # Create lightweight version
    lightweight = final[lightweight_cols]

    lightweight.to_pickle(os.path.join(data_folder, "clean", "leasehold_flats_lw.p"))


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

    # Create lightweight version
    lightweight = final[[col for col in lightweight_cols if col in final.columns]]
    lightweight.to_pickle(os.path.join(data_folder, "clean", "leasehold_flats_lw.p"))


# %%
