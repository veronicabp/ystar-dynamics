from utils import *


def calculate_hazard_rate(data_folder, start_year=2003):

    # Load the leasehold flats data
    leasehold_path = os.path.join(data_folder, "clean", "leasehold_flats.p")
    leasehold_df = pd.read_pickle(leasehold_path)

    # Calculate the total and missing extensions
    total_extensions = leasehold_df[
        (leasehold_df["closed_lease"])
        & (leasehold_df["number_years"] == 99)
        & (leasehold_df["extension_amount"].round() == 90)
    ]["property_id"].nunique()

    missing_pre = leasehold_df[
        (leasehold_df["number_years"] == 189) & (leasehold_df["L_date_trans"].isna())
    ]["property_id"].nunique()

    hazard_correction = 1 + (missing_pre / total_extensions)
    print("Hazard correction:", hazard_correction)

    # Round and modify values
    leasehold_df["extension_year"] = np.where(
        leasehold_df["has_been_extended"] | leasehold_df["has_extension"],
        leasehold_df["date_extended"].dt.year,
        np.nan,
    )

    leasehold_df["extension_amount"] = np.floor(leasehold_df["extension_amount"])
    leasehold_df["extension_amount"] = np.where(
        leasehold_df["has_been_extended"] & ~leasehold_df["extension"],
        90,
        leasehold_df["extension_amount"],
    )
    leasehold_df["duration"] = np.floor(leasehold_df["duration"])

    # Drop duplicate property entries
    leasehold_df = leasehold_df[
        ~(leasehold_df["has_extension"] & ~leasehold_df["extension"])
    ].copy()
    leasehold_df.drop_duplicates(subset="property_id", inplace=True)

    # Generate starting duration
    leasehold_df[f"duration{start_year}"] = leasehold_df["duration"] + (
        leasehold_df["year"] - start_year
    )
    leasehold_df.loc[
        (leasehold_df["date_extended"] <= leasehold_df["date_trans"].dt.to_period("D"))
        & (leasehold_df["date_extended"].dt.year > start_year)
        & leasehold_df["extension"],
        f"duration{start_year}",
    ] -= leasehold_df["extension_amount"]

    leasehold_df[f"extended{start_year}"] = leasehold_df["extension_year"] == start_year
    leasehold_df = leasehold_df[leasehold_df[f"duration{start_year}"] >= 0].copy()

    # Generate yearly updates for durations and extensions
    for year in range(start_year, 2024):
        print(year)
        prev_year = year - 1
        if year != start_year:
            leasehold_df[f"duration{year}"] = leasehold_df[f"duration{prev_year}"] - 1
            leasehold_df.loc[
                leasehold_df["extension_year"] == year, f"duration{year}"
            ] += leasehold_df["extension_amount"]
            leasehold_df[f"extended{year}"] = leasehold_df["extension_year"] == year

    # Drop invalid durations
    leasehold_df["mindur"] = leasehold_df[
        [f"duration{year}" for year in range(start_year, 2024)]
    ].min(axis=1)
    leasehold_df = leasehold_df[
        ~(
            (leasehold_df["mindur"] < 0)
            & (leasehold_df["has_been_extended"] | leasehold_df["extension"])
        )
    ].copy()

    id_vars = [
        "property_id",
        "number_years",
        "date_from",
        "date_extended",
        "extension_amount",
        "extension",
        "has_been_extended",
    ]
    cols = id_vars + [
        col
        for col in leasehold_df.columns
        if re.search(r"(duration|extended)\d{4}", col)
    ]
    leasehold_df = leasehold_df[cols].copy()

    leasehold_df["has_been_extended"] = leasehold_df["has_been_extended"].astype(int)
    leasehold_df["extension"] = leasehold_df["extension"].astype(int)

    # Reshape long using Dask to avoid memory issues
    ddf = dd.from_pandas(leasehold_df, npartitions=20)
    dur_df = ddf.melt(
        id_vars=id_vars,
        value_vars=[
            col for col in leasehold_df.columns if re.search(r"duration\d{4}", col)
        ],
        var_name="variable",
        value_name="duration",
    )
    dur_df["year"] = dur_df["variable"].str.extract(r"(\d{4})")[0].astype(int)
    dur_df = dur_df.drop("variable", axis=1)

    ext_df = ddf.melt(
        id_vars=id_vars,
        value_vars=[
            col for col in leasehold_df.columns if re.search(r"extended\d{4}", col)
        ],
        var_name="variable",
        value_name="extended",
    )
    ext_df["year"] = ext_df["variable"].str.extract(r"(\d{4})")[0].astype(int)
    ext_df = ext_df.drop("variable", axis=1)

    leasehold_long = dur_df.merge(ext_df, on=id_vars + ["year"])
    leasehold_long = leasehold_long.compute()

    # Remove dates before the lease was initiated
    leasehold_long = leasehold_long[
        ~(
            (leasehold_long.year < leasehold_long.date_from.dt.year)
            & (
                ~(
                    (leasehold_long.extension == 1)
                    | (leasehold_long.has_been_extended == 1)
                )
            )
        )
    ].copy()
    leasehold_long.dropna(subset="duration", inplace=True)

    leasehold_long.sort_values(by=["property_id", "year"], inplace=True)
    leasehold_long["L_duration"] = leasehold_long.groupby("property_id")[
        "duration"
    ].shift(1)
    leasehold_long["L_duration_bin"] = leasehold_long["L_duration"].apply(
        lambda x: np.round(x / 5) * 5 if x < 70 else x
    )
    leasehold_long = leasehold_long[leasehold_long["L_duration"] != 0]
    leasehold_long.to_pickle(os.path.join(data_folder, "clean", "leasehold_panel.p"))

    # Hazard rate calculation
    hazard_df = leasehold_long[
        (leasehold_long["year"] >= 2010) & (leasehold_long["year"] < 2020)
    ]
    hazard_df = hazard_df.groupby("L_duration", as_index=False).agg(
        {"extended": "mean"}
    )
    hazard_df = hazard_df[hazard_df["L_duration"] >= 80]
    hazard_df["hazard_rate"] = hazard_df["extended"] * hazard_correction

    hazard_df["inv_hazard"] = 1 - hazard_df["hazard_rate"]
    hazard_df["prod_inv_hazard"] = hazard_df["inv_hazard"].cumprod()
    hazard_df["cum_prob"] = 100 * (1 - hazard_df["prod_inv_hazard"])

    # Save results
    hazard_df = hazard_df[["L_duration", "cum_prob"]].rename(
        columns={"L_duration": "whb_duration"}
    )
    hazard_output_path = os.path.join(data_folder, "clean", "hazard_rate.p")
    hazard_df.to_pickle(hazard_output_path)
