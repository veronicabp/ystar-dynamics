from utils import *


def hazard_rate(data_folder):

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

    # Define the start year
    start_year = 2003

    # Round and modify values
    leasehold_df["extension_year"] = np.where(
        leasehold_df["has_been_extended"] | leasehold_df["has_extension"],
        leasehold_df["date_extended"].dt.year,
        np.nan,
    )

    leasehold_df["extension_amount"] = (
        leasehold_df["extension_amount"].fillna(0).astype(int)
    )
    leasehold_df["extension_amount"] = np.where(
        leasehold_df["has_been_extended"] & ~leasehold_df["extension"],
        90,
        leasehold_df["extension_amount"],
    )
    leasehold_df["duration"] = leasehold_df["duration"].fillna(0).astype(int)

    # Drop duplicate property entries
    leasehold_df = leasehold_df[
        ~(leasehold_df["has_extension"] & ~leasehold_df["extension"])
    ]
    leasehold_df["tag"] = (
        leasehold_df.groupby("property_id")["property_id"].transform("count") == 1
    )
    leasehold_df = leasehold_df[leasehold_df["tag"]]

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
    leasehold_df = leasehold_df[leasehold_df[f"duration{start_year}"] >= 0]

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
    ]

    # Reshape to long format
    leasehold_long = pd.wide_to_long(
        leasehold_df,
        stubnames=["duration", "extended"],
        i=[
            "property_id",
            "number_years",
            "date_from",
            "date_extended",
            "extension_amount",
        ],
        j="year",
    ).reset_index()

    leasehold_long["L_duration"] = leasehold_long.groupby("property_id")[
        "duration"
    ].shift(1)
    leasehold_long["L_duration_bin"] = leasehold_long["L_duration"].apply(
        lambda x: round(x, 5) if x >= 70 else int(x)
    )
    leasehold_long = leasehold_long[leasehold_long["L_duration"] != 0]

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
