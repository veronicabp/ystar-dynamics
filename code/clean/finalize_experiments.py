# %%

from utils import *


def clean_rsi(df, tag):
    # Set 'var' based on the tag
    if tag in ["_linear", "_all"]:
        var = f"d_pres{tag}"
    else:
        var = "d_log_price"

    df[f"did_rsi{tag}"] = df[var] - df["d_rsi"]

    # Rename columns
    df.rename(
        columns={
            "d_rsi": f"d_rsi{tag}",
            "num_controls": f"num_controls{tag}",
            "radius": f"radius{tag}",
        },
        inplace=True,
    )

    # Keep only the specified columns
    keep_cols = [
        "property_id",
        "date_trans",
        f"did_rsi{tag}",
        f"d_rsi{tag}",
        f"num_controls{tag}",
        f"radius{tag}",
    ]
    df = df[keep_cols]

    return df


def combine_rsis(
    data_folder,
    tags=["", "_linear", "_flip", "_bmn", "_yearly", "_nocons"],
):
    # Loop over each tag
    rsi_dfs = []
    for tag in tags:
        # Read the CSV file
        filename = os.path.join(data_folder, "clean", f"rsi{tag}.p")
        df = pd.read_pickle(filename)
        df = clean_rsi(df, tag)
        # Save the data
        rsi_dfs.append(df)

    return rsi_dfs


def create_experiments(df_main, rsi_dfs, data_folder, merge_hazard=True):

    # Loop over each tag and merge the data
    for rsi_df in rsi_dfs:
        df_main = df_main.merge(rsi_df, on=["property_id", "date_trans"], how="left")

    # Set would-have-been duration
    df_main["T"] = df_main["whb_duration"]
    df_main["T_at_ext"] = df_main["L_duration"] - (
        (df_main["date_extended"].dt.to_timestamp() - df_main["L_date_trans"]).dt.days
        / 365.25
    )
    df_main["T5"] = df_main["T"].apply(lambda x: 5 * round(x / 5))
    df_main["T10"] = df_main["T"].apply(lambda x: 10 * round(x / 10))

    # Generate variables
    df_main["k"] = df_main["extension_amount"]
    df_main["k90"] = (df_main["k"].apply(lambda x: 5 * round(x / 5))) == 90
    df_main["k700p"] = df_main["extension_amount"] > 700
    df_main["k90u"] = (
        (df_main["extension_amount"] > 30)
        & (df_main["extension_amount"] < 90)
        & (~df_main["k90"])
    )
    df_main["k200u"] = df_main["extension_amount"] < 200
    df_main["year2"] = (df_main["year"] // 2) * 2
    df_main["year5"] = (df_main["year"] // 5) * 5

    # Merge in hazard rate
    if merge_hazard:
        df_main["whb_duration"] = df_main["whb_duration"].round()
        df_hazard = pd.read_pickle(os.path.join(data_folder, "clean", "hazard_rate.p"))
        df_main = df_main.merge(df_hazard, on="whb_duration", how="left")
        df_main["Pi"] = df_main["cum_prob"] / 100
        df_main["Pi"] = df_main["Pi"].fillna(0)
        df_main.drop(columns=["cum_prob"], inplace=True)

    # Remove 1% of outliers
    for col in df_main.columns:
        if col.startswith("did"):
            lower = df_main[col].quantile(0.005)
            upper = df_main[col].quantile(0.995)
            df_main[f"{col}_win"] = df_main[col].clip(lower, upper)
            df_main.loc[df_main[col] != df_main[f"{col}_win"], col] = np.nan

    # # Remove 1% of outliers by year
    # for col in df_main.columns:
    #     if col.startswith("did"):
    #         # Apply the clipping operation by year
    #         def clip_by_group(group):
    #             lower = group[col].quantile(0.005)
    #             upper = group[col].quantile(0.995)
    #             group[f"{col}_win"] = group[col].clip(lower, upper)
    #             group.loc[group[col] != group[f"{col}_win"], col] = np.nan
    #             return group

    #         # Use groupby to apply the clipping operation per year
    #         df_main = (
    #             df_main.groupby("year").apply(clip_by_group).reset_index(drop=True)
    #         )

    # Keep sample period
    df_main = df_main[df_main["year"] >= 2000]

    # Drop properties at the very low end of the yield curve
    df_main = df_main[df_main["T"] > 30]

    # Drop properties extended within a month of purchase
    df_main["years_diff"] = years_between_dates(
        df_main["date_extended"] - df_main["L_date_trans"].dt.to_period("D")
    )
    df_main = df_main[df_main["years_diff"] > 1 / 12]

    # Create version without flippers
    df_flip = df_main.copy()
    df_main = df_main[df_main["years_held"] >= 2]

    # Drop private data from the Land Registry
    df_public = df_main.copy()
    cols_to_drop = [
        col
        for col in df_public.columns
        if col.endswith(
            ("number_years", "date_from", "date_expired", "date_registered")
        )
    ]
    cols_to_drop.extend(["class_title_code", "deed_date", "unique_id"])
    df_public.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Drop private data from Rightmove
    cols_to_drop = [
        "property_id_rm",
        "postcode_rm",
        "propertytype",
        "newbuildflag",
        "retirementflag",
        "sharedownershipflag",
        "auctionflag",
        "furnishedflag",
        "currentenergyrating",
        "list_date",
        "archive_date",
        "hmlrprice",
        "listingprice",
        "time_on_market",
        "price_change_pct",
        "lat_rm",
        "lon_rm",
        "transtype",
    ]
    df_public.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Drop private data from Zoopla
    cols_to_drop = [col for col in df_public.columns if col.endswith("_z")]
    cols_to_drop.extend(
        [
            "receptions",
            "floors",
            "listing_status",
            "category",
            "property_type",
            "listingid",
        ]
    )
    df_public.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    return df_main, df_flip, df_public


def run_create_experiments(data_folder):
    print("Creating experiments final dataset.")
    df = pd.read_pickle(os.path.join(data_folder, "clean", "leasehold_flats.p"))
    extensions = df.drop(df[~df.extension].index)

    rsi_dfs = []
    for tag in ["", "_bmn", "_yearly", "_postcode", "_hedonics", "_nocons", "_flip"]:
        print(f">Loading rsi{tag}")
        rsi = pd.read_pickle(os.path.join(data_folder, "working", f"rsi{tag}.p"))
        rsi_clean = clean_rsi(rsi, tag)
        rsi_dfs.append(rsi_clean)

    rc = pd.read_pickle(os.path.join(data_folder, "working", f"restrictive_controls.p"))
    rc_clean = clean_rsi(rc.rename(columns={"d_log_price_ctrl": "d_rsi"}), "_rc")
    rc_clean.rename(
        columns={col: col.replace("_rsi", "") for col in rc_clean.columns}, inplace=True
    )

    df_main, df_flip, df_public = create_experiments(
        extensions, rsi_dfs + [rc_clean], data_folder
    )
    df_main.to_pickle(os.path.join(data_folder, "clean", "experiments.p"))
    df_flip.to_pickle(os.path.join(data_folder, "clean", "experiments_flip.p"))
    df_public.to_pickle(os.path.join(data_folder, "clean", "experiments_public.p"))


def update_create_experiments(data_folder, prev_data_folder):
    print("Creating experiments final dataset.")
    df = pd.read_pickle(os.path.join(data_folder, "clean", "leasehold_flats.p"))
    extensions = df.drop(df[~df.extension].index)

    rsi = pd.read_pickle(os.path.join(data_folder, "working", f"rsi.p"))
    rsi_clean = clean_rsi(rsi, "")
    df_main, _, _ = create_experiments(
        extensions, [rsi_clean], data_folder, merge_hazard=False
    )

    # Merge with previous extensions
    df_main.drop(df_main[df_main.did_rsi.isna()].index, inplace=True)
    df_main_old = pd.read_pickle(
        os.path.join(prev_data_folder, "clean", "experiments.p")
    )

    if "_merge" in df_main_old.columns:
        df_main_old.drop(columns=["_merge"], inplace=True)

    df_main_old = df_main_old.merge(
        df_main[["property_id", "date_trans"]],
        on=["property_id", "date_trans"],
        how="left",
        indicator=True,
    )
    df_main_old = df_main_old[df_main_old._merge == "left_only"]

    df_main = pd.concat([df_main, df_main_old])

    df_main.to_pickle(os.path.join(data_folder, "clean", "experiments.p"))
