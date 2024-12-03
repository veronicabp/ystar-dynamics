from utils import *


def keep_cols(df):
    df["year_from"] = df["date_from"].dt.year
    df["year_registered"] = df["date_registered"].dt.year
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
        "date_rent",
        "L_date_rent",
        "time_on_market",
        "price_change_pct",
        "age",
        "T",
        "k",
        "k90",
        "k700p",
        "T_at_ext",
    ]

    cols_to_keep += [
        col
        for col in df
        if ("price" in col or "pres" in col) and col not in cols_to_keep
    ]
    cols_to_keep += (
        [col for col in hedonics]
        + [f"date_{col}" for col in hedonics]
        + [f"L_date_{col}" for col in hedonics]
    )
    df = df[cols_to_keep]

    df.loc[df.lpa_code.isna(), "lpa_code"] = ""

    return df


def clean_for_dta(file_path, data_folder):
    # Experiments
    df = pd.read_pickle(os.path.join(data_folder, file_path))

    csv_path = os.path.join(data_folder, file_path.replace(".p", ".csv"))
    df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)

    # Convert dates to stata format
    for date in ["date_trans", "L_date_trans"]:
        if date in df.columns:
            stata_base_date = pd.Timestamp("1960-01-01")
            df[date] = pd.to_datetime(df[date])
            df[date] = (df[date] - stata_base_date).dt.days

    stata_path = os.path.join(data_folder, file_path.replace(".p", ".dta"))
    df.to_stata(stata_path, write_index=False)


def output_dta(data_folder):

    for file in [
        "clean/experiments.p",
        "clean/ystar_estimates.p",
        "clean/leasehold_flats.p",
    ]:
        print(f"Saving {file}")
        clean_for_dta(file, data_folder)
