from utils import *


def ashe_code_edits(df, year):
    if year < 2004:
        df.loc[df["Code"] == 251, "Code"] = 253
    if year < 2008:
        df.loc[df["Code"].isin([271, 272, 275]), "Code"] = 601
        df.loc[df["Code"].isin([270, 273, 276]), "Code"] = 602
        df.loc[df["Code"].isin([253]), "Code"] = 603
        df.loc[df["Code"].isin([250, 252]), "Code"] = 604
    if year < 2011:
        df.loc[df["Code"] == 603, "Code"] = 607
        df.loc[df["Code"] == 604, "Code"] = 608

        df.loc[df["Code"] == 602, "Code"] = 603
        df.loc[df["Code"] == 601, "Code"] = 602
    return df


def convert_to_gss(df, data_folder):
    ashe_codes = pd.read_csv(
        os.path.join(data_folder, "raw", "ons", "ashe", "ashe_codes.csv")
    )
    df = df.merge(ashe_codes, left_on="Code", right_on="ashe_code", how="inner")
    df = df.drop(columns=["Code", "ashe_code"])
    return df


def convert_to_2021_codes(df, data_folder):
    gss_map = pd.read_stata(
        os.path.join(data_folder, "raw", "ons", "gss_codes", "map_gss_over_time.dta")
    )
    for year in range(2008, 2021):
        df[f"gss_code{year}"] = df["gss_code"]
        df = df.merge(gss_map, on=f"gss_code{year}", how="left", indicator=True)
        df.loc[df["_merge"] == "both", "gss_code"] = df["gss_code2021"]

        df = (
            df.groupby(["gss_code", f"gss_code{year}"])
            .agg({f"earn": "mean", "num_jobs": "mean", "Description": "first"})
            .reset_index()
        )

        # Get weighted mean
        df.loc[df.num_jobs == 0, "num_jobs"] = 0.0001
        df["earn"] = df["earn"] * df["num_jobs"]
        df = (
            df.groupby("gss_code")
            .agg(
                {
                    f"earn": sum_with_nan,
                    "num_jobs": sum_with_nan,
                    "Description": "first",
                }
            )
            .reset_index()
        )
        df["earn"] = df["earn"] / df["num_jobs"]
    return df


def sum_with_nan(series):
    if series.isna().any():  # Check if there's any NaN in the series
        return np.nan
    else:
        return series.sum()


def combine_ashe_data(data_folder):
    raw_folder = os.path.join(data_folder, "raw")
    ashe_folder = os.path.join(raw_folder, "ons", "ashe")
    dirs = [
        os.path.join(ashe_folder, d)
        for d in os.listdir(ashe_folder)
        if os.path.isdir(os.path.join(ashe_folder, d))
    ]
    dfs = []

    for d in tqdm(dirs):
        for file in os.listdir(d):
            if "7.1a" in file:
                earnings_file = file
        year = int(re.search(r"\d{4}", earnings_file).group())
        earnings_file_path = os.path.join(d, earnings_file)

        earnings = pd.read_excel(
            earnings_file_path, sheet_name="Male Full-Time", header=4
        )
        earnings = earnings[~earnings["Code"].isna()]
        earnings = earnings[
            ~earnings["Description"].isin(["Inner London", "Outer London"])
        ]
        earnings = earnings.rename(columns={"(thousand)": "num_jobs", "Mean": f"earn"})
        earnings = earnings[["Description", "Code", "num_jobs", f"earn"]]
        earnings = ashe_code_edits(earnings, year)

        # Convert to numeric
        earnings["earn"] = pd.to_numeric(earnings["earn"], errors="coerce")
        earnings["num_jobs"] = pd.to_numeric(earnings["num_jobs"], errors="coerce")

        if year <= 2011:
            earnings = convert_to_gss(earnings, data_folder)
        else:
            earnings = earnings.rename(columns={"Code": "gss_code"})

        earnings = convert_to_2021_codes(earnings, data_folder)
        earnings["year"] = year
        earnings = earnings.rename(columns={"gss_code": "lpa_code"})
        dfs.append(earnings)

    df = pd.concat(dfs)
    df.to_pickle(os.path.join(data_folder, "working", "ashe_earnings.p"))
