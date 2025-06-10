from utils import *


def read_and_append(file1, file2, data_folder, subfolder, sheet="4. spot curve"):
    """
    Reads two Excel files and appends their data into a single DataFrame.
    Args:
        file1 (str): Name of the first Excel file.
        file2 (str): Name of the second Excel file.
        data_folder (str): Path to the folder containing the data files.
        subfolder (str): Subfolder within the data folder where the files are located.
        sheet (str): Name of the sheet to read from the Excel files.
    Returns:
        pd.DataFrame: A DataFrame containing the concatenated data from both files.
    """
    df1 = pd.read_excel(
        os.path.join(data_folder, "raw", "boe", subfolder, file1),
        sheet_name=sheet,
        skiprows=3,
    )
    df2 = pd.read_excel(
        os.path.join(data_folder, "raw", "boe", subfolder, file2),
        sheet_name=sheet,
        skiprows=3,
    )
    return pd.concat([df1, df2], ignore_index=True)


def clean_boe_data(df, tag=""):
    """
    Cleans the Bank of England interest rate data.
    Args:
        df (pd.DataFrame): DataFrame containing the raw interest rate data.
        tag (str): Suffix to append to the column names for differentiation.
    Returns:
        pd.DataFrame: A cleaned DataFrame with the date and interest rates.
    """
    df["date"] = pd.to_datetime(df["years:"], errors="coerce")
    df.rename(
        columns={col: f"uk{col}y{tag}" for col in df.columns if type(col) == int},
        inplace=True,
    )

    desired_years = [1, 5, 10, 25, 30]
    df = df[
        ["date"]
        + [f"uk{y}y{tag}" for y in desired_years if f"uk{y}y{tag}" in df.columns]
    ].copy()

    df.dropna(subset=["date"], inplace=True)

    return df


def get_boe_interest_rates(data_folder):
    """
    Reads and cleans Bank of England interest rate data, including nominal and real rates,
    and calculates forward rates for UK government bonds.
    Args:
        data_folder (str): Path to the folder containing the data files.
    """
    # ------------------------------------------------------------------------------
    # 1. Nominal interest rates
    # ------------------------------------------------------------------------------
    nominal_raw = read_and_append(
        "GLC Nominal month end data_1970 to 2015.xlsx",
        "GLC Nominal month end data_2016 to present.xlsx",
        data_folder,
        "glcnominalmonthedata",
    )
    nominal_df = clean_boe_data(nominal_raw)

    # ------------------------------------------------------------------------------
    # 2. Real interest rates
    # ------------------------------------------------------------------------------
    real_raw = read_and_append(
        "GLC Real month end data_1979 to 2015.xlsx",
        "GLC Real month end data_2016 to present.xlsx",
        data_folder,
        "glcrealmonthedata",
    )
    real_df = clean_boe_data(real_raw, tag="_real")
    merged = pd.merge(nominal_df, real_df, on="date", how="outer")  # or "inner"
    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month
    merged.dropna(subset=["year"], inplace=True)

    # ------------------------------------------------------------------------------
    # 4. Calculate forward rates
    # ------------------------------------------------------------------------------

    # Forward rates:
    merged["uk10y20"] = 100 * (
        ((1 + merged["uk30y"] / 100) ** 30 / (1 + merged["uk10y"] / 100) ** 10)
        ** (1 / 20)
        - 1
    )
    merged["uk10y15"] = 100 * (
        ((1 + merged["uk25y"] / 100) ** 25 / (1 + merged["uk10y"] / 100) ** 10)
        ** (1 / 15)
        - 1
    )
    merged["uk10y20_real"] = 100 * (
        (
            (1 + merged["uk30y_real"] / 100) ** 30
            / (1 + merged["uk10y_real"] / 100) ** 10
        )
        ** (1 / 20)
        - 1
    )
    merged["uk10y15_real"] = 100 * (
        (
            (1 + merged["uk25y_real"] / 100) ** 25
            / (1 + merged["uk10y_real"] / 100) ** 10
        )
        ** (1 / 15)
        - 1
    )

    merged.to_stata(
        os.path.join(data_folder, "clean", "uk_interest_rates.dta"), write_index=False
    )
