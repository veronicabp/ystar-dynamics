from utils import *
from clean.rsi import *


def construct_rent_rsi(data_folder, start_date=1995, end_date=2024, n_jobs=6):
    file = os.path.join(data_folder, "working", "for_rent_rsi.p")
    df = pd.read_pickle(file)

    df["date"] = df["year_rm"]
    df["L_date"] = df["L_year_rm"]
    df = df[~df["L_date"].isna()]
    df = df[df["date"] != df["L_date"]]

    extensions, controls = get_extensions_controls(df)

    rsi = get_rsi(
        extensions,
        controls,
        start_date=start_date,
        end_date=end_date,
        case_shiller=False,
        price_var="d_log_rent",
        n_jobs=n_jobs,
    )

    outfile = f"rsi_rent.p"
    rsi.to_pickle(os.path.join(data_folder, "working", outfile))

    rsi = get_rsi(
        extensions,
        controls,
        start_date=start_date,
        end_date=end_date,
        case_shiller=False,
        price_var="d_log_rent_res",
        n_jobs=n_jobs,
    )

    outfile = f"rsi_rent_resid.p"
    rsi.to_pickle(os.path.join(data_folder, "working", outfile))
