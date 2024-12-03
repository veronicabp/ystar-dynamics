from utils import *
from clean.rsi import *


def restrictive_controls(row, control_dict):
    # Restrict by duration
    controls = control_dict[row.duration2023]

    if len(controls) == 0:
        return [None] * 5

    # Restrict by distance
    controls["distance"] = controls.apply(
        lambda x: haversine(row.lat_rad, row.lon_rad, x["lat_rad"], x["lon_rad"]),
        axis=1,
    )
    radius = np.ceil(controls["distance"].min())
    controls = controls[controls["distance"] <= radius]

    return [
        controls.log_price.mean(),
        controls.L_log_price.mean(),
        controls.d_log_price.mean(),
        len(controls),
        radius,
    ]


def apply_restrictive_controls(
    extensions,
    control_dict,
    case_shiller=None,
    connect_all=None,
    price_var=None,
    add_constant=None,
):
    extensions[
        [
            "log_price_ctrl",
            "L_log_price_ctrl",
            "d_log_price_ctrl",
            "num_controls",
            "radius",
        ]
    ] = extensions.apply(
        lambda row: restrictive_controls(row, control_dict),
        axis=1,
        result_type="expand",
    )
    return extensions


def construct_restrictive_controls(data_folder, start_date=1995, end_date=2023):
    df = load_data(
        data_folder,
        filepath="clean/leasehold_flats.p",
        extra_cols=["log_price", "L_log_price"],
    )
    df.drop(df[df.years_held <= 2].index, inplace=True)

    df["date"] = df["year"]
    df["L_date"] = df["L_year"]

    df = df[df["date"] != df["L_date"]]
    df = df[df["years_held"] > 2]

    extensions, controls = get_extensions_controls(df)

    result = get_rsi(
        extensions,
        controls,
        start_date=start_date,
        end_date=end_date,
        func=apply_restrictive_controls,
        groupby=["area", "date", "L_date"],
        n_jobs=os.cpu_count(),
    )

    result.to_pickle(os.path.join(data_folder, "working", "restrictive_controls.p"))
