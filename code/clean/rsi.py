# %%

from utils import *


def haversine(lat1, lon1, lat2, lon2):
    """
    calculate the Haversine distance between two points on the earth in kilometers.

    lat1 : float
            first latitude coordinate
    lon1 : float
            first longitude coordinate
    lat2 : float
            second latitude coordinate
    lon2 : float
            second longitude coordinate
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def restrict_data(
    row,
    control_dict,
    text="",
    restrict_price=False,
    remove_previous=False,
    connect_all=False,
    radii=[0.1, 0.5] + [i for i in range(1, 21)],
):
    # start = time.time()
    # text += f"\n\n\n\n\nFinding controls for {row.property_id}, purchased in {row.L_date_trans} and sold in {row.date_trans} (dur2023={np.round(row.duration2023)})\n"

    # Restrict duration
    controls = control_dict[row.duration2023]

    # Remove previous transactions of this property
    if remove_previous:
        controls = controls.loc[row.property_id != controls["property_id"]]

    # Restrict price
    if restrict_price:
        controls = controls.loc[
            abs(controls["L_log_price_tres"] - row.L_log_price_tres) <= 0.1
        ]

    # Restrict by distance
    controls["distance"] = controls.apply(
        lambda x: haversine(row.lat_rad, row.lon_rad, x["lat_rad"], x["lon_rad"]),
        axis=1,
    )

    # text += f"Full controls:\n{controls[['property_id','duration2023','date','L_date','distance']]}\n\n"

    # Choose smallest distance such that the dates are connected
    dates = None
    for radius in radii:
        sub = controls[controls["distance"] <= radius]

        if len(sub) == 0:
            continue

        uf = get_union(sub)
        connected_dates = uf.build_groups()

        # Check if we want to get an index for all dates, or only pre and post-extension transactions
        if connect_all:
            dates_to_connect = row.dates_to_connect
        else:
            dates_to_connect = [row.date, row.L_date]

        connected = True
        for d in dates_to_connect[1:]:
            if not uf.are_connected(d, dates_to_connect[0]):
                connected = False
                break

        if connected or (
            radius == radii[-1] and uf.are_connected(row.date, row.L_date)
        ):  # If we're at the max radius and the extension window dates are connected, build RSI for those dates
            dates = sorted(uf.get_group(row.date))
            # Make sure number of observations is greater than number of parameters
            sub = sub[sub["date"].isin(dates)]
            if len(sub) > len(dates):
                controls = sub
                break

        elif len(sub) == len(controls):
            break

        # text += f"\nRADIUS TOO SMALL: {radius}\n"
        # text += f"\n{sub[['property_id','date','L_date','duration2023','distance']].head(50)}\n\n\n"
        # connected_dates = uf.build_groups()
        # for key in connected_dates:
        # text += f"{key}: {sorted(connected_dates[key])}\n"

    # text += f"Restricted controls for radius {radius}:\n{controls[['property_id','duration2023','date','L_date','distance']]}\n\n"
    # print(text)

    # end = time.time()
    # print('-->Restrict data time:', end-start)

    return controls, dates, text, radius


def rsi(
    df,
    price_var="d_log_price",
    dummy_vars=[],
    weight_col=None,
    get_se=False,
    get_resid=False,
    add_constant=True,
):

    y = df[price_var]
    X = df[dummy_vars]

    if add_constant:
        X = sm.add_constant(X)

    if weight_col:
        result = sm.WLS(y, X, missing="drop", weights=df[weight_col]).fit()
    else:
        result = sm.OLS(y, X, missing="drop").fit()

    if add_constant:
        constant = result.params["const"]
        params = result.params.drop("const")
        bse = result.bse.drop("const")
    else:
        constant = 0
        params = result.params
        bse = result.bse

    if get_se:
        return [0] + list(params), [0] + list(bse), constant, result.summary()

    elif get_resid:
        return pd.DataFrame(
            {
                "pid_control": df["property_id"],
                "date_trans_control": df["date_trans"],
                "residuals": result.resid,
                "years_held": df["years_held"],
                "distance": df["distance"],
            }
        )
    else:
        return [0] + list(params), constant, result.summary()


def rsi_wrapper(
    row,
    control_dict,
    case_shiller=True,
    price_var="d_log_price",
    restrict_price=False,
    add_constant=True,
    radii=[0.1, 0.5] + [i for i in range(1, 21)],
):

    text = ""
    controls, dates, _, radius = restrict_data(
        row, control_dict, restrict_price=restrict_price, radii=radii, text=text
    )

    if not dates or len(controls) <= 2:
        text += "No controls\n\n"
        return [None, None, None]

    dummy_vars = [f"d_{date}" for i, date in enumerate(dates) if i != 0]

    # Get repeat sales index
    if case_shiller:
        controls["weight"] = 1 / (
            controls["b_cons"]
            + controls["b_years_held"] * controls["years_held"]
            + controls["b_distance"] * controls["distance"]
        )
        if len(controls[controls["weight"] <= 0]) > 0:
            print(f"ERROR: Negative weights")
            controls = controls[controls["weight"] > 0]
        params, constant, summary = rsi(
            controls,
            dummy_vars=dummy_vars,
            price_var=price_var,
            weight_col="weight",
            add_constant=add_constant,
        )
    else:
        params, constant, summary = rsi(
            controls,
            dummy_vars=dummy_vars,
            price_var=price_var,
            add_constant=add_constant,
        )

    d_rsi = (
        params[dates.index(row["date"])] - params[dates.index(row["L_date"])] + constant
    )

    # text += f"{summary}\n\n"
    # text += f"Change of {row['property_id']} controls from {row['L_date']} to {row['date']} is {round(d_rsi,3)} -- constant is {constant}. Vs {row['d_log_price']} for treated"
    # text += f'Radius = {radius}'
    # print(text)

    N = len(controls)
    return [d_rsi, N, radius]


def apply_rsi(
    extensions,
    control_dict,
    price_var="d_log_price",
    case_shiller=True,
    connect_all=False,
    add_constant=True,
):
    extensions[["d_rsi", "num_controls", "radius"]] = extensions.apply(
        lambda row: rsi_wrapper(
            row,
            control_dict,
            price_var=price_var,
            case_shiller=case_shiller,
            add_constant=add_constant,
        ),
        axis=1,
        result_type="expand",
    )
    return extensions


def apply_rsi_residuals(
    extensions,
    control_dict,
    price_var="d_log_price",
    case_shiller=False,
    connect_all=False,
    add_constant=True,
    radii=[0.1, 0.5] + [i for i in range(1, 21)],
):
    # start = time.time()
    # text = ""
    dfs = [pd.DataFrame()]
    for row in extensions.itertuples():
        controls, dates, text, radius = restrict_data(row, control_dict, radii=radii)
        if not dates:
            continue

        dummy_vars = [f"d_{date}" for i, date in enumerate(dates) if i != 0]
        output = rsi(controls, dummy_vars=dummy_vars, get_resid=True)
        output["pid_treated"] = row.property_id
        output["date_trans_treated"] = row.date_trans
        dfs.append(output)

    df = pd.concat(dfs)
    # end = time.time()
    # print('Iteration time:', end-start)
    return df


def apply_rsi_full(
    extensions,
    control_dict,
    price_var="d_log_price",
    case_shiller=False,
    connect_all=False,
    add_constant=True,
    radii=[0.1, 0.5] + [i for i in range(1, 21)],
):

    text = ""
    data = {
        "property_id": [],
        "date_trans": [],
        "date": [],
        "rsi": [],
        "se": [],
        "num_controls": [],
        "radius": [],
        "constant": [],
    }
    for row in extensions.itertuples():
        controls, dates, text, radius = restrict_data(
            row, control_dict, radii=radii, connect_all=connect_all
        )
        if not dates:
            continue

        dummy_vars = [f"d_{date}" for i, date in enumerate(dates) if i != 0]
        params, se, constant, summary = rsi(
            controls, dummy_vars=dummy_vars, get_se=True
        )
        N = len(controls)

        for i, date in enumerate(dates):
            data["property_id"].append(row.property_id)
            data["date_trans"].append(row.date_trans)
            data["date"].append(date)
            data["rsi"].append(params[i])
            data["se"].append(se[i])
            data["num_controls"].append(N)
            data["radius"].append(radius)
            data["constant"].append(constant)

    return pd.DataFrame(data)


def split_into_chunks(grouped_data, threshold_size):
    """
    split data set into smaller chunks of data

    grouped_data : dictionary
            data set separated by categories
    threshold_size : int
            maximum number of rows per data chunk
    """
    chunks = []
    for key, group in grouped_data.items():
        if len(group) > threshold_size:
            n_chunks = ceil(len(group) / threshold_size)
            for chunk in np.array_split(group, n_chunks):
                chunks.append((key, chunk))
        else:
            chunks.append((key, group))
    return chunks


def process_chunk(chunk):
    (
        extensions,
        control_dict,
        case_shiller,
        connect_all,
        add_constant,
        price_var,
        func,
    ) = chunk
    output = func(
        extensions,
        control_dict,
        case_shiller=case_shiller,
        connect_all=connect_all,
        price_var=price_var,
        add_constant=add_constant,
    )
    return output


def get_dummies(df, start_date=1995, end_date=2023, date_var="date"):
    df[date_var] = df[date_var].astype(int)
    df[f"L_{date_var}"] = df[f"L_{date_var}"].astype(int)

    d1 = pd.get_dummies(df[date_var], prefix="d")
    dn1 = pd.get_dummies(df[f"L_{date_var}"], prefix="d")

    new_columns = {}
    for date in range(start_date, end_date + 1):
        col_name = f"d_{date}"
        d1_col = d1[col_name].astype("long") if col_name in d1 else 0
        dn1_col = dn1[col_name].astype("long") if col_name in dn1 else 0
        new_columns[col_name] = d1_col - dn1_col

    new = pd.DataFrame(new_columns)
    # Using pd.concat to add all the new columns at once
    df = pd.concat([df, new], axis=1)
    return df


def get_rsi(
    extensions,
    controls,
    price_var="d_log_price",
    case_shiller=True,
    connect_all=False,
    add_constant=True,
    func=apply_rsi,
    start_date=1995,
    end_date=2023,
    duration_margin=5,
    parallelize=True,
    groupby="area",
    n_jobs=10,
):

    for df in [extensions, controls]:
        df.drop(df[df[price_var].isna()].index, inplace=True)
        df["lat_rad"] = np.deg2rad(df["latitude"])
        df["lon_rad"] = np.deg2rad(df["longitude"])

    # Get dummies
    controls = get_dummies(controls, start_date=start_date, end_date=end_date)

    # Group by postcode area
    threshold_size = np.minimum(len(extensions) / n_jobs, 500)
    extensions_grouped = split_into_chunks(
        {name: group for name, group in extensions.groupby(groupby)}, threshold_size
    )
    controls_grouped = {name: group for name, group in controls.groupby(groupby)}

    skipped = 0
    chunks = []
    for key, group in extensions_grouped:
        durations_list = group["duration2023"].unique()
        # print(key, durations_list)

        if key not in controls_grouped:
            skipped += 1
            continue

        controls_subgroup = controls_grouped[key]
        control_dict = {}
        for duration2023 in durations_list:
            control_dict[duration2023] = controls_subgroup[
                abs(controls_subgroup["duration2023"] - duration2023) <= duration_margin
            ].copy()
        chunks.append(
            (
                group,
                control_dict,
                case_shiller,
                connect_all,
                add_constant,
                price_var,
                func,
            )
        )
    print(f"Num chunks: {len(chunks)}")
    print(f"Skipped {skipped}.")

    results = pqdm(chunks, process_chunk, n_jobs=n_jobs)

    valid_results = [r for r in results if isinstance(r, (pd.DataFrame, pd.Series))]
    invalid_results = [
        r for r in results if not isinstance(r, (pd.DataFrame, pd.Series))
    ]
    if len(invalid_results) > 0:
        print(f"Error! {invalid_results}")

    output = pd.concat(valid_results)
    return output


def add_weights(df, residuals):
    X = residuals[["years_held", "distance"]]
    X = sm.add_constant(X)
    y = residuals["residuals"] ** 2
    result = sm.OLS(y, X).fit()

    df["b_cons"] = result.params[0]
    df["b_years_held"] = result.params[1]
    df["b_distance"] = result.params[2]
    return df


def restrict_columns(df):
    df = df[
        [
            "property_id",
            "date_trans",
            "L_date_trans",
            "year",
            "L_year",
            "quarter",
            "L_quarter",
            "years_held",
            "d_log_price",
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
        ]
    ]
    return df


def load_data(folder, restrict_cols=True):
    df = pd.read_pickle(os.path.join(folder, "clean", "leasehold_flats.p"))

    # Drop impossible controls
    len_df = len(df)
    df.drop(
        df[df.number_years < 50].index, inplace=True
    )  # Short lease - likely to be a commercial lease

    max_dur = df[df.extension].duration2023.max() + 5
    df.drop(
        df[(~df.extension) & (df.duration > max_dur)].index, inplace=True
    )  # Duration is too long to be a control

    df.drop(
        df[df.not_valid_ext].index, inplace=True
    )  # Weird extension-like case that we do not know how to classify

    df.drop(
        df[df.L_date_trans.isna()].index, inplace=True
    )  # First transaction of property -- for RSI we need two
    print(f"Dropped {len_df - len(df)} impossible controls.")

    # Keep only relevant columns
    if restrict_cols:
        df = restrict_columns(df)

    # Create a date variable
    df["date"] = df["year"] * 4 + df["quarter"]
    df["L_date"] = df["L_year"] * 4 + df["L_quarter"]
    df = df[df["date"] != df["L_date"]]

    return df


def get_extensions_controls(df):
    extensions = df.drop(df[~df.extension].index)
    controls = df.drop(df[(df.extension)].index)

    # print('\n\nExtensions:')
    # print(extensions)
    print("Num extensions:", len(extensions))

    # print('\n\nControls:')
    # print(controls)
    print("Num controls:", len(controls))

    return extensions, controls


def construct_rsi(
    data_folder, start_year=1995, start_month=1, end_year=2024, end_month=1
):

    start_date = start_year * 4 + start_month
    end_date = end_year * 4 + end_month

    df = load_data(data_folder)

    # # Get weights
    print("Getting residuals")
    # extensions, controls = get_extensions_controls(df.drop(df[df.years_held<2].index)):
    # residuals = get_rsi(extensions, controls, start_date=start_date, end_date=end_date, func=apply_rsi_residuals)
    # residuals.to_pickle(os.path.join(data_folder, 'working', 'residuals.p'))

    residuals = pd.read_pickle(os.path.join(data_folder, "working", "residuals.p"))
    df = add_weights(df, residuals)

    # # Get RSI - including flippers
    # print('Getting RSI with flippers')
    # extensions, controls = get_extensions_controls(df)
    # rsi = get_rsi(extensions, controls, start_date=start_date, end_date=end_date, case_shiller=True)
    # rsi.to_pickle(os.path.join(data_folder, 'clean', 'rsi_flip.p'))

    # # RSI - excluding flippers (Baseline Method)
    # print("Getting baseline RSI")
    df.drop(df[df.years_held < 2].index, inplace=True)
    extensions, controls = get_extensions_controls(df)
    # rsi = get_rsi(
    #     extensions,
    #     controls,
    #     start_date=start_date,
    #     end_date=end_date,
    #     case_shiller=True,
    # )
    # rsi.to_pickle(os.path.join(data_folder, "clean", "rsi.p"))

    # Hedonics variation

    # # No weights
    # print("Getting BMN RSI")
    # print(f'Num cores: {os.cpu_count()}')
    # rsi = get_rsi(
    #     extensions,
    #     controls,
    #     start_date=start_date,
    #     end_date=end_date,
    #     case_shiller=False,
    #     n_jobs=os.cpu_count()
    # )
    # rsi.to_pickle(os.path.join(data_folder, "clean", "rsi_bmn.p"))

    # # Yearly
    print("Getting annual RSI")
    df["date"] = df["year"]
    df["L_date"] = df["L_year"]
    df.drop(df[df.date == df.L_date].index, inplace=True)
    extensions, controls = get_extensions_controls(df)
    # rsi = get_rsi(
    #     extensions,
    #     controls,
    #     start_date=start_year,
    #     end_date=end_year,
    #     case_shiller=True,
    # )
    # rsi.to_pickle(os.path.join(data_folder, "clean", "rsi_yearly.p"))

    # Postcode RSI
    print("Postcode RSI:")
    rsi = get_rsi(
        extensions,
        controls,
        start_date=start_year,
        end_date=end_year,
        case_shiller=True,
        groupby="postcode",
    )
    rsi.to_pickle(os.path.join(data_folder, "clean", "rsi_postcode.p"))


def update_rsi(
    data_folder,
    prev_data_folder,
    original_data_folder,
    start_date=1995 * 4 + 1,
    end_date=2024 * 4 + 6,
):
    df = load_data(data_folder)

    # Load weights from original version
    residuals = pd.read_pickle(
        os.path.join(original_data_folder, "working", "residuals.p")
    )
    df = add_weights(df, residuals)

    # Process all new extensions + any extensions in the last year
    old_df = pd.read_pickle(
        os.path.join(prev_data_folder, "clean", "leasehold_flats.p")
    )
    extensions = df[df.extension].merge(
        old_df[["property_id", "date_trans"]],
        on=["property_id", "date_trans"],
        how="left",
        indicator=True,
    )
    extensions = extensions[
        (extensions._merge == "left_only")
        | ((extensions.date_trans.max() - extensions.date_trans).dt.days < 365)
    ].copy()
    controls = df.drop(df[(df.extension)].index)

    print("\n\nExtensions:")
    print(extensions)
    print("Min date:", extensions.date_trans.min())
    print("Max date:", extensions.date_trans.max())

    print("\n\nControls:")
    print(controls)

    # Construct RSI
    rsi = get_rsi(
        extensions,
        controls,
        start_date=start_date,
        end_date=end_date,
        case_shiller=True,
    )
    rsi.to_pickle(os.path.join(data_folder, "clean", "rsi.p"))


# %%
