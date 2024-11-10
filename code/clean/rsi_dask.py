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
    controls["distance"] = haversine(
        row.lat_rad, row.lon_rad, controls["lat_rad"].values, controls["lon_rad"].values
    )

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
    controls,
    case_shiller=True,
    price_var="d_log_price",
    restrict_price=False,
    add_constant=True,
    radii=[0.1, 0.5] + [i for i in range(1, 21)],
):

    text = ""
    controls, dates, _, radius = restrict_data(
        row, controls, restrict_price=restrict_price, radii=radii, text=text
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
    start_date=1995,
    end_date=2023,
    duration_margin=5,
    parallelize=True,
    groupby="area",
    n_jobs=10,
    client=None,
):

    # Get SLURM environment variables
    cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", default="1"))
    num_tasks = int(os.getenv("SLURM_NTASKS", default="1"))
    num_nodes = int(os.getenv("SLURM_NNODES", default="1"))
    memory_per_node = 32768

    total_cpus = num_tasks * cpus_per_task

    print(f"Num nodes: {num_nodes}")
    print(f"Num tasks: {num_tasks}")
    print(f"CPUs per task: {cpus_per_task}")
    print(f"Total CPUs: {total_cpus}")
    print(f"Memory per node: {memory_per_node}")

    # # Set up Dask LocalCluster
    # cluster = LocalCluster(
    #     n_workers=total_cpus,
    #     threads_per_worker=1,
    #     processes=True,
    #     memory_limit='auto',
    #     local_directory="/tmp",  # Optional: specify a directory for worker data
    # )

    # client = Client(cluster)

    # Set up SLURMCluster
    cluster = LocalCluster(
        n_workers=cpus_per_task,      # One worker per CPU core
        threads_per_worker=1,         # Each worker is single-threaded
        memory_limit='auto',          # Automatically allocate memory for each worker
        local_directory="/tmp"         # Temporary directory for worker data, optional
    )

    client = Client(cluster)

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
        chunks.append((group, control_dict))
    print(f"Num chunks: {len(chunks)}")
    print(f"Skipped {skipped}.")

    scattered_chunks = client.scatter(chunks, broadcast=True, timeout=60)  # timeout in seconds

    def process_partition(chunk):
        extensions_sub, control_dict = chunk
        print(f"Processing chunk with {len(extensions_sub)} rows.")

        # Call get_rsi for this area
        extensions_sub[["d_rsi", "num_controls", "radius"]] = extensions_sub.apply(
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
        return extensions_sub

    # Create delayed tasks
    tasks = [delayed(process_partition)(chunk) for chunk in scattered_chunks]

    # Execute tasks in parallel
    futures = client.compute(tasks)
    progress(futures)
    results = client.gather(futures)

    # Combine results
    output = pd.concat(results)

    client.close()
    # cluster.close()
    return output


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

    print("Num extensions:", len(extensions))
    print("Num controls:", len(controls))

    return extensions, controls


def construct_rsi(
    data_folder, start_year=1995, start_month=1, end_year=2024, end_month=1
):
    # client = Client(scheduler_file='scheduler.json')
    client = None

    start_date = start_year * 4 + start_month
    end_date = end_year * 4 + end_month

    df = load_data(data_folder)
    df.drop(df[df.years_held < 2].index, inplace=True)

    # df["area_count"] = df.groupby("area")["area"].transform("count")
    # df = df[df.area_count <= 25_000]

    df = df[df.area.isin(["AL", "BR"])]
    extensions, controls = get_extensions_controls(df)

    print("Processing the following areas:", sorted(df.area.unique()))

    print("Getting BMN RSI")
    print(f"Num cores: {os.cpu_count()}")
    rsi = get_rsi(
        extensions,
        controls,
        start_date=start_date,
        end_date=end_date,
        case_shiller=False,
        n_jobs=10,
        client=client,
    )
    rsi.to_pickle(os.path.join(data_folder, "clean", "rsi_test2.p"))
