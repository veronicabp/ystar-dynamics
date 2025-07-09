from utils import *


# Define a function to format numbers with commas and no decimals
def format_func(value, tick_number):
    formatted_value = "{:,.0f}".format(value)
    return formatted_value


def lpa_map(data_folder, figures_folder):
    raw_folder = os.path.join(data_folder, "raw")
    clean_folder = os.path.join(data_folder, "clean")

    # Define the colors for the colormap
    c1 = "#03045E"
    c2 = "#82EEFD"

    # Create the color map
    colors = [c1, c2]
    blue_cmap = LinearSegmentedColormap.from_list("my_blues", colors)

    # Paths to files
    ons_folder = os.path.join(raw_folder, "ons")
    local_authority_file = os.path.join(
        ons_folder, "LAD_DEC_2021_UK_BFC", "LAD_DEC_2021_UK_BFC.shp"
    )
    regions_file = os.path.join(
        ons_folder, "NUTS1_Jan_2018_UGCB_in_the_UK", "NUTS1_Jan_2018_UGCB_in_the_UK.shp"
    )
    experiments_file = os.path.join(clean_folder, "experiments.dta")

    local_authorities = gpd.read_file(local_authority_file)
    regions = gpd.read_file(regions_file)
    experiments = pd.read_stata(experiments_file)

    # Get number of experiments for each local authority
    experiments = experiments.groupby("lpa_code").size().reset_index(name="count")

    # Merge
    gdf = local_authorities.merge(
        experiments,
        left_on="LAD21CD",
        right_on="lpa_code",
        how="left",
        suffixes=(None, "_rK"),
    )

    ##############
    # Heat map
    ##############
    file = os.path.join(figures_folder, "extension_heatmap.png")
    bin_edges = [10, 50, 100, 200, 300, 500, 1000]

    gdf.loc[
        (gdf["LAD21CD"].str.startswith(("W", "E"))) & (gdf["count"].isna()), "count"
    ] = 0

    fig, ax = plt.subplots(figsize=(20, 20), dpi=300)
    local_authorities.plot(ax=ax, color="gray")
    gdf.plot(
        column="count",
        cmap=blue_cmap,
        ax=ax,
        legend=False,
        scheme="User_Defined",
        classification_kwds=dict(bins=bin_edges),
    )

    regions.boundary.plot(edgecolor="black", ax=ax)

    # Create a color bar legend with a gradient from dark blue to light blue
    norm = Normalize(vmin=min(bin_edges), vmax=max(bin_edges))
    sm = ScalarMappable(cmap=blue_cmap, norm=norm)
    sm._A = []  # Array of data values to associate with the colormap.
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=18)  # Set the font size for the color bar ticks
    cbar.set_label("Count", size=18)  # Set the label for the color bar
    cbar.formatter = FuncFormatter(lambda x, _: "{:,d}".format(int(x)))
    cbar.update_ticks()

    ax.set_xlim([0, 700000])
    ax.set_ylim([0, 1000000])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(file, bbox_inches="tight", pad_inches=0)

    #############################
    # Zoom in on England + Wales
    #############################
    file = os.path.join(figures_folder, "extension_heatmap_zoom.png")
    bin_edges = [10, 50, 100, 200, 300, 500, 1000]

    gdf.loc[
        (gdf["LAD21CD"].str.startswith(("W", "E"))) & (gdf["count"].isna()), "count"
    ] = 0

    fig, ax = plt.subplots(figsize=(20, 20), dpi=300)
    local_authorities.plot(ax=ax, color="gray")
    gdf.plot(
        column="count",
        cmap=blue_cmap,
        ax=ax,
        legend=False,
        scheme="User_Defined",
        classification_kwds=dict(bins=bin_edges),
    )

    regions.boundary.plot(edgecolor="black", ax=ax)

    # Create a color bar legend with a gradient from dark blue to light blue
    norm = Normalize(vmin=min(bin_edges), vmax=max(bin_edges))
    sm = ScalarMappable(cmap=blue_cmap, norm=norm)
    sm._A = []  # Array of data values to associate with the colormap.
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=18)  # Set the font size for the color bar ticks
    cbar.set_label("Count", size=18)  # Set the label for the color bar
    cbar.formatter = FuncFormatter(lambda x, _: "{:,d}".format(int(x)))
    cbar.update_ticks()

    ax.set_xlim([100000, 650000])
    ax.set_ylim([0, 600000])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(file, bbox_inches="tight", pad_inches=0)


def significance_symbol(coeff, se):
    """Return significance symbols based on p-value."""

    t_stat = coeff / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    if p_value < 0.001:
        return "\sym{{*}{*}{*}}"
    elif p_value < 0.01:
        return "\sym{{*}{*}}"
    elif p_value < 0.05:
        return "\sym{*}"
    else:
        return ""


def bootstrap_ses(df, n_bootstrap=1000):
    # Store the estimated parameters
    params_bootstrapped = np.zeros((n_bootstrap, 3))

    # Perform bootstrap resampling
    for i in tqdm(range(n_bootstrap)):
        # Resample the data with replacement
        resample_index = np.random.choice(df.index, size=len(df), replace=True)
        df_resampled = df.loc[resample_index]

        # Estimate parameters on the resampled data
        res = estimate_ystar_alpha(df_resampled)

        # Save the estimated parameters
        params_bootstrapped[i, :] = res.x

    # Compute the standard error for each parameter
    standard_errors = params_bootstrapped.std(axis=0)

    return standard_errors


def estimate_ystar_alpha(df):

    def model(params):
        ystar = params[0] / 100
        alpha_u80 = params[1]
        alpha_o80 = params[2]

        p1 = 1 - np.exp(-ystar * (df["T"] + df["k"]))
        p0 = 1 - np.exp(-ystar * df["T"])
        p0_option_val = (
            df["over80"]
            * (df["Pi"] * (1 - alpha_o80) + (1 - df["Pi"]) * (1 - alpha_u80))
        ) * (np.exp(-ystar * df["T"]) - np.exp(-ystar * (df["T"] + 90)))
        did_est = np.log(p1) - np.log(p0 + p0_option_val)
        return did_est

    def nlls(params):
        return model(params) - df["did_rsi"]

    # Estimate ystar as if there were full holdup
    res = least_squares(
        nlls, x0=[3, 1, 1], bounds=([0, 0, 0], [np.inf, 1, 1]), loss="linear"
    )
    return res


def estimate(df):
    print("Estimating Coefficients")
    coeffs = estimate_ystar_alpha(df).x
    print("Getting Standard Errors")
    ses = bootstrap_ses(df)
    return coeffs, ses


def construct_alpha_table(data_folder, tables_folder):

    file = os.path.join(data_folder, "clean", "experiments.p")
    df = pd.read_pickle(file)
    df = df[~df["did_rsi"].isna()]
    df = df[df.year >= 2003].copy()
    df["over80"] = df["T"] > 80
    pre = df[df.year <= 2010]
    post = df[df.year > 2010]

    coeffs0, ses0 = estimate(pre)
    coeffs1, ses1 = estimate(post)

    coefficients = {"pre": coeffs0, "post": coeffs1}
    std_errors = {"pre": ses0, "post": ses1}

    # Create table
    num_pre = len(pre)
    num_post = len(post)

    # Assuming the coefficients, standard errors and p-values are structured like this:
    variables = [r"$y^*$", r"$\alpha_{t}^{H}$", r"$\alpha_{t}^{L}$"]

    # Create the LaTeX table
    latex_table = r"\begin{tabular}{lcc}" + "\n"
    latex_table += r"\hline" + "\n"
    latex_table += r"& \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)}\\" + "\n"
    latex_table += r"\hline" + "\n"

    for i, var in enumerate(variables):
        latex_table += f"{var}"
        for period in coefficients.keys():
            latex_table += f"& {coefficients[period][i]:.2f}{significance_symbol(coefficients[period][i], std_errors[period][i])}"
        latex_table += r"\\" + "\n"
        for period in coefficients.keys():
            latex_table += f"& ({std_errors[period][i]:.2f})"
        latex_table += r"\\" + "\n"

    latex_table += r"\hline" + "\n"
    latex_table += r"Period & Pre 2010 & Post 2010\\" + "\n"
    latex_table += f"N & {num_pre:,} & {num_post:,}" + r"\\" + "\n"
    latex_table += r"\hline" + "\n"
    latex_table += r"\hline" + "\n"
    latex_table += (
        r"\multicolumn{2}{l}{{\footnotesize{}Standard errors in parentheses}}\\" + "\n"
    )
    latex_table += (
        r"\multicolumn{2}{l}{{\footnotesize{}\sym{*} $p<0.05$, \sym{{*}{*}} $p<0.01$, \sym{{*}{*}{*}} $p<0.001$}}"
        + "\n"
    )
    latex_table += r"\end{tabular}"

    file = os.path.join(tables_folder, "estimate_alphas.tex")
    with open(file, "w") as f:
        f.write(latex_table)


def extract_info(row):
    pattern = r"OESTERREICH (BDSZ )?\d{4} ((.+%)|(ZERO)) \d\d/\d\d/\d\d"
    m = re.search(pattern, row["name"])

    if m:
        info = m.group().replace("OESTERREICH", "").strip()
        bond_type = "BDSZ" if "BDSZ" in info else "Standard"
        info = info.replace("BDSZ", "").strip()
        issue_year = int(re.search(r"\d{4}", info).group())
        info = info.replace(str(issue_year), "").strip()
        coupon_rate = re.search(r"((.+%)|(ZERO))", info).group()
        maturity_str = re.search(r"\d\d/\d\d/\d\d", info).group()

    else:
        search = re.search(r"OESTERREICH (CPN.|PRCL) STRIP \d\d/\d\d/\d\d", row["name"])
        match = search.group()

        if "CPN." in match:
            bond_type = "Coupon Strip"
        else:
            bond_type = "Prachttaler Strip"

        issue_year = np.nan
        coupon_rate = "0"
        maturity_str = re.search(r"\d\d/\d\d/\d\d", match).group()
    return pd.Series(
        [
            issue_year,
            convert_fraction_to_decimal(coupon_rate.replace("%", "")),
            maturity_str,
            bond_type,
        ]
    )


def convert_fraction_to_decimal(s):

    if s == "ZERO":
        return 0.0

    m = re.match(r"^(\d+)/(\d+)$", s)
    if m:
        return int(m.group(1)) / int(m.group(2))

    m = re.match(r"(\d+)\s+(\d+)/(\d+)", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    return float(s)


def extract_maturity_date(row):
    dt = pd.to_datetime(row["maturity_date"], format="%d/%m/%y")
    if (
        dt.year < row["issue_year"]
        or dt < row["date"]
        or any(x in row["name"] for x in ["20/09/17", "30/06/20"])
    ):
        dt = dt.replace(year=dt.year + 100)
    return dt


# def forward_rate_from_yield(zc_df, year, month, low=50, high=90):
#     sub = zc_df[
#         (zc_df["year"] == year)
#         & (zc_df["month"] == month)
#         & (zc_df["term_remaining"].isin([low, high]))
#     ]
#     if len(sub) != 2:
#         return np.nan
#     low_r = sub[sub["term_remaining"] == low]["yield"].values[0]
#     high_r = sub[sub["term_remaining"] == high]["yield"].values[0]
#     return ((1 + high_r) ** high / (1 + low_r) ** low) ** (1 / (high - low)) - 1


def calculate_yield_curve(df, year=2020, month=1):
    sub = df[(df.year == year) & (df.month == month)]

    # Calculate yield curve using QuantLib
    calendar, today, bondSettlementDate, bondSettlementDays = get_calendar(year, month)

    frequency = ql.Annual
    dc = ql.ActualActual(ql.ActualActual.ISMA)
    accrualConvention = ql.ModifiedFollowing
    convention = ql.ModifiedFollowing
    redemption = 100.0

    # print(today)

    # Create bond helpers for each row
    instruments = []
    for i, row in sub.iterrows():

        issue_date = ql.Date(1, int(row["month"]), int(row["issue_year"]))
        maturity = ql.Date(1, int(row["maturity_month"]), int(row["maturity_year"]))
        coupon = row["coupon_rate"] / 100
        price = row["price"]

        if maturity <= today:
            continue

        # print(maturity)

        schedule = ql.Schedule(
            bondSettlementDate,
            maturity,
            ql.Period(frequency),
            calendar,
            accrualConvention,
            accrualConvention,
            ql.DateGeneration.Backward,
            False,
        )
        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(price)),
            bondSettlementDays,
            100.0,
            schedule,
            [coupon],
            dc,
            convention,
            redemption,
        )

        instruments.append(helper)

    params = [bondSettlementDate, instruments, dc]

    # Piecewise approach
    piecewiseMethods = {
        "logLinearDiscount": ql.PiecewiseLogLinearDiscount(*params),
        "logCubicDiscount": ql.PiecewiseLogCubicDiscount(*params),
        "linearZero": ql.PiecewiseLinearZero(*params),
        "cubicZero": ql.PiecewiseCubicZero(*params),
        "linearForward": ql.PiecewiseLinearForward(*params),
        "splineCubicDiscount": ql.PiecewiseSplineCubicDiscount(*params),
    }

    # Fitted approach
    fittingMethods = {
        "NelsonSiegelFitting": ql.NelsonSiegelFitting(),
        "SvenssonFitting": ql.SvenssonFitting(),
        "SimplePolynomialFitting": ql.SimplePolynomialFitting(2),
        "ExponentialSplinesFitting": ql.ExponentialSplinesFitting(),
    }

    fittedBondCurveMethods = {
        label: ql.FittedBondDiscountCurve(*params, method)
        for label, method in fittingMethods.items()
    }

    yield_curves = piecewiseMethods.copy()
    for key in fittedBondCurveMethods:
        yield_curves[key] = fittedBondCurveMethods[key]

    return yield_curves


def get_calendar(year, month):
    calendar = ql.TARGET()
    today = calendar.adjust(ql.Date(1, month, year))
    ql.Settings.instance().evaluationDate = today

    bondSettlementDays = 2
    bondSettlementDate = calendar.advance(today, ql.Period(bondSettlementDays, ql.Days))

    return calendar, today, bondSettlementDate, bondSettlementDays


def forward_rate_from_yield(
    df, year=2020, month=1, low=50, high=90, method="logLinearDiscount"
):

    calendar, today, bondSettlementDate, bondSettlementDays = get_calendar(year, month)
    yield_curves = calculate_yield_curve(df, year=year, month=month)
    curve = yield_curves[method]

    def get_rate(mat):
        return curve.zeroRate(
            calendar.advance(bondSettlementDate, ql.Period(mat, ql.Years)),
            ql.Actual360(),
            ql.Continuous,
        ).rate()

    high = int(
        np.min([high, df[(df.year == year) & (df.month == month)].term_remaining.max()])
        - 1
    )

    low_rate = get_rate(low)
    high_rate = get_rate(high)

    return (((1 + high_rate) ** high) / ((1 + low_rate) ** low)) ** (
        1 / (high - low)
    ) - 1


def calculate_yield(P, T, cr, FV=100):
    if T == 0:
        ytm = npf.irr([-P, FV])

    else:
        cash_flows = [cr * FV] * T
        cash_flows[-1] += FV  # Add FV to the last cash flow
        cash_flows = [-P] + cash_flows  # Initial outflow (negative)

        ytm = npf.irr(cash_flows)
    return ytm


def other_long_run_bonds(data_folder, figures_folder):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    LEGEND_SIZE = 10
    plt.rc("font", size=SMALL_SIZE)
    plt.rc("axes", titlesize=MEDIUM_SIZE)
    plt.rc("axes", labelsize=MEDIUM_SIZE)
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)
    plt.rc("legend", fontsize=LEGEND_SIZE)

    # --- Austrian bond yields ---
    ds_path = os.path.join(data_folder, "raw", "datastream", "AustriaPrice1.xlsx")
    df = pd.read_excel(ds_path, header=None, skiprows=3)

    # Reshape and clean
    names = df.iloc[0, 1:]
    codes = df.iloc[1, 1:]
    df = df.drop([0, 1]).reset_index(drop=True)
    df.columns = ["date"] + [f"{n}_{c}" for n, c in zip(names, codes)]
    df = df.melt(id_vars=["date"], var_name="name_code", value_name="price")
    df = df.dropna(subset=["price"])
    df = df[df["name_code"] != "#ERROR_nan"]
    df = df[df["name_code"].str.contains("DEFAULT PRICE")]

    # Extract fields
    df[["name", "code"]] = df["name_code"].str.split("_", expand=True)
    df["name"] = df["name"].str.replace(" - DEFAULT PRICE", "")
    df[["issue_year", "coupon_rate", "maturity_date", "type"]] = df.apply(
        extract_info, axis=1
    )
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["maturity_date"] = df.apply(extract_maturity_date, axis=1)
    df["term_remaining"] = ((df["maturity_date"] - df["date"]).dt.days + 1) / 365
    df = df[df["type"].isin(["Standard", "Coupon Strip"])].copy()

    df["yield"] = df.progress_apply(
        lambda row: calculate_yield(
            row["price"],
            math.ceil(row["term_remaining"]),
            row["coupon_rate"] / 100,
            FV=100,
        ),
        axis=1,
    )

    # Plot Austrian bond yields
    labels = {
        "OESTERREICH 2013 2.4% 23/05/34": "Expires May 2034",
        "OESTERREICH 2012 3.8% 26/01/62": "Expires Jan. 2062",
        "OESTERREICH 2017 2.1% 20/09/17": "Expires Sep. 2117",
        "OESTERREICH 2020 0.85% 30/06/20": "Expires June 2120",
    }
    plt.figure(figsize=(10, 6))
    for key, lab in labels.items():
        tmp = df[(df["name"] == key) & (df["year"] >= 2015)]
        grp = tmp.groupby("year")["yield"].mean().reset_index()
        plt.plot(grp["year"], grp["yield"] * 100, label=lab, linewidth=3)
    plt.legend(frameon=False)
    plt.ylabel("Yield")
    plt.xticks(range(2015, 2024, 2))
    plt.grid(True)
    plt.savefig(
        os.path.join(figures_folder, "austrian_bond_yields.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --- Austrian forward rates ---
    df = df[df["type"] == "Standard"].copy()

    df["maturity_year"] = df["maturity_date"].dt.year
    df["maturity_month"] = df["maturity_date"].dt.month
    df = (
        df.groupby(
            [
                "year",
                "month",
                "name",
                "coupon_rate",
                "maturity_year",
                "maturity_month",
                "type",
            ]
        )[["price", "term_remaining", "date", "issue_year"]]
        .mean()
        .reset_index()
    )

    df_no_2117 = df[df["name"] != "OESTERREICH 2017 2.1% 20/09/17"].copy()

    fw = df[["year", "month"]].drop_duplicates().reset_index(drop=True)
    fw["date"] = fw["year"] + (fw["month"] - 1) / 12
    fw["r50f40"] = fw.apply(
        lambda r: forward_rate_from_yield(df, int(r["year"]), int(r["month"])), axis=1
    )
    fw["r50f40_exc2117"] = fw.apply(
        lambda r: forward_rate_from_yield(df_no_2117, int(r["year"]), int(r["month"])),
        axis=1,
    )
    plt.figure(figsize=(10, 6))
    plt.plot(fw["date"], fw["r50f40"] * 100, label="50y40 Forward Rate", linewidth=3)
    plt.plot(
        fw["date"],
        fw["r50f40_exc2117"] * 100,
        label="50y40 Forward Rate (Excluding 2117 Bond)",
        linewidth=3,
    )
    plt.legend()
    plt.ylabel("Yield")
    plt.xticks(range(2017, 2024, 2))
    plt.grid(True)
    plt.savefig(
        os.path.join(figures_folder, "austrian_forward_rates.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --- Corporate bond yields ---
    corp_path = os.path.join(data_folder, "raw", "trace", "WRDS_bondret.dta")
    df_corp = pd.read_stata(corp_path)
    df_corp["year"] = pd.to_datetime(df_corp["DATE"]).dt.year

    # MIT bonds
    mit_labels = {
        "US57571KAB08": "Expires Nov. 2096",
        "US575718AA93": "Expires July 2111",
        "US575718AB76": "Expires July 2114",
        "US575718AF80": "Expires July 2116",
    }
    plt.figure(figsize=(10, 6))
    for isin, lab in mit_labels.items():
        tmp = df_corp[(df_corp["company_symbol"] == "MIT") & (df_corp["ISIN"] == isin)]
        grp = tmp.groupby("year")["YIELD"].mean().reset_index()
        plt.plot(grp["year"], grp["YIELD"] * 100, label=lab, linewidth=3)
    plt.legend(frameon=False)
    plt.ylabel("Yield")
    plt.xticks(range(2002, 2024, 2))
    plt.grid(True)
    plt.savefig(
        os.path.join(figures_folder, "MIT_bond_yields.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Long-run corporate bonds
    long_labels = {
        "US191216AF75": "Coca-Cola, Exp: 2093",
        "US459200AP64": "IBM, Exp: 2096",
        "US760719BH68": "HSBC, Exp: 2097",
    }
    plt.figure(figsize=(10, 6))
    for isin, lab in long_labels.items():
        tmp = df_corp[df_corp["ISIN"] == isin]
        grp = tmp.groupby("year")["YIELD"].mean().reset_index()
        plt.plot(grp["year"], grp["YIELD"] * 100, label=lab, linewidth=3)
    plt.legend(frameon=False)
    plt.ylabel("Yield")
    plt.xticks(range(2002, 2024, 2))
    plt.grid(True)
    plt.savefig(
        os.path.join(figures_folder, "long_run_corp_bond_yields.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
