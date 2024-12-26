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
