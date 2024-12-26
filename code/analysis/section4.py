from utils import *


def constant_r(t, T, r=0.037):
    return r


def recursive_price(T, time=1, r=0.037, g=0.007, D=1, r_func=constant_r):
    if T == 0:
        return 0
    else:
        dividend = D * (1 + g)
        next_price = recursive_price(
            T - 1, r=r, g=g, time=time + 1, D=dividend, r_func=r_func
        )
        return (dividend + next_price) / (1 + r_func(time, T=T, r=r))


def gordon_growth(T, r=0.037, g=0.007, D=1):
    p1tT = 0
    for i in range(1, int(T) + 1):
        p1tT += D * ((1 + g) / (1 + r)) ** i
    return p1tT


def d_log_price(T, r=0.037, g=0.007, k=90):
    return log(gordon_growth(T + k, r=r, g=g)) - log(gordon_growth(T, r=r, g=g))


def nlls(theta, T, y, g=0.007):
    est_y = np.array(list(map(lambda t: d_log_price(t, r=theta + g, g=g, k=90), T)))
    return y - est_y


def objective_function(params, P, T, g, k, func):
    r = params[0]
    estimated_P = func(T=int(T), r=r, g=g, k=k)
    return P - estimated_P


def estimate_ystar(P, T, g=0.007, k=90, func=d_log_price, initial_guess=[0.05]):

    result = least_squares(objective_function, initial_guess, args=(P, T, g, k, func))

    if result.success:
        return result.x[0] - g
    else:
        print("Could not converge!")
        return None


def metafunc(alpha=1, beta=None, gamma=None, delta=None):
    def func(x, r=None, g=0.007, T=None):
        y = 0.037 + alpha * np.e ** (-0.1 * (x + 40))
        if y > 0:
            return y
        else:
            return 0

    return func


# Set multiple possible forward yield functions
def f1(t, T=None, r=0.037, g=0.007):
    return r


def f2(t, T=None, r=0.037, g=0.007):
    rlim = r
    if t <= 40:
        return (rlim - 0.02) + 0.02 / (1 + e ** (-0.2 * (t - 5)))
    else:
        return rlim


def f3(t, T=None, r=0.037, g=0.007):
    rlim = r
    if t <= 40:
        return (rlim + 0.02) - 0.02 / (1 + e ** (-0.2 * (t - 5)))
    else:
        return rlim


def f4(t, T=None, r=0.037, g=0.007):
    rlim = r
    if t <= 37.71:
        return rlim - 0.01 * sin(0.3 * (t - 6))
    else:
        return rlim


def f5(t, T=None, r=0.037, g=0.007):
    if t <= 40:
        return t * (r - 0.01) / 40 + 0.01
    else:
        return r


def f6(t, T=None, r=0.037, g=0.007):
    if t <= 40:
        return g
    else:
        return r


def f_nelson_siegel(
    t, T=None, r=0.036, beta1=0.05, beta2=-0.1, tau=3, adjust=0.0070, g=0.007
):

    t_ns = (t + 2) / tau

    if t <= 40:
        return (
            r
            + adjust
            + beta1 * (e ** (-t_ns))
            + beta2 * ((1 - e ** (-t_ns)) / t_ns - e ** (-t_ns))
        )
    else:
        return r


def plot_functions(
    df,
    var_prefix="rK_T_",
    label_format=r"$y_{}(T)$",
    xlabel="Duration (T)",
    ylabel="Estimated y*",
    file_name="file.png",
    figures_folder="",
):
    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(1, 7):
        ax.plot(
            df.duration,
            100 * df[f"{var_prefix}{i}"],
            label=label_format.format(i),
            color=f"C{i-1}",
            linewidth=2,
        )

    ax.legend(prop={"family": "Times New Roman", "size": 20})
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim([0, 5])

    plt.savefig(os.path.join(figures_folder, file_name))


################ Figure 2 ################
def f_form(alpha, beta, gamma, delta, epsilon, x):
    # Flexible functional form
    return alpha - beta * epsilon ** (-gamma * (x + delta))


def yield_curve_objective(theta, y):
    alpha = theta[0]
    beta = theta[1]
    gamma = theta[2]
    delta = theta[3]
    epsilon = theta[4]

    df = pd.DataFrame({"duration": list(range(0, 41))})
    df["f"] = 1 + df.apply(
        lambda row: f_form(alpha, beta, gamma, delta, epsilon, row.duration), axis=1
    )
    df["cum_forwards"] = df.f.cumprod()
    df["yield_curve"] = df.cum_forwards ** (1 / (df.duration))

    est_s0Y = df.loc[df.duration == 0, "f"].values[0] - 1
    est_r10Y = df.loc[df.duration == 10, "yield_curve"].values[0] - 1
    est_r25Y = df.loc[df.duration == 25, "yield_curve"].values[0] - 1
    est_f10Y15Y = (((1 + est_r25Y) ** 25) / ((1 + est_r10Y) ** 10)) ** (1 / 15) - 1

    est_y = np.array([est_s0Y, est_r10Y, est_f10Y15Y])
    return y - est_y


def create_yield_curve_figures(data_folder, figures_folder):

    df = pd.read_stata(os.path.join(data_folder, "clean", "uk_interest_rates.dta"))
    df = df[(df.year >= 2010) & (df.year < 2020)]

    # Create f curve that passes through the average bond yields (2010-2021) -- weighted by the number of observations in each year
    s0Y = df.uk1y.mean() / 100
    r10Y = df.uk10y.mean() / 100
    f10Y15Y = df.uk10y15.mean() / 100

    # Solve for parameters
    x0 = np.array([0.03, 0.04, 0.5, -10, 1.1])
    y = np.array([s0Y, r10Y, f10Y15Y])
    params = least_squares(yield_curve_objective, x0, args=(y,)).x

    # Turn all of this into a function
    def f_example(t, T=None, r=None, g=0.007):
        return (
            0.002507
            + f_form(params[0], params[1], params[2], params[3], params[4], t)
            + g
        )

    df = pd.DataFrame({"duration": [T for T in range(1, 120)]})
    r = 0.037
    g = 0.007
    Tbar = 40

    # Create price for each function
    functions = [f1, f2, f3, f4, f5, f6, f_nelson_siegel, f_example]
    for i in tqdm(range(1, 9)):
        func = functions[i - 1]
        df[f"f{i}"] = df.apply(lambda row: func(row.duration, r=r, g=g), axis=1) - g

        # Get yield curve
        df[f"forwards_p1{i}"] = 1 + df[f"f{i}"]
        df[f"cum_forwards{i}"] = df[f"forwards_p1{i}"].cumprod()
        df[f"yield_curve{i}"] = df[f"cum_forwards{i}"] ** (1 / (df.duration)) - 1

        df[f"price_Tbar{i}"] = recursive_price(Tbar, r=r, g=g, r_func=func)

        df[f"price{i}"] = df.apply(
            lambda row: recursive_price(row.duration + 90, r=r, g=g, r_func=func),
            axis=1,
        )
        df[f"L_price{i}"] = df.apply(
            lambda row: recursive_price(row.duration, r=r, g=g, r_func=func), axis=1
        )
        df[f"d_log_price{i}"] = np.log(df[f"price{i}"]) - np.log(df[f"L_price{i}"])

        # Estimate constant r for each T
        df[f"ystar_T_{i}"] = df.apply(
            lambda row: estimate_ystar(row[f"d_log_price{i}"], row.duration, g=g),
            axis=1,
        )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df.duration, 100 * df.f8, label=r"$y(s)$", color="black", linewidth=3)
    ax.plot(
        df.duration,
        100 * df.ystar_T_8,
        label=r"$\hat{y}*(s)$",
        color="darkcyan",
        linestyle="dashed",
        linewidth=3,
    )
    ax.scatter(
        df[df.duration == 70].duration,
        100 * df[df.duration == 70].ystar_T_8,
        color="darkcyan",
        s=100,
        label=r"$\hat{y}*(70)$",
    )
    ax.legend(prop={"family": "Times New Roman", "size": 20})
    ax.set_ylabel("Yield")
    ax.set_xlabel(r"Duration ($s$)")

    plt.savefig(os.path.join(figures_folder, "yield_curve_example.png"))

    ################ Appendix Figures ################
    # Plot Forward Curves
    df = df[df.duration <= 100]
    plot_functions(
        df,
        var_prefix="f",
        label_format=r"$y_{}(T)$",
        ylabel="Instantaneous Forward Rate",
        file_name="forward_curves.png",
        figures_folder=figures_folder,
    )

    # Plot Yield Curves
    plot_functions(
        df,
        var_prefix="yield_curve",
        label_format=r"$\rho_{}(T)$",
        ylabel="Yield Curve",
        file_name="yield_curves.png",
        figures_folder=figures_folder,
    )

    # Plot Estimated long run rate
    plot_functions(
        df,
        var_prefix="ystar_T_",
        label_format=r"$y_{}*(T)$",
        ylabel="Estimated y*",
        file_name="natural_rates.png",
        figures_folder=figures_folder,
    )

    # Effect of extending to 160
    df = pd.DataFrame({"duration": list(range(1, 71))})
    df["k"] = 160 - df.duration

    functions = [f1, f2, f3, f4, f5, f6]
    for i in tqdm(range(1, 7)):
        func = functions[i - 1]
        df[f"price{i}"] = df.apply(
            lambda row: recursive_price(160, r=r, g=g, r_func=func), axis=1
        )
        df[f"L_price{i}"] = df.apply(
            lambda row: recursive_price(row.duration, r_func=func, r=r, g=g), axis=1
        )
        df[f"d_log_price{i}"] = np.log(df[f"price{i}"]) - np.log(df[f"L_price{i}"])
        df[f"rhat{i}"] = df.apply(
            lambda row: estimate_ystar(
                row[f"d_log_price{i}"], row.duration, g=g, k=160 - row.duration
            ),
            axis=1,
        )

    # Plot:
    plot_functions(
        df,
        var_prefix="rhat",
        label_format=r"$y_{}*$",
        ylabel="Estimated y*",
        file_name="extension_to_160.png",
        figures_folder=figures_folder,
    )


################ Figure 3: Differencing Out ################


def estimated_long_run_rate(func, T=70):
    p0 = recursive_price(T, r_func=func)
    p1 = recursive_price(T + 90, r_func=func)
    dp = np.log(p1) - np.log(p0)
    y_t = estimate_ystar(dp, T)
    return y_t


def estimated_spot_rate(func):
    R = 1
    P = recursive_price(1000, r_func=func)
    s_t = R / P
    return s_t


def relative_position(x, arr):
    return (x - arr.min()) / (arr.max() - arr.min())


def add_line_collection(ax, x, y, cmap, linestyle="solid"):
    colors = y / y.max()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=3, linestyle=linestyle)
    lc.set_array(colors)
    ax.add_collection(lc)
    return ax


def create_differencing_out_figure(figures_folder):
    # Define a set of functions
    time = int(pi * 2 * 10) + 3
    r = 0.037
    g = 0.007

    funcs = [metafunc(2 * np.sin(0.1 * t)) for t in range(time)]
    x_axis = np.linspace(1, 100, 1000)

    # Get forward curves
    forward_curves = []
    for func in funcs:
        forward_curve = 100 * (np.array([func(x) for x in x_axis]) - g)
        forward_curve[forward_curve < 0] = 0
        forward_curves.append(forward_curve)

    # Get y* and spot curve at each time
    ystars = 100 * np.array([estimated_long_run_rate(func) for func in funcs])
    spot_rates = 100 * np.array([estimated_spot_rate(func) for func in funcs])

    # Create color map based on max/min values
    color_positions = [relative_position(s, spot_rates) for s in spot_rates]

    # Plot yield curves at specific points
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim((-5, 100))
    ax.set_ylim((-0.5, 7))

    labels = ["a", "b", "c", "d", "e", "f", "g"]
    marks = [0, 5, 8, 16, 35, 38, 48]

    for count, t in enumerate(marks):
        color = cmap(color_positions[t])
        ax.plot(x_axis, forward_curves[t], color=color, linewidth=3)
        ax.text(-2, forward_curves[t][0], labels[count], color=color)

    # Mark the first line again so this color is the one that shows in the long end
    ax.plot(x_axis, np.full(len(x_axis), 3), color=cmap(color_positions[0]))

    ax.axhline(y=0, color="black", linestyle="dashed")

    ax.set_xlabel(r"Duration ($s$)")
    ax.set_ylabel(r"$y(s)$")
    ax.text(90, 3.3, r"$y*$", color=cmap(color_positions[0]))

    plt.savefig(os.path.join(figures_folder, "yield_curve_simulation_pt1.png"))

    # Add marks at all places where yield curve is flat
    for t in range(1, time):
        if (
            t + 1 < time
            and (
                forward_curves[t][0] >= (r - g) * 100
                and forward_curves[t + 1][0] < (r - g) * 100
            )
            or (
                forward_curves[t][0] <= (r - g) * 100
                and forward_curves[t + 1][0] > (r - g) * 100
            )
        ):
            marks.append(t)
            labels.append("a")

    # Plot estimated y* and spot curve
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.linspace(0, 10, time)
    ax = add_line_collection(ax, x, ystars, cmap, linestyle="solid")
    ax = add_line_collection(ax, x, spot_rates, cmap, linestyle="dashed")

    ax.set_ylim(2, spot_rates.max() + 0.15)
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)

    # Draw vertical lines at each mark
    for t in marks:
        color = cmap(relative_position(spot_rates[t], spot_rates))
        ax.axvline(x=(t / time) * 10, color=color, linestyle="dotted")

    # Add horizontal line at y*
    ax.axhline(y=(r - g) * 100, color=color, linestyle="dotted", linewidth=3)

    # Legend
    custom_lines = [
        Line2D([0], [0], color=cmap(0.5), lw=4),
        Line2D([0], [0], color=cmap(0.5), lw=4, linestyle="dashed"),
        Line2D([0], [0], color=cmap(0.5), lw=4, linestyle="dotted"),
    ]

    ax.legend(
        custom_lines,
        [r"$\hat{y}*$", r"$R/P$", r"$y*$"],
        loc="upper right",
        prop={"family": "Times New Roman", "size": 20},
    )

    # Axis ticks
    ticks = (np.array(marks) / time) * 10
    ax.set_xticks(ticks, labels=labels)

    # Axis labels
    ax.set_xlabel("Shape of Yield Curve")

    plt.savefig(os.path.join(figures_folder, "yield_curve_simulation_pt2.png"))


def residual_plots(data_folder, figures_folder):
    df = pd.read_pickle(os.path.join(data_folder, "working", "experiment_pids.p"))
    df["experiment"] = df.groupby(["experiment_pid", "experiment_date"]).ngroup()

    # Drop past transaction pairs of treated property used in control index
    df = df[~((df["experiment_pid"] == df["property_id"]) & (df["type"] == "control"))]
    df["extension"] = (df["type"] == "extension").astype(int)

    # Merge with leasehold_flats
    lease_cols = [
        "property_id",
        "date_trans",
        "bedrooms",
        "bathrooms",
        "livingrooms",
        "floorarea",
        "age",
        "log_rent",
        "date_bedrooms",
        "date_bathrooms",
        "date_livingrooms",
        "date_floorarea",
        "date_yearbuilt",
        "date_rent",
        "L_bedrooms",
        "L_bathrooms",
        "L_livingrooms",
        "L_floorarea",
        "L_log_rent",
        "L_date_bedrooms",
        "L_date_bathrooms",
        "L_date_livingrooms",
        "L_date_floorarea",
        "L_date_yearbuilt",
        "L_date_rent",
        "L_date_trans",
    ]
    df_lease = pd.read_pickle(os.path.join(data_folder, "clean", "leasehold_flats.p"))[
        lease_cols
    ]
    df = pd.merge(df, df_lease, on=["property_id", "date_trans"], how="left")

    # Merge with renovations
    renov_cols = ["property_id", "date_trans", "date_rm", "renovated"]
    df_renov = pd.read_pickle(os.path.join(data_folder, "working", "renovations.p"))[
        renov_cols
    ]
    df = pd.merge(df, df_renov, on=["property_id", "date_trans"], how="left")

    df = df.rename(
        columns={
            "property_id": "property_id_c",
            "date_trans": "date_trans_c",
            "experiment_pid": "property_id",
            "experiment_date": "date_trans",
        }
    )

    # Merge with experiments
    exp_cols = ["property_id", "date_trans", "k90", "date_extended"]
    df_exp = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))
    df_exp.dropna(subset=["did_rsi"], inplace=True)
    df_exp = df_exp[exp_cols]
    df = pd.merge(df, df_exp, on=["property_id", "date_trans"], how="inner")

    df["date_age"] = df["date_yearbuilt"]

    for var in ["bedrooms", "bathrooms", "livingrooms", "floorarea", "age", "rent"]:
        date_col = "date_" + var
        if date_col in df.columns:
            df["year_" + var] = df[date_col].dt.year

    # Create labels
    var_labels = {
        "floorarea": "Floor Area",
        "log_rent": "Log Rent",
        "bathrooms": "Bathrooms",
        "bedrooms": "Bedrooms",
        "livingrooms": "Living Rooms",
        "age": "Property Age",
    }

    ##############################################
    # Appendix Figures: Density Plots
    ##############################################

    df = df.copy()
    for var in ["bedrooms", "bathrooms", "livingrooms", "floorarea", "age", "log_rent"]:
        print(f"Getting density for {var_labels[var]}.")
        tmp = df.dropna(subset=[var]).copy()

        # Drop experiments which don't have at least one extension/control with hedonics data
        tmp["num_ext"] = tmp.groupby(["experiment"])["extension"].transform("sum")
        tmp["num_obs"] = tmp.groupby(["experiment"])["extension"].transform("count")
        tmp.drop(
            tmp[(tmp.num_ext == tmp.num_obs) | (tmp.num_ext == 0)].index, inplace=True
        )

        # Determine which year variable to use
        if var == "log_rent":
            year_var = "year_rent"
        else:
            year_var = "year_" + var

        # Residualize
        tmp["interaction"] = (
            tmp["experiment"].astype(str) + "_" + tmp[year_var].astype(str)
        )
        tmp = residualize(tmp, var, [], ["interaction"], f"{var}_res")

        lab = var_labels[var]
        title = f"{lab}, Residualized"

        # Plot density
        plt.figure(figsize=(8, 6))
        sns.kdeplot(
            data=tmp[tmp["extension"] == 1],
            x=var + "_res",
            bw_method=1,
            label="Extended",
            color="blue",
        )
        sns.kdeplot(
            data=tmp[tmp["extension"] == 0],
            x=var + "_res",
            bw_method=1,
            label="Not Extended",
            color="gray",
            linestyle="--",
        )
        plt.xlabel(title)
        plt.ylabel("Density")
        plt.legend(loc="upper right", fontsize="small", frameon=False)
        # plt.title(f"Density of Residualized {lab}")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, f"{var}_kdensity.png"))

    ###############################################
    # Table 2: Placebo Test
    ###############################################
    # Take differences
    df["date_extended"] = df["date_extended"].dt.to_timestamp()

    for var in ["bedrooms", "bathrooms", "livingrooms", "floorarea"]:
        date_var = "date_" + var
        L_date_var = "L_date_" + var
        L_var = "L_" + var
        conditions = (
            (df[date_var] > df["date_extended"])
            & (df["date_extended"] > df[L_date_var])
            & (df[var] >= df[L_var])
        )
        df[f"d_{var}"] = np.where(conditions, df[var] - df[L_var], np.nan)
        df.loc[(df[f"d_{var}"] > 0), f"d_{var}"] = 1

    df["n"] = np.arange(len(df))

    # Take mean
    collapsed = (
        df.groupby(["experiment", "extension"])
        .agg(
            d_bedrooms=("d_bedrooms", "mean"),
            d_bathrooms=("d_bathrooms", "mean"),
            d_livingrooms=("d_livingrooms", "mean"),
            d_floorarea=("d_floorarea", "mean"),
            renovated=("renovated", "mean"),
            count=("n", "count"),
        )
        .reset_index()
    )

    collapsed.to_stata(
        os.path.join(data_folder, "clean", "renovations_by_experiment.dta")
    )
