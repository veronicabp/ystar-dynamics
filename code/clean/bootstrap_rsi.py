# %%

from utils import *
from clean.rsi import *
from clean.finalize_experiments import *


def bootstrap_rsi(
    data_folder,
    start_year=1995,
    start_month=1,
    end_year=2024,
    end_month=1,
    bootstrap_iter=1000,
):

    start_date = start_year * 4 + start_month
    end_date = end_year * 4 + end_month
    df = load_data(data_folder)
    df.drop(df[df.years_held < 2].index, inplace=True)
    df = df[df.area == "AL"]

    residuals = pd.read_pickle(os.path.join(data_folder, "working", "residuals.p"))
    df = add_weights(df, residuals)

    # Resample full dataset

    # For each iteration of bootstrap
    ystar_estimates = []

    for b in range(bootstrap_iter):

        boot_sample = resample(df)
        extensions, controls = get_extensions_controls(boot_sample.copy())

        try:
            # 1. Create RSI controls    extensions, controls = get_extensions_controls(df)
            rsi = get_rsi(
                extensions,
                controls,
                start_date=start_date,
                end_date=end_date,
                case_shiller=True,
            )
        except Exception as e:
            print(f"Exception: {e}")
            print(boot_sample.head())
            continue

        rsi_clean = clean_rsi(rsi, "")

        extensions = boot_sample.drop(boot_sample[~boot_sample.extension].index)
        experiments, _, _ = create_experiments(extensions, [rsi_clean], data_folder)

        # 2. Estimate y-star
        ystar, se = estimate_ystar(experiments)
        ystar_estimates.append(ystar)

    # Bootstrap standard errors:
    ystar_estimates = np.array(ystar_estimates)
    std_errors = ystar_estimates.std(axis=0)

    return ystar_estimates, std_errors


# %%
