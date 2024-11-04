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
    bootstrap_iter=100,
):
    
    london_areas = ['N','NW','W','SW','SE','E','EC','WC']
    area = 'AL'

    # Set up start and end dates
    start_date = start_year * 4 + start_month
    end_date = end_year * 4 + end_month
    df = load_data(data_folder)
    df.drop(df[df.years_held < 2].index, inplace=True)
    df = df[df.area==area]

    # Load residuals and add weights
    residuals = pd.read_pickle(os.path.join(data_folder, "working", "residuals.p"))
    df = add_weights(df, residuals)

    # Define the output CSV path
    output_csv = os.path.join(data_folder, "working", "bootstrap_ystar.csv")
    
    # Load existing results to continue from the next open row
    if os.path.exists(output_csv):
        ystar_df = pd.read_csv(output_csv)
        start_seed = len(ystar_df)  # Seed based on the next row
    else:
        # Create the CSV file with a header if it doesn't exist
        ystar_df = pd.DataFrame(columns=["ystar_estimates"])
        ystar_df.to_csv(output_csv, index=False)
        start_seed = 0

    # Perform bootstrap
    for b in range(start_seed, start_seed + bootstrap_iter):
        print(f'\nBootstrap {b}')
        print('-'*20)

        # Set random seed based on current iteration
        boot_sample = resample(df, random_state=b)
        extensions, controls = get_extensions_controls(boot_sample.copy())

        try:
            # 1. Create RSI controls
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

        # Append the ystar estimate to the CSV
        ystar_df = pd.DataFrame({"ystar estimates": [ystar]})
        ystar_df.to_csv(output_csv, mode='a', index=False, header=False)