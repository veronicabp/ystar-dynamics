from utils import *
from clean.finalize_experiments import *


# def compile_bootstrap(data_folder):

#     df = pd.read_pickle(os.path.join(data_folder, "clean", "leasehold_flats.p"))
#     extensions = df.drop(df[~df.extension].index)

#     results_file = os.path.join(data_folder, "clean", "rsi_bootstraps.csv")
#     if os.path.exists(results_file):
#         results = pd.read_csv(results_file)
#     else:
#         results = pd.DataFrame(
#             {"bootstrap_id": [], "estimate": [], "standard_errors": []}
#         )

#     # Loop through the files and process
#     for file in os.listdir(os.path.join(data_folder, "working", "rsi")):
#         if file.endswith(".p"):
#             b = int(re.search(r"rsi_\d+", file).group().replace("rsi_", ""))

#             if b in results["bootstrap_id"].unique():
#                 continue

#             rsi = pd.read_pickle(os.path.join(data_folder, "working", "rsi", file))
#             rsi_clean = clean_rsi(rsi, "")
#             df_main, _, _ = create_experiments(extensions, [rsi_clean], data_folder)
#             print("Num obs:", len(df_main[df_main.did_rsi.notna()]))

#             ystar, se = estimate_ystar(df_main)

#             # Add the new row of bootstrap id and estimate
#             new_row = pd.DataFrame(
#                 {"bootstrap_id": [b], "estimate": [ystar], "standard_errors": [se]}
#             )
#             results = pd.concat([results, new_row], ignore_index=True)

#             # Save the updated DataFrame to CSV after each iteration
#             results.sort_values(by="bootstrap_id").to_csv(results_file, index=False)

#     # Optionally print the results after the loop
#     ystar_estimates = np.array(results.estimate)
#     print("Estimates:", ystar_estimates)
#     print("Num Estimates:", len(ystar_estimates))
#     print("Bootstrap Std. Err.:", np.std(ystar_estimates))


def compile_bootstrap(data_folder):
    df = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))

    for boot_file in sorted(
        os.listdir(os.path.join(data_folder, "working", "bootstrap"))
    ):
        boot = pd.read_pickle(
            os.path.join(data_folder, "working", "bootstrap", boot_file)
        ).drop_duplicates()
        df = df.merge(boot, on=["property_id", "date_trans"], how="inner")

    baseline_ests = []
    baseline_ses = []
    boot_ests = []
    boot_ses = []

    for boot_col in [col for col in df.columns if col.startswith("d_rsi_boot")]:
        print(boot_col)
        df[f"di{boot_col}"] = df["d_log_price"] - df[boot_col]
        sub = df[df[f"di{boot_col}"].notna()].copy()

        baseline_est, baseline_se = estimate_ystar(sub)
        boot_est, boot_se = estimate_ystar(sub, lhs_var=f"di{boot_col}")

        baseline_ests.append(baseline_est)
        baseline_ses.append(baseline_se)

        boot_ests.append(boot_est)
        boot_ses.append(boot_se)

    print("Bootstrap SE:", np.std(boot_ests))
    print("Mean Basline Robust SE:", np.mean(baseline_ses))
