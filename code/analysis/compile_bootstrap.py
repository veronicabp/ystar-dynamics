from utils import *
from clean.finalize_experiments import *


def compile_bootstrap(data_folder):
    df = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))

    for boot_file in sorted(
        os.listdir(os.path.join(data_folder, "working", "bootstrap"))
    ):
        if not boot_file.endswith(".p"):
            continue
        boot = pd.read_pickle(
            os.path.join(data_folder, "working", "bootstrap", boot_file)
        ).drop_duplicates()
        df = df.merge(boot, on=["property_id", "date_trans"], how="inner")

    baseline_ests = []
    baseline_ses = []
    boot_ests = []
    boot_ses = []
    boot_sample_size = []

    for boot_col in tqdm([col for col in df.columns if col.startswith("d_rsi_boot")]):
        df[f"di{boot_col}"] = df["d_log_price"] - df[boot_col]
        sub = df[df[f"di{boot_col}"].notna()].copy()

        baseline_est, baseline_se = estimate_ystar(sub, get_se=True)
        boot_est, boot_se = estimate_ystar(sub, lhs_var=f"di{boot_col}", get_se=False)

        baseline_ests.append(baseline_est)
        baseline_ses.append(baseline_se)

        boot_ests.append(boot_est)
        boot_ses.append(boot_se)

        boot_sample_size.append(len(sub))

    print("Bootstrap SE:", np.std(boot_ests))
    print("Mean Basline Robust SE:", np.mean(baseline_ses))
    print("Mean Bootstrap Sample Size:", np.mean(boot_sample_size))

    df = pd.DataFrame(
        {
            "baseline_est": baseline_ests,
            "baseline_se": baseline_ses,
            "boot_est": boot_ests,
            "boot_se": boot_ses,
            "boot_sample_size": boot_sample_size,
        }
    )
    df.to_pickle(os.path.join(data_folder, "clean", "boostrap_estimates.p"))
