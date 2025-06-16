from utils import *
from clean.finalize_experiments import *


def compile_bootstrap(data_folder):
    df = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))

    for boot_file in sorted(
        os.listdir(os.path.join(data_folder, "working", "bayes_bootstrap"))
    ):
        if not boot_file.endswith(".p"):
            continue
        boot = pd.read_pickle(
            os.path.join(data_folder, "working", "bayes_bootstrap", boot_file)
        ).drop_duplicates()
        df = df.merge(boot, on=["property_id", "date_trans"], how="inner")

    boot_ests = []

    for i, boot_col in enumerate(
        tqdm([col for col in df.columns if col.startswith("d_rsi_boot")])
    ):
        df[f"di{boot_col}"] = df["d_log_price"] - df[boot_col]
        sub = df[df[f"di{boot_col}"].notna()].copy()

        # Resample treated properties
        boot_df = sub.sample(n=len(sub), replace=True, random_state=1111 + i)
        boot_est, _ = estimate_ystar(boot_df, lhs_var=f"di{boot_col}", get_se=False)

        boot_ests.append(boot_est)

    print("Bootstrap SE:", np.std(boot_ests))
