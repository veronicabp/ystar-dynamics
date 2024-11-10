# %%

from utils import *
from clean.rsi import *
from clean.finalize_experiments import *


def bootstrap_rsi(
    data_folder,
    bootstrap_iter=500,
    start_year=1995,
    start_month=1,
    end_year=2024,
    end_month=1,
    n_jobs=16,
):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_date = start_year * 4 + start_month
    end_date = end_year * 4 + end_month

    df = load_data(data_folder)
    df.drop(df[df.years_held < 2].index, inplace=True)

    # df = df[(df.quarter == 1) & (df.L_quarter == 1)]

    for b in range(100,bootstrap_iter):

        if rank == 0:
            print(f'Bootstrap iter {b}\n{"="*10}\n\n')

        boot_sample = resample(df, random_state=b)

        extensions, controls = get_extensions_controls(boot_sample)
        split = split_df_by_area(extensions, size)

        local_extensions = split[rank]
        local_controls = controls.drop(
            controls[~controls.area.isin(local_extensions.area.unique())].index
        )

        # Get DF for this process
        print(
            f"[{rank}/{size}]:\n\nNum Ext: {len(local_extensions)}\nNum Ctrl: {len(local_controls)}\n Local DF areas: {sorted(local_extensions.area.unique())}\n"
        )

        rsi = get_rsi(
            local_extensions,
            local_controls,
            start_date=start_date,
            end_date=end_date,
            case_shiller=False,
            n_jobs=n_jobs,
            rank=rank,
        )
        all_results = comm.gather(rsi, root=0)

        if rank == 0:
            file = os.path.join(data_folder, "working", "rsi", f"rsi_{b}.p")
            combined_results = pd.concat(all_results)
            combined_results.to_pickle(file)
