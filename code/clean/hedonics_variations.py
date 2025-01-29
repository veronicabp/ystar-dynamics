from utils import *
from clean.rsi import *


def get_rsi_hedonic_variations(
    data_folder,
    start_year=1995,
    start_quarter=1,
    end_year=2024,
    end_quarter=1,
    n_jobs=16,
):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # print(f"Running on node [{rank}/{size}].")

    start_date = start_year * 4 + start_quarter
    end_date = end_year * 4 + end_quarter

    df = load_data(data_folder)
    hedonics = pd.read_pickle(
        os.path.join(data_folder, "working", "merged_hmlr_hedonics.p")
    )

    hedonics_cols = [
        col
        for col in hedonics.columns
        if col.startswith("pres") or col.startswith("tpres")
    ]

    # hedonics_cols = hedonics_cols[-30:]

    if rank == 0:
        print("Cols to process:", hedonics_cols)

    hedonics.drop(
        columns=[
            col
            for col in hedonics.columns
            if col not in hedonics_cols + ["property_id", "date_trans"]
        ],
        inplace=True,
    )
    hedonics = hedonics.merge(
        df[["property_id"]].drop_duplicates(),
        on=["property_id"],
        how="inner",
    )

    for col in hedonics_cols:

        output_file = os.path.join(
            data_folder, "working", "hedonics_variations", f"rsi_{col}.p"
        )

        # If we've already run this one, continue
        if os.path.exists(output_file):
            continue

        if rank == 0:
            print(f"\n\n{col}:\n" + "=" * 20)

        this_hedonic = (
            hedonics[["property_id", "date_trans", col]].dropna(subset=[col]).copy()
        )

        # Merge on first date
        df_hedonic = df.merge(
            this_hedonic, on=["property_id", "date_trans"], how="inner"
        )

        # Merge on second date
        df_hedonic = df_hedonic.merge(
            this_hedonic.rename(
                columns={"date_trans": "L_date_trans", col: f"L_{col}"}
            ),
            on=["property_id", "L_date_trans"],
            how="inner",
        )

        df_hedonic[f"d_pres"] = df_hedonic[col] - df_hedonic[f"L_{col}"]

        # Get RSI for this hedonic
        local_extensions, local_controls = get_local_extensions_controls(
            df_hedonic, rank, size
        )
        # print(
        #     f"[{rank}/{size}]:\n\nNum Ext: {len(local_extensions)}\nNum Ctrl: {len(local_controls)}\n Local DF areas: {sorted(local_extensions.area.unique())}\n"
        # )

        rsi = get_rsi(
            local_extensions,
            local_controls,
            start_date=start_date,
            end_date=end_date,
            case_shiller=False,
            price_var="d_pres",
            n_jobs=n_jobs
        )
        rsi_gather = comm.gather(rsi, root=0)

        if rank == 0:
            combined_rsi = pd.concat(rsi_gather)
            combined_rsi.to_pickle(output_file)
            print(f"Saved {output_file}")
