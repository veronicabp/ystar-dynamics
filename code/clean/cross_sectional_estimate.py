from utils import *


def convert_lad08_lad21(data_folder):
    raw_folder = os.path.join(data_folder, "raw")
    ons_folder = os.path.join(raw_folder, "ons")

    lad08_file = os.path.join(
        ons_folder, "LAD_DEC_2008_GB_BFC", "LAD_DEC_2008_GB_BFC.shp"
    )
    lad21_file = os.path.join(
        ons_folder, "LAD_DEC_2021_UK_BFC", "LAD_DEC_2021_UK_BFC.shp"
    )
    hilber_file = os.path.join(
        raw_folder, "original", "ecoj12213-sup-0001-DataS1", "dta files", "data LPA.dta"
    )

    lad08_gdf = gpd.read_file(lad08_file)
    lad21_gdf = gpd.read_file(lad21_file)
    hilber = pd.read_stata(hilber_file)

    fields = [
        "pdevel90_m2",
        "refusal_maj_7908",
        "delchange_maj1",
        "delchange_maj5",
        "delchange_maj6",
        "rindex2",
        "male_earn_real",
    ]
    hilber = hilber[["lpa_code", "lpa_name", "year"] + fields]

    # Calculate total area of each codes system
    lad08_gdf["lad08_area"] = lad08_gdf.area
    lad21_gdf["lad21_area"] = lad21_gdf.area

    # Calculate overlap
    # Spatial join - this associates each LPA with an LAD
    # print("Calculating intersection:")
    join_gdf = gpd.sjoin(lad21_gdf, lad08_gdf, how="inner", predicate="intersects")

    # Calculate intersection area
    join_gdf["intersection_area"] = join_gdf.apply(
        lambda row: row["geometry"]
        .intersection(
            lad08_gdf.loc[lad08_gdf["LAD08CD"] == row["LAD08CD"], "geometry"].values[0]
        )
        .area,
        axis=1,
    )

    # Merge in Hilber data to 2008 data
    join_gdf = join_gdf.merge(
        hilber, left_on="LAD08CD", right_on="lpa_code", how="left"
    )
    # print("Joined GDF:\n", join_gdf)

    # Aggregate by 2021 codes, taking weighted mean
    for field in fields:
        join_gdf[field] = (
            join_gdf[field] * join_gdf["intersection_area"] / join_gdf["lad21_area"]
        )

    # Collapse
    agg = {field: "sum" for field in fields}
    agg["lad21_area"] = "first"
    df = join_gdf.groupby(["LAD21NM", "LAD21CD", "year"]).agg(agg).reset_index()
    df = df[["LAD21NM", "LAD21CD", "year"] + fields]
    df = df[df.LAD21CD.str.startswith("E")]

    # Drop Isles of Scilly (missing data)
    df = df[df.LAD21CD != "E06000053"]

    df = df.rename(columns={"LAD21CD": "lpa_code", "LAD21NM": "lpa_name"})
    return df


def get_inflation_deflator(data_folder):
    raw_folder = os.path.join(data_folder, "raw")
    working_folder = os.path.join(data_folder, "working")
    fred_path = os.path.join(raw_folder, "fred", "CPGRLE01GBM659N.csv")
    cpi_fred = pd.read_csv(fred_path)

    # Use OECD CPI in early years where missing retail price index
    cpi_fred["DATE"] = pd.to_datetime(
        cpi_fred["DATE"], format="%Y-%m-%d", errors="coerce"
    )
    cpi_fred["year"] = cpi_fred["DATE"].dt.year
    cpi_yearly = cpi_fred.groupby("year", as_index=False).agg(
        {"CPGRLE01GBM659N": "mean"}
    )
    cpi_yearly.rename(columns={"CPGRLE01GBM659N": "pct_change"}, inplace=True)

    # Use retail price index when it is available, per Hilber & Vermeulen
    ons_path = os.path.join(raw_folder, "ons", "inflation", "series-100224.csv")
    ons_df = pd.read_csv(ons_path)
    ons_df = ons_df.iloc[8:55].copy()
    ons_df.rename(
        columns={
            "Title": "year",
            "RPI All Items Excl Mortgage Interest (RPIX): Percentage change over 12 months": "pct_change",
        },
        inplace=True,
    )
    ons_df["year"] = pd.to_numeric(ons_df["year"], errors="coerce")
    ons_df["pct_change"] = pd.to_numeric(ons_df["pct_change"], errors="coerce")

    # Merge
    merged = pd.merge(
        ons_df, cpi_yearly, on="year", how="outer", suffixes=("", "_fred")
    )
    merged["pct_change"] = merged["pct_change"].fillna(merged["pct_change_fred"])

    # Create RPI series
    merged.sort_values(by="year", inplace=True)
    merged["rpi"] = np.nan
    merged.loc[merged["year"] == 2008, "rpi"] = 1

    # Identify the index of year=2008 (if it exists)
    idx_2008 = merged.index[merged["year"] == 2008].item()
    # Forward fill for years after 2008
    for i in range(idx_2008 + 1, len(merged)):
        merged.loc[i, "rpi"] = merged.loc[i - 1, "rpi"] * (
            1 + merged.loc[i, "pct_change"] / 100.0
        )
    # Backward fill for years before 2008
    for i in range(idx_2008 - 1, -1, -1):
        merged.loc[i, "rpi"] = merged.loc[i + 1, "rpi"] / (
            1 + merged.loc[i + 1, "pct_change"] / 100.0
        )

    rpi_df = merged[["year", "rpi"]].copy()
    return rpi_df


def get_house_prices(data_folder):
    raw_folder = os.path.join(data_folder, "raw")
    working_folder = os.path.join(data_folder, "working")

    merged_hmlr_path = os.path.join(working_folder, "merged_hmlr_all.p")
    merged_hmlr_df = pd.read_pickle(merged_hmlr_path)

    lpa_codes_path = os.path.join(raw_folder, "geography", "lpa_codes.dta")
    lpa_codes_df = pd.read_stata(lpa_codes_path)

    merged_df = pd.merge(merged_hmlr_df, lpa_codes_df, on="postcode", how="inner")
    merged_df = merged_df[merged_df["lpa_code"].str.startswith("E")]

    # Weighting to remove composition differences
    merged_df["N"] = merged_df.groupby(["lpa_code"])["log_price"].transform("count")
    merged_df["N_i"] = merged_df.groupby(["lpa_code", "type"])["log_price"].transform(
        "count"
    )

    merged_df["N_t"] = merged_df.groupby(["lpa_code", "year"])["log_price"].transform(
        "count"
    )
    merged_df["N_it"] = merged_df.groupby(["lpa_code", "type", "year"])[
        "log_price"
    ].transform("count")

    merged_df["sh_i"] = merged_df["N_i"] / merged_df["N"]
    merged_df["sh_it"] = merged_df["N_it"] / merged_df["N_t"]
    merged_df["w_it"] = merged_df["sh_i"] / merged_df["sh_it"]

    merged_df = merged_df[["lpa_code", "year", "price", "w_it"]].copy()

    # Take weighted average
    def weighted_avg(group):
        w = group["w_it"]
        return (group["price"] * w).sum() / w.sum()

    collapsed = merged_df.groupby(["lpa_code", "year"], as_index=False).apply(
        weighted_avg
    )
    collapsed.rename(columns={None: "price"}, inplace=True)

    # Create price index
    collapsed = collapsed.sort_values(["lpa_code", "year"]).reset_index(drop=True)

    def make_hpi(g):
        # Divide by the first (earliest) row in that group
        first_price = g["price"].iloc[0]
        g["hpi"] = g["price"] / first_price
        return g

    result = collapsed.groupby("lpa_code", group_keys=False).apply(make_hpi)

    return result[["lpa_code", "year", "hpi", "price"]].copy()


def get_hpi(data_folder):
    hmlr_price_index = get_house_prices(data_folder)
    hilber_df = convert_lad08_lad21(data_folder)
    rpi_df = get_inflation_deflator(data_folder)

    merged_df = pd.merge(
        hilber_df, hmlr_price_index, on=["lpa_code", "year"], how="outer"
    )

    merged_df = pd.merge(merged_df, rpi_df, on="year", how="left")

    # Make Hilber & Vermeulen data nominal
    merged_df["rpi1974"] = np.where(merged_df["year"] == 1974, merged_df["rpi"], np.nan)
    merged_df["rpi1974"] = merged_df.groupby("lpa_code")["rpi1974"].transform(
        lambda x: x.fillna(x.mean())
    )
    merged_df["hilber_index"] = merged_df["rindex2"] * (
        merged_df["rpi"] / merged_df["rpi1974"]
    )

    merged_df["hilber_index1995"] = np.where(
        merged_df["year"] == 1995, merged_df["hilber_index"], np.nan
    )
    merged_df["hilber_index1995"] = merged_df.groupby("lpa_code")[
        "hilber_index1995"
    ].transform(lambda x: x.fillna(x.mean()))

    # If year<1995, use the hilber_index as hpi
    # If year>=1995, multiply existing hpi by hilber_index1995
    merged_df["hpi"] = np.where(
        merged_df["year"] < 1995,
        merged_df["hilber_index"],
        merged_df["hilber_index1995"] * merged_df["hpi"],
    )

    merged_df["log_hpi"] = np.log(merged_df["hpi"] / 100.0)
    return merged_df[["lpa_code", "year", "hpi", "log_hpi"]].copy()


def expand_hilber_data(data_folder):
    ashe_df = pd.read_pickle(os.path.join(data_folder, "working", "ashe_earnings.p"))
    hilber_df = convert_lad08_lad21(data_folder)

    hilber_cs_vars = hilber_df[
        ["lpa_code", "pdevel90_m2", "refusal_maj_7908", "delchange_maj5"]
    ].drop_duplicates()

    merged = pd.merge(
        hilber_df[["lpa_code", "year", "rindex2", "male_earn_real"]],
        ashe_df[ashe_df.year != 2022].drop(columns="Description"),
        on=["lpa_code", "year"],
        how="outer",
    )
    merged = merged.merge(hilber_cs_vars, on="lpa_code", how="inner")

    hpi_df = get_hpi(data_folder)
    rpi_df = get_inflation_deflator(data_folder)

    merged = merged.merge(hpi_df, on=["lpa_code", "year"], how="left")
    merged = merged.merge(rpi_df, on=["year"], how="left")

    merged["male_earn_nom"] = merged["male_earn_real"] * merged["rpi"]
    merged["earn"] = np.where(
        merged["earn"].isna(), merged["male_earn_nom"], merged["earn"]
    )

    # 5) gen log_earn = log(earn)
    merged["log_earn"] = np.log(merged["earn"])
    merged.drop(
        columns=["rindex2", "male_earn_real", "male_earn_nom", "num_jobs"], inplace=True
    )
    merged.to_pickle(os.path.join(data_folder, "working", "expanded_hilber_data.p"))

    return merged


def get_local_housing_betas(df):

    df.sort_values(by=["lpa_code", "year"], inplace=True)
    df["d_log_hpi"] = df.log_hpi - df.groupby("lpa_code").log_hpi.shift(1)

    df.dropna(subset="earn", inplace=True)

    dummies = pd.get_dummies(df["lpa_code"], prefix="lpa", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    dummy_cols = list(dummies.columns)
    for col in dummy_cols:
        df[f"{col}_x_log_earn"] = df[col] * df["log_earn"]
        df = df.copy()

    df["lpa_code"] = df["lpa_code"].astype("category")
    df["year"] = df["year"].astype("category")

    interaction_vars = [col for col in df.columns if col.endswith("_x_log_earn")]
    result = AbsorbingLS(
        df["log_hpi"], df[interaction_vars], absorb=df[["lpa_code", "year"]]
    ).fit()

    lpas = sorted(df.lpa_code.unique())
    betas = []
    d_hpi = []
    for lpa in lpas:
        betas.append(result.params[f"lpa_{lpa}_x_log_earn"])
        d_hpi.append(df[df.lpa_code == lpa].d_log_hpi.mean())

    return pd.DataFrame({"lpa_code": lpas, "beta": betas, "d_log_hpi": d_hpi})


def find_overlap_clusters(gdf):
    # Step 1: Initialize a list to store clusters
    clusters = []

    # Step 2: Create a spatial index for fast spatial queries
    sindex = gdf.sindex

    # Step 3: Loop through each geometry to find its overlaps
    for idx, geom in enumerate(gdf.geometry):
        # Find all geometries that intersect with the current geometry
        possible_matches_index = list(sindex.query(geom, predicate="intersects"))

        # Find if this geometry intersects with any existing cluster
        found_cluster = False
        for cluster in clusters:
            if any(i in cluster for i in possible_matches_index):
                # If it intersects, add all overlapping geometries to this cluster
                cluster.update(possible_matches_index)
                found_cluster = True
                break

        # If no existing cluster was found, create a new one
        if not found_cluster:
            clusters.append(set(possible_matches_index))

    # Step 4: Merge clusters that have overlaps (i.e., connected components)
    merged_clusters = []
    while clusters:
        cluster = clusters.pop(0)
        merged = True
        while merged:
            merged = False
            for other_cluster in clusters[:]:
                if cluster & other_cluster:  # Check if they share any element
                    cluster |= other_cluster  # Merge them
                    clusters.remove(other_cluster)
                    merged = True
        merged_clusters.append(cluster)

    return merged_clusters


def calculate_share(main_df, geography_df, field, geography_name="RMSect"):
    # Perform a spatial intersection between geography and gdf with data
    intersection = gpd.overlay(main_df, geography_df, how="intersection")

    # Calculate the area of the total sectors and the intersection
    geography_df["total_area"] = geography_df.geometry.area
    intersection[f"{field}_area"] = intersection.geometry.area

    # Merge the intersection data back with the sectors to calculate the flood risk share
    share = intersection.groupby(geography_name)[f"{field}_area"].sum().reset_index()
    geography_df = geography_df.merge(share, on=geography_name, how="left")

    # Calculate the share of each sector that is at flood risk
    geography_df[f"{field}_share"] = (
        geography_df[f"{field}_area"] / geography_df["total_area"]
    )
    geography_df.loc[geography_df[f"{field}_share"].isna(), f"{field}_share"] = 0

    return geography_df


def get_climate_exposure(data_folder):

    ######## Flood Risk
    sectors_file = os.path.join(
        data_folder, "raw", "geography", "GB_Postcodes", "PostalSector.shp"
    )
    sectors = gpd.read_file(sectors_file)
    sectors = sectors.to_crs(epsg=27700)

    lpas_file = os.path.join(
        data_folder, "raw", "ons", "LAD_DEC_2021_UK_BFC", "LAD_DEC_2021_UK_BFC.shp"
    )
    lpas = gpd.read_file(lpas_file)
    lpas = lpas.to_crs(epsg=27700)

    flood_risk_file = os.path.join(
        data_folder,
        "raw",
        "flood_risk",
        "FloodRiskAreas-SHP",
        "data",
        "Flood_Risk_Areas.shp",
    )
    flood_risk = gpd.read_file(flood_risk_file)
    flood_risk = flood_risk.to_crs(epsg=27700)
    clusters = find_overlap_clusters(flood_risk)

    dissolved_geometries = []
    for cluster in clusters:
        combined_geom = unary_union(flood_risk.loc[list(cluster)].geometry)
        dissolved_geometries.append(combined_geom)

    # Create a new GeoDataFrame with dissolved geometries
    flood_risk = gpd.GeoDataFrame(geometry=dissolved_geometries, crs=flood_risk.crs)
    flood_risk_lpas = calculate_share(
        flood_risk, lpas, "flood_risk", geography_name="LAD21CD"
    )
    flood_risk_lpas = (
        flood_risk_lpas[["LAD21CD", "flood_risk_share"]]
        .rename(columns={"LAD21CD": "lpa_code"})
        .copy()
    )

    ######## Subsidence risk
    year = 2030
    geoclimate_file = os.path.join(
        data_folder,
        "raw",
        "geoclimate",
        "GeoClimateUKCP18OpenData",
        "GeoclimateUKCP18_Open",
        f"GeoClimateUKCP18_ShrinkSwell_{year}_Average_Open.shp",
    )
    geoclimate = gpd.read_file(geoclimate_file)
    geoclimate = geoclimate.to_crs(epsg=27700)

    geoclimate = geoclimate[geoclimate.CLASS != "Unavailable"]
    probable = calculate_share(
        geoclimate[geoclimate.CLASS == "Probable"],
        lpas,
        f"probable_risk_{year}",
        geography_name="LAD21CD",
    )
    possible = calculate_share(
        geoclimate[geoclimate.CLASS.isin(["Probable", "Possible"])],
        lpas,
        f"possible_risk_{year}",
        geography_name="LAD21CD",
    )
    subsidence_risk = probable.merge(possible, on="LAD21CD", suffixes=("", "_dup"))
    subsidence_risk.drop(
        columns=[col for col in subsidence_risk.columns if col.endswith("_dup")],
        inplace=True,
    )

    subsidence_risk = (
        subsidence_risk[
            ["LAD21CD", f"probable_risk_{year}_share", f"possible_risk_{year}_share"]
        ]
        .rename(columns={"LAD21CD": "lpa_code"})
        .copy()
    )

    return flood_risk_lpas, subsidence_risk


def get_cross_sectional_estimates(data_folder, precutoff=2009, postcutoff=2022):

    hilber_df = pd.read_pickle(
        os.path.join(data_folder, "working", "expanded_hilber_data.p")
    )

    # Get local housing beta, measure of housing risk
    betas = get_local_housing_betas(hilber_df)
    flood_risk, subsidence_risk = get_climate_exposure(data_folder)

    # Estimate y-star at the local authority level
    df = pd.read_pickle(os.path.join(data_folder, "clean", "experiments.p"))
    df = df[df.lpa_code.str.startswith("E")].copy()
    df.dropna(subset="did_rsi_yearly", inplace=True)

    data = []
    lpas = sorted(df.lpa_code.unique())
    for lpa in tqdm(lpas):
        ystar_pre, se_pre = estimate_ystar(
            df[(df.lpa_code == lpa) & (df.year <= precutoff)].copy(),
            lhs_var="did_rsi_yearly",
        )

        ystar_post, se_post = estimate_ystar(
            df[(df.lpa_code == lpa) & (df.year >= postcutoff)].copy(),
            lhs_var="did_rsi_yearly",
        )

        ystar, se = estimate_ystar(
            df[(df.lpa_code == lpa)].copy(),
            lhs_var="did_rsi_yearly",
        )

        region = list(df[df.lpa_code == lpa].region)[0]
        data.append([lpa, ystar_pre, ystar_post, ystar, se_pre, se_post, se, region])

    ystar_estimates = pd.DataFrame(
        data,
        columns=[
            "lpa_code",
            "ystar_pre",
            "ystar_post",
            "ystar_all",
            "se_pre",
            "se_post",
            "se_all",
            "region",
        ],
    )

    ystar_estimates["d_ystar"] = ystar_estimates.ystar_post - ystar_estimates.ystar_pre
    ystar_estimates["w"] = 1 / (ystar_estimates.se_pre**2 + ystar_estimates.se_post**2)
    ystar_estimates["w_all"] = 1 / (ystar_estimates.se_all**2)

    lower = ystar_estimates["d_ystar"].quantile(0.05)
    upper = ystar_estimates["d_ystar"].quantile(0.95)
    ystar_estimates[f"d_ystar_win"] = ystar_estimates["d_ystar"].clip(lower, upper)

    hilber_invariants = hilber_df[
        ["lpa_code", "pdevel90_m2", "refusal_maj_7908", "delchange_maj5"]
    ].drop_duplicates()

    ystar_estimates = ystar_estimates.merge(
        hilber_invariants, on=["lpa_code"], how="left"
    )

    ystar_estimates = ystar_estimates.merge(betas, on=["lpa_code"], how="left")
    ystar_estimates = ystar_estimates.merge(flood_risk, on=["lpa_code"], how="left")
    ystar_estimates = ystar_estimates.merge(
        subsidence_risk, on=["lpa_code"], how="left"
    )

    ystar_estimates.to_pickle(
        os.path.join(data_folder, "clean", f"ystar_by_lpas_{precutoff}-{postcutoff}.p")
    )
