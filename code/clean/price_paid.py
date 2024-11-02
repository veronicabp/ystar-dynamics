from utils import *


def load_data(data_folder):
    price_paid_file = os.path.join(data_folder, "raw", "hmlr", "pp-complete.csv")
    return pd.read_csv(
        price_paid_file,
        index_col=False,
        names=[
            "unique_id",
            "price",
            "date_trans",
            "postcode",
            "type",
            "new",
            "tenure",
            "street_number",
            "flat_number",
            "street",
            "locality",
            "city",
            "district",
            "county",
            "ppd_category",
            "record_status",
        ],
    )


def select_sample(price_paid, main_sample=False):

    # If flagged, restrict to leasehold flats
    if main_sample:
        price_paid = price_paid[price_paid["type"] == "F"]
        price_paid = price_paid[price_paid["tenure"] == "L"]

    # Drop missing or unknown
    price_paid = price_paid[
        ~((price_paid.street_number.isna()) & (price_paid.flat_number.isna()))
    ]
    price_paid = price_paid[~price_paid.postcode.isna()]
    price_paid = price_paid[price_paid.tenure != "U"]
    price_paid = price_paid[price_paid["type"] != "O"]

    # Convert date
    price_paid["date_trans"] = pd.to_datetime(price_paid["date_trans"])

    # Create property id
    price_paid["property_id"] = (
        price_paid[["flat_number", "street_number", "postcode"]]
        .fillna("")
        .agg(" ".join, axis=1)
    )
    price_paid["property_id"] = price_paid["property_id"].str.upper()
    price_paid["property_id"] = (
        price_paid["property_id"].str.replace(r"\s+", " ", regex=True).str.strip()
    )
    price_paid["property_id"] = price_paid["property_id"].apply(remove_punt)
    price_paid["property_id"] = price_paid["property_id"].str.strip()

    return price_paid


def drop_duplicates(price_paid):
    # Drop duplicates
    price_paid.drop_duplicates(
        subset=[col for col in price_paid.columns if col != "unique_id"], inplace=True
    )
    price_paid.sort_values(
        by=["property_id", "date_trans", "locality", "street", "city"], inplace=True
    )

    # If there are two transactions on the same date for PPD category A and B, keep only the one for category A
    price_paid["dup"] = price_paid.duplicated(
        subset=["property_id", "date_trans"], keep=False
    )
    price_paid["dup_cat"] = price_paid.duplicated(
        subset=["property_id", "date_trans", "ppd_category"], keep=False
    )
    price_paid = price_paid[
        ~((price_paid.dup != price_paid.dup_cat) & (price_paid.ppd_category == "B"))
    ].copy()

    # If there are multiple types per property/transaction, drop
    price_paid["dup"] = price_paid.duplicated(
        subset=["property_id", "date_trans"], keep=False
    )
    price_paid["dup_type"] = price_paid.duplicated(
        subset=["property_id", "date_trans", "type"], keep=False
    )
    price_paid = price_paid[price_paid.dup == price_paid.dup_type].copy()

    # If there are multiple tenures per property/transaction, drop
    price_paid["dup"] = price_paid.duplicated(
        subset=["property_id", "date_trans"], keep=False
    )
    price_paid["dup_tenure"] = price_paid.duplicated(
        subset=["property_id", "date_trans", "tenure"], keep=False
    )
    price_paid = price_paid[price_paid.dup == price_paid.dup_tenure].copy()

    # For the duplicates that remain, just take mean price across them and flag
    price_paid["multiple_prices"] = price_paid.duplicated(
        subset=["property_id", "date_trans"], keep=False
    )
    price_paid["price"] = price_paid.groupby(["property_id", "date_trans"])[
        "price"
    ].transform("mean")

    price_paid.drop_duplicates(subset=["property_id", "date_trans"], inplace=True)

    # Flag properties that are listed as multiple types/tenures
    for cat in ["type", "tenure"]:
        price_paid["dup"] = price_paid.duplicated(subset=["property_id"], keep=False)
        price_paid[f"dup_{cat}"] = price_paid.duplicated(
            subset=["property_id", cat], keep=False
        )
        price_paid[f"multiple_{cat}s"] = price_paid.dup != price_paid[f"dup_{cat}"]

    price_paid.drop(
        columns=[col for col in price_paid.columns if "dup" in col], inplace=True
    )

    return price_paid


def create_merge_keys(price_paid):
    # Create merge keys
    price_paid = create_string_id(
        price_paid,
        key_name="merge_key_1",
        columns=["flat_number", "street_number", "street", "city", "postcode"],
    )
    price_paid = create_string_id(
        price_paid,
        key_name="merge_key_2",
        columns=[
            "flat_number",
            "street_number",
            "street",
            "locality",
            "city",
            "postcode",
        ],
    )

    return price_paid


def clean_price_paid(data_folder):
    print("Cleaning price data.")
    price_paid = load_data(data_folder)
    price_paid = select_sample(price_paid)
    price_paid = drop_duplicates(price_paid)
    price_paid = create_merge_keys(price_paid)

    # Save leaseholds and freeholds
    price_paid[price_paid.tenure == "F"].to_pickle(
        os.path.join(data_folder, "working", "price_data_freeholds.p")
    )
    price_paid[price_paid.tenure == "L"].to_pickle(
        os.path.join(data_folder, "working", "price_data_leaseholds.p")
    )


def update_price_paid(data_folder):
    print("Updating price data.")
    price_paid = load_data(data_folder)
    price_paid = select_sample(price_paid, main_sample=True)
    price_paid = drop_duplicates(price_paid)
    price_paid = create_merge_keys(price_paid)

    # Save full data
    price_paid.to_pickle(os.path.join(data_folder, "working", "price_data.p"))
