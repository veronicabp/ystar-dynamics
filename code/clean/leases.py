from utils import *

text_to_num = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "fourty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

spell_months = SpellChecker(language=None)
spell_months.word_frequency.load_words(MONTHS)


def extract_number(text):
    """
    identifies whether there is a number in a string
    the number may be in digits or it may be spelled out (e.g. twenty-two, 22)

    text : string
            string from which to extract number
    """

    if not set(text.split()) & set(text_to_num):
        return text

    # Split the text into words
    words = text.split()
    # Initialize an empty list to hold the processed words
    processed_words = []
    # Initialize an empty list to hold the current spelled-out number
    spelled_number = []
    # Iterate over the words
    for word in words:
        # If the word is a digit, append it to processed_words and continue
        if word.isdigit():
            processed_words.append(word)
            continue
        # Try to convert the word to a number
        try:
            num = w2n.word_to_num(word)
            # If the conversion is successful, append the word to spelled_number
            spelled_number.append(word)
        except ValueError:
            # If the conversion fails and spelled_number is not empty, convert spelled_number to a number and append it to processed_words
            if spelled_number:
                spelled_number_str = " ".join(spelled_number)
                try:
                    num = w2n.word_to_num(spelled_number_str)
                    processed_words.append(str(num))
                except ValueError as ve:
                    pass
                    # print(f"Cannot convert {spelled_number_str} to a number. {ve}")
                spelled_number = []
            # Append the word to processed_words
            processed_words.append(word)
    # If spelled_number is not empty after iterating over all words, convert it to a number and append it to processed_words
    if spelled_number:
        spelled_number_str = " ".join(spelled_number)
        try:
            num = w2n.word_to_num(spelled_number_str)
            processed_words.append(str(num))
        except ValueError as ve:
            pass
            # print(f"Cannot convert {spelled_number_str} to a number. {ve}")
    # Join processed_words into a string and return it
    return " ".join(processed_words)


def correct_months(text):
    """
    corrects typos in month names

    text : string
            string to correct
    """
    search = re.search(" .{1,12}[1-2][0-9][0-9][0-9]", text)
    if search:
        match = search.group()
        words = match.split()[:-1]
        for word in words:
            if word in MONTHS or word.isnumeric() or len(word) <= 2:
                continue
            corrected = spell_months.correction(word)
            if corrected:
                text = text.replace(word, corrected)
    return text


def correct_typos(text):
    """
    corrects typos in a string using the TextBlob function

    text : string
            string to clean
    """
    # If number is attached to year, separate it
    search = re.search("[0-9]year", text)
    if search:
        match = search.group()
        num = re.search("[0-9]", match).group()
        text = text.replace(f"{num}year", f"{num} year")

    return str(TextBlob(text).correct())


def remove_cardinal_numbers(text):
    """
    convert cardinal numbers to regular numbers

    text : string
            string to clean
    """
    for regex in ["1st", "2nd", "3rd", "[1-9]?[0-9]th"]:
        search = re.search(regex, text)
        if search:
            match = search.group()
            num = int(re.search("[1-9]?[0-9]", match).group())
            text = text.replace(match, str(num))
    return text


def correct_holidays(text):
    """
    convert holidays into months

    text : string
            string to clean
    """
    holiday_months = {
        "christmas": "december",
        "midsummer": "june",
        "lady day": "march",
        "ladyday": "march",
        "michaelmas": "september",
        "michaelson": "september",
        "day": "",
    }
    for holiday in holiday_months:
        text = text.replace(holiday, holiday_months[holiday])
    return text


def get_date(
    text, key_words=["from", "commencing", "starting", "beginning"], end_keys=["year"]
):
    """
    extract a date from a string

    text : string
            string to from which to extract
    key_words : list (string)
            key words that precede this type of date
    end_keys : list (string)
            key words that follow this type of date
    """

    regex = "("
    for word in key_words:
        regex += word + "|"
    regex = regex[:-1] + ")"

    search = re.search(f"{regex}.*", text)
    if search:
        match = search.group() + " "

        # Go until the first year in this string
        years = re.findall("[1-4][0-9][0-9][0-9]", match)
        if len(years) > 0:
            year = years[0]
            submatch = match.split(year)[0] + year

            # If we think we have other information here aside from the start date, don't risk it
            parse_date = True
            for key in end_keys:
                if key in submatch:
                    parse_date = False
                    break

            if parse_date:
                # Get date
                try:
                    date = parse(submatch, fuzzy=True)
                    return date.strftime("%d-%m-%Y"), submatch
                except:
                    # If could not get date, use year
                    # print('Could not extract year from "', submatch, '" so instead we are using "', year, '" as the date')
                    return f"01-01-{year}", submatch
    return None, ""


def get_date_from(text):
    """
    extract the lease origination date

    text : string
            string to clean
    """
    return get_date(
        text,
        key_words=["from", "commencing", "starting", "beginning", "form"],
        end_keys=[" to ", "until", "expiring", "ending", "terminating", "year"],
    )


def get_date_start(text):
    """
    extract the date at the beginning of the string

    text : string
            string to clean
    """
    return get_date(text, key_words=["^"])


def get_date_to(text):
    """
    extract the lease end date

    text : string
            string to clean
    """
    return get_date(
        text,
        key_words=[
            " to ",
            "until",
            "expiring",
            "ending",
            "terminating",
            "expire",
            "end",
            "terminate",
        ],
    )


def get_number_years(text, date_from=None, date_to=None):
    """
    extract the lease length

    text : string
            string to clean
    date_from : string
            lease start date
    date_to : string
            lease end date
    """
    number_years, _ = get_number_years_exact(text)
    if number_years:
        return number_years
    search = re.search("(term|lease|for).{0,10}[1-9][0-9]?[0-9]?[0-9]?", text)
    if search:
        match = search.group()
        number_years = int(re.search("[1-9][0-9]?[0-9]?[0-9]?", match).group())
        return number_years
    elif date_from and date_to:
        try:
            date_format = "%d-%m-%Y"
            t = datetime.strptime(date_to, date_format) - datetime.strptime(
                date_from, date_format
            )
        except:
            t = parse(date_to, fuzzy=True) - parse(date_from, fuzzy=True)
        return t.days / 365
    return None


def get_number_years_exact(text):
    """
    extract the lease length using a precise regex expression

    text : string
            string to clean
    """
    search = re.search("[1-9][0-9]?[0-9]?[0-9]? years", text)
    if search:
        match = search.group()
        number_years = int(re.search("[1-9][0-9]?[0-9]?[0-9]?", match).group())
        return number_years, match
    return None, ""


def add_years(text):
    """
    flag a year in a string if it is not already flagged

    text : string
            string to clean
    """
    search = re.search("^[1-9][0-9]?[0-9]?[0-9]?", text)
    if search:
        match = search.group()
        if f"{match} year" not in text:
            text = text.replace(match, f"{match} years")
    return text


def date_from_is_registration(text):
    """
    check if the lease origination date is the registration date

    text : string
            string to clean
    """
    search = re.search(
        "((date(d)?|years) of (the |this )?(registered )?lease|commencement date|date(d)? as mentioned therein|date(d)? thereof|date(d)? hereof)",
        text,
    )
    if search:
        return True
    return False


def extract_term(original_text, date_registered=None):
    """
    extract relevant information about a lease from the recorded text field

    original_text : string
            raw text recorded in the lease document
    date_registered : Date
            date that lease was registered
    """

    if type(original_text) != str:
        return [None, None, None]

    text = original_text.lower()
    text = (
        text.replace(",", "")
        .replace(".", "-")
        .replace("~", "")
        .replace(" year ", " years ")
    )
    text = " ".join(text.split())
    text = remove_cardinal_numbers(text)
    text = correct_months(text)
    text = correct_typos(text)
    text = extract_number(text)
    text = re.sub(r"less \d+ (day|week)", "", text)
    text = correct_holidays(text)

    substring = text

    number_years, number_years_str = get_number_years_exact(substring)
    substring = substring.replace(number_years_str, "")

    # print("Number years:", number_years)

    # Check if date from is just the registration date
    date_from = None
    if date_from_is_registration(substring):
        date_from = date_registered

        # print("Looking for date from as date registered:", date_from)

    # If not, search for it in the text
    if date_from == None:
        date_from, date_from_str = get_date_from(substring)
        substring = substring.replace(date_from_str, "")

        # print("Looking for date from in text:", date_from)

    date_to, date_to_str = get_date_to(substring)
    substring = substring.replace(date_to_str, "")

    # If still could not find date from, check and see if it's at the beginning of the string
    if date_from == None:
        date_from, date_from_str = get_date_start(substring)
        substring = substring.replace(date_from_str, "")

        # print("Looking for date from at the beginning of the sentence:", date_from)

    if number_years == None:
        substring = add_years(substring)
        number_years = get_number_years(substring, date_from=date_from, date_to=date_to)

    if number_years == None and date_from == None and date_to == None:
        out = "\n\n\n---------------------------------------------------"
        out += "\nCould not parse the following text:"
        out += "\nText: " + original_text
        if text != original_text.lower():
            out += "\nCorrected text: " + text
        out += f"\nDate from: {date_from}"
        out += f"\nDate to: {date_to}"
        out += f"\nNumber years: {number_years}"
        # print(out)

    return [number_years, date_from, date_to]


def apply_extract_term(chunk):
    """
    wrapper for extract_term()

    chunk : numpy array
            chunk of data
    """
    chunk[["number_years", "date_from", "date_to"]] = chunk.progress_apply(
        lambda row: extract_term(row["term"], date_registered=row["date_registered"]),
        axis=1,
        result_type="expand",
    )
    return chunk


def process_row(row):
    return extract_term(row["term"], date_registered=row["date_registered"])


def get_merge_key(s):
    if type(s) != str:
        return s

    s = (
        s.upper()
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
        .replace("FLAT", "")
        .replace("APARTMENT", "")
    )
    return " ".join(s.split())


def parallelize(df, n_jobs=int(os.cpu_count()) - 2):
    start = time.time()
    df.reset_index(drop=True, inplace=True)
    rows = df.to_dict("records")

    results = pqdm(rows, process_row, n_jobs=n_jobs)
    new_cols = pd.DataFrame(results, columns=["number_years", "date_from", "date_to"])
    df[["number_years", "date_from", "date_to"]] = new_cols
    end = time.time()
    print(f"Time elapsed: {round((end-start)/60,2)} minutes.")

    return df


def extract_postcode(merge_key):
    pattern = r"[A-Z][A-Z]?[0-9][0-9]?[A-Z]? [0-9][A-Z][A-Z]"
    match = re.search(pattern, merge_key)
    if match:
        return match.group(0)
    return None


def drop_duplicates(df):
    # Deal with duplicates
    df.drop_duplicates(subset=["merge_key", "number_years", "date_from"], inplace=True)

    # If duplicates refer to a very long lease, only keep one of them
    df["duration2023"] = df["number_years"] - years_between_dates(
        pd.Period("2023-01-01", freq="D") - df["date_from"]
    )
    df["min_duration2023"] = df.groupby("merge_key")["duration2023"].transform("min")
    df = df[
        ~(
            (df["min_duration2023"] > 300)
            & (df["duration2023"] != df["min_duration2023"])
        )
    ].copy()

    # If duplicates refer to a very similar lease, keep one. If not, drop them.
    df["mean_duration2023"] = df.groupby("merge_key")["duration2023"].transform("mean")
    df["sd_duration2023"] = df.groupby("merge_key")["duration2023"].transform("std")
    df = df[~((df["sd_duration2023"] > 10) & (~df["sd_duration2023"].isna()))].copy()

    # Remove duplicate rows by 'merge_key', keeping the first occurrence
    df = df.drop_duplicates(subset=["merge_key"], keep="first")

    return df


def convert_dates(df, format="DD-MM-YYYY", tags=["from", "to", "registered"]):
    for tag in tags:
        # Convert to dates

        if format == "DD-MM-YYYY":
            df[f"year_{tag}"] = pd.to_numeric(
                df[f"date_{tag}"].str.slice(6, 10), errors="coerce"
            )
            df[f"month_{tag}"] = pd.to_numeric(
                df[f"date_{tag}"].str.slice(3, 5), errors="coerce"
            )
            df[f"day_{tag}"] = pd.to_numeric(
                df[f"date_{tag}"].str.slice(0, 2), errors="coerce"
            )
        elif format == "MM-DD-YYYY":
            df[f"year_{tag}"] = pd.to_numeric(
                df[f"date_{tag}"].str.slice(6, 10), errors="coerce"
            )
            df[f"month_{tag}"] = pd.to_numeric(
                df[f"date_{tag}"].str.slice(0, 2), errors="coerce"
            )
            df[f"day_{tag}"] = pd.to_numeric(
                df[f"date_{tag}"].str.slice(3, 5), errors="coerce"
            )
        else:
            raise ValueError("Date format is incorrect.")

        df.loc[df[f"month_{tag}"] > 12, f"month_{tag}"] = 1
        df.loc[df[f"day_{tag}"] > 31, f"day_{tag}"] = 1

        df["period"] = (
            df[f"year_{tag}"].fillna(0).astype(int).astype(str)
            + "-"
            + df[f"month_{tag}"].fillna(0).astype(int).astype(str).str.zfill(2)
            + "-"
            + df[f"day_{tag}"].fillna(0).astype(int).astype(str).str.zfill(2)
        )
        df.loc[
            (df[f"year_{tag}"].isna())
            | (df[f"month_{tag}"].isna())
            | (df[f"day_{tag}"].isna()),
            "period",
        ] = np.nan

        df[f"date_{tag}"] = pd.PeriodIndex(df["period"], freq="D")
        df.drop(columns=["period"], inplace=True)
    return df


def extract_terms(df, n_jobs=int(os.cpu_count()) - 2):
    # Create merge key
    df["merge_key"] = df.progress_apply(
        lambda row: get_merge_key(row["associated property description"]), axis=1
    )

    # Drop if missing date or address data
    df = df[~df.merge_key.isna()]
    df = df[~df["term"].isna()]
    df = df[["merge_key", "term", "date of lease", "id", "uprn", "unique_id"]]

    df["postcode"] = df["merge_key"].apply(extract_postcode)
    df = df[~df.postcode.isna()]

    df.loc[~df["date of lease"].isna(), ["year_registered"]] = (
        df["date of lease"]
        .loc[~df["date of lease"].isna()]
        .astype(str)
        .str[-4:]
        .astype(int)
    )
    df.loc[~df["date of lease"].isna(), ["month_registered"]] = (
        df["date of lease"]
        .loc[~df["date of lease"].isna()]
        .astype(str)
        .str[3:5]
        .astype(int)
    )
    df = df.rename(columns={"date of lease": "date_registered"})

    df = parallelize(df)
    df = convert_dates(df, format="DD-MM-YYYY")

    # If missing lease start date, use end date to infer
    df.loc[
        (df.date_from.isna()) & (~df.date_to.isna()) & (~df.number_years.isna()),
        "date_from",
    ] = (
        df["date_to"] - df["number_years"].fillna(0).astype(int) * 365
    )

    df.loc[
        (df.date_from.isna()) & (df.number_years.isna()) & (~df.date_to.isna()),
        "number_years",
    ] = years_between_dates(df["date_to"] - df["date_registered"])
    df.loc[(df.date_from.isna()) & (~df.date_to.isna()), "date_from"] = (
        df.date_registered
    )

    df = df[(~df.date_from.isna()) & (~df.number_years.isna()) & (df.number_years >= 0)]

    df = drop_duplicates(df)
    df.drop(
        columns=["mean_duration2023", "min_duration2023", "sd_duration2023"],
        inplace=True,
    )

    return df


def clean_purchased_leases(data_folder):
    # Closed titles
    dfs = []
    folder = os.path.join(data_folder, "raw", "hmlr", "purchased_titles")
    files = [file for file in os.listdir(folder) if file.endswith("xlsx")]
    for file in files:
        print(file)
        file = os.path.join(folder, file)
        this_df = pd.read_excel(file)
        dfs.append(this_df)
    closed_leases = pd.concat(dfs, ignore_index=True)

    closed_leases = closed_leases.rename(
        columns={
            "TITLE_CLOS_DATE": "date_registered",
            "TERM": "term",
            "CLOS_YEAR": "year_registered",
            "TRANSACTION_ID": "unique_id",
            "DEED_DATE": "date_deed",
        }
    )
    closed_leases.columns = closed_leases.columns.str.lower()

    closed_leases.drop_duplicates(subset="unique_id", inplace=True)

    closed_leases["term"] = closed_leases["term"].str.replace("_x000D_", " ")
    closed_leases["term"] = closed_leases["term"].replace(r"\s+|\\n", " ", regex=True)
    closed_leases = parallelize(closed_leases, n_jobs=n_jobs)
    closed_leases["closed_lease"] = True
    closed_leases["unique_id"] = "{" + closed_leases["unique_id"] + "}"

    # Other purchased titles
    file = os.path.join(
        data_folder,
        "raw",
        "hmlr",
        "purchased_titles",
        "Princeton Lease Information for Issue.csv",
    )
    other_leases = pd.read_csv(file, encoding="latin1")
    other_leases = other_leases.rename(
        columns={
            "Date Registered": "date_registered",
            "Registered Lease Details": "term",
            "Client Reference": "unique_id",
        }
    )
    other_leases.columns = other_leases.columns.str.lower()

    other_leases.drop_duplicates(subset="unique_id", inplace=True)

    other_leases["date_registered"] = pd.to_datetime(
        other_leases["date_registered"], format="mixed"
    )
    other_leases["date_registered"] = other_leases["date_registered"].dt.strftime(
        "%d/%m/%Y"
    )
    other_leases = parallelize(other_leases, n_jobs=n_jobs)
    other_leases["closed_lease"] = False

    # Combine
    purchased_leases = pd.concat([closed_leases, other_leases]).reset_index(drop=True)
    purchased_leases = convert_dates(
        purchased_leases, tags=["registered", "deed", "from", "to"]
    )

    # If lease has an expiration date but not an origination date, use the deed date as the origination date
    purchased_leases["flag"] = (
        (purchased_leases.date_from.isna())
        & (~purchased_leases.date_to.isna())
        & (~purchased_leases.date_deed.isna())
    )
    purchased_leases.loc[
        (purchased_leases.flag) & (purchased_leases.number_years.isna()), "number_years"
    ] = years_between_dates(purchased_leases["date_to"] - purchased_leases["date_deed"])
    purchased_leases.loc[
        (purchased_leases.flag)
        | (
            (purchased_leases.date_from.isna())
            & (~purchased_leases.number_years.isna())
        ),
        "date_from",
    ] = purchased_leases.date_deed

    purchased_leases = purchased_leases[
        ~(
            (purchased_leases.number_years.isna())
            | (purchased_leases.number_years < 0)
            | (purchased_leases.date_from.isna())
        )
    ].copy()
    purchased_leases.drop_duplicates(subset="unique_id", inplace=True)
    purchased_leases = keep_columns(purchased_leases, additional_cols=["closed_lease"])

    return purchased_leases


def identify_extensions(base_df, new_df):
    base_df.rename(
        columns={x: f"{x}_prev" for x in base_df.columns if x != "id"}, inplace=True
    )
    df = new_df.merge(base_df, on="id", indicator=True, how="outer")

    df["extension"] = (df._merge == "both") & (
        df.duration2023 - df.duration2023_prev > 30
    )
    df["extension_amount"] = np.nan
    df.loc[df.extension, "extension_amount"] = df.duration2023 - df.duration2023_prev

    for var in ["number_years", "date_from", "date_registered"]:
        df[f"{var} (pre_extension)"] = df[f"{var}_prev"]
        df.loc[~df.extension, f"{var} (pre_extension)"] = np.nan

    # Drop cases of subleases (significant decrease in lease duration) because we don't have a good way of matching these to transactions
    df = df[~(df.duration2023 - df.duration2023_prev < -10)].copy()

    for col in new_df.columns:
        if col != "id" and f"{col}_prev" in df.columns:
            df.loc[df._merge == "right_only", col] = df[f"{col}_prev"]
            df.drop(columns=f"{col}_prev", inplace=True)

    # Drop duplicates -- cases where ID is different but merge key is the same
    df = drop_duplicates(df)
    return df


def rename_columns(df):
    df.columns = df.columns.str.lower()
    df.rename(
        columns={
            "os uprn": "uprn",
            "associated property description id": "id",
            "unique identifier": "unique_id",
        },
        inplace=True,
    )
    return df


def keep_columns(df, additional_cols=[]):
    cols_to_keep = (
        [
            "merge_key",
            "term",
            "id",
            "uprn",
            "unique_id",
            "postcode",
            "extension",
            "extension_amount",
            "duration2023",
        ]
        + [col for col in df.columns if "date" in col or "number_years" in col]
        + additional_cols
    )
    cols_to_keep = list(set(cols_to_keep) & set(df.columns))
    return df[cols_to_keep]


def get_new_leases(df_old, df_new):
    new_leases = df_new.merge(
        df_old[["id", "term"]].drop_duplicates(),
        on=["id", "term"],
        how="left",
        indicator=True,
    )
    new_leases.drop(new_leases[new_leases._merge != "left_only"].index, inplace=True)
    new_leases.drop(columns="_merge", inplace=True)
    return new_leases


def clean_leases(data_folder):

    ####### Clean closed/purchased titles
    purchased_leases = clean_purchased_leases(data_folder)
    purchased_leases.to_pickle(
        os.path.join(data_folder, "working", "purchased_lease_data.p")
    )

    ####### Clean open titles
    print("Cleaning open titles.")
    # Separate into pre and post May 2023 -- this is when we stop being able to track extensions via purchased titles
    leases_may2023 = rename_columns(
        pd.read_csv(os.path.join(data_folder, "raw", "hmlr", "LEASES_FULL_2023_05.csv"))
    )
    new_leases = rename_columns(
        pd.read_csv(os.path.join(data_folder, "raw", "hmlr", "LEASES_FULL_2024_02.csv"))
    )
    new_leases = get_new_leases(leases_may2023, new_leases)

    # Extract information
    leases_may2023 = extract_terms(leases_may2023)
    new_leases = extract_terms(new_leases)
    all_leases = identify_extensions(leases_may2023, new_leases)

    all_leases = keep_columns(all_leases)
    all_leases.to_pickle(os.path.join(data_folder, "working", "lease_data.p"))


def update_leases(data_folder, prev_data_folder):
    # Select new leases
    print("Loading lease data.")

    data_folder_path = os.path.join(data_folder, "raw", "hmlr")
    prev_data_folder_path = os.path.join(prev_data_folder, "raw", "hmlr")

    new_leases_file = [
        file for file in sorted(os.listdir(data_folder_path)) if "LEASES_FULL_" in file
    ][-1]
    old_leases_file = [
        file
        for file in sorted(os.listdir(prev_data_folder_path))
        if "LEASES_FULL_" in file
    ][-1]

    new_leases = rename_columns(
        pd.read_csv(os.path.join(data_folder_path, new_leases_file))
    )
    old_leases = rename_columns(
        pd.read_csv(os.path.join(prev_data_folder_path, old_leases_file))
    )
    new_leases = get_new_leases(old_leases, new_leases)

    # Extract terms + clean
    print("Extracting terms.")
    processed_leases = extract_terms(new_leases)

    # Identify extensions
    print("Identifying extensions relative to previous file.")
    old_leases = pd.read_pickle(
        os.path.join(prev_data_folder, "working", "lease_data.p")
    )
    all_leases = identify_extensions(old_leases, processed_leases)

    # Save
    all_leases = keep_columns(all_leases)
    all_leases.to_pickle(os.path.join(data_folder, "working", "lease_data.p"))


if __name__ == "__main__":
    print("In program.")
