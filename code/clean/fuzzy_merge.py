from utils import *


def clean_number(s):
    """
    removes filler words from an address number

    s : string
            string to clean
    """
    return s.replace("THE", "").replace("FLAT", "").replace("APARTMENT", "").strip()


def has_numbers(s):
    """
    checks if a string contains a number

    s : string
            string to check
    """
    return any(char.isdigit() for char in s)


def get_common_words(s1, s2):
    """
    returns the number of words that two strings have in common

    s1 : string
            first string
    s2 : string
            second string
    """

    if type(s1) != str or type(s2) != str:
        return 0
    return len(list(set(s1.split(" ")) & set(s2.split(" "))))


def tokenize(s, n):
    """
    tokenizes an address and creates (1) a set of necessary strings and (2) a set of all non-trivial strings

    s : string
            address string to tokenize
    n : string
            necessary string which must be identified in match (e.g. postcode)
    """

    s = clean_number(str(s))
    n = str(n)
    l = s.split()
    necessary = []
    desired = []

    # There are certain key words that are necessary
    for term in [
        "ASSOCIATED WITH",
        "GARAGE",
        "PARKING",
        "STORAGE",
        "LAND",
        "BALCONY ADJOINING",
        "GROUND",
        "GARDEN FLOOR",
        "LOWER",
        "FIRST",
        "SECOND",
        "THIRD",
        "FOURTH",
        "FLOOR",
        "BASEMENT",
    ] + [
        f"LAND LYING TO THE {direction} OF"
        for direction in ["NORTH", "SOUTH", "EAST", "WEST"]
    ]:
        if (len(term.split()) == 1 and term in s.split()) or (
            len(term.split()) > 1 and term in s
        ):
            necessary.append(term)
    for word in l:
        if (
            word.isnumeric()
            or has_numbers(word)
            or word in list("abcdefghijklmnopqrstuvwxy".upper())
        ) and word != "0":
            necessary.append(word)
            desired.append(word)
        else:
            desired.append(word)
    if len(necessary) == 0:
        necessary = desired
    necessary.append(n)
    desired.append(n)
    return [tuple(sorted(necessary)), tuple(sorted(desired))]


def fuzzy_merge(
    data1,
    data2,
    pid1="",
    pid2="",
    to_tokenize1="",
    to_tokenize2="",
    exact_ids=["property_id", "uprn"],
    output_vars=[],
):
    """
    conducts a fuzzy merge of two data sets.

    data1 : DataFrame
            first data set on which to conduct fuzzy merge
    data2 : DataFrame
            second data set on which to conduct fuzzy merge
    pid1 : string
            unique identifier key for data1
    pid2: string
            unique identifier key for data2
    to_tokenize1 : string
            key for data1 which contains the full property address for the merge
    to_tokenize2 : string
            key for data2 which contains the full property address for the merge
    exact_ids : list (string)
            keys in data1 and data2 on which to conduct a perfect merge
    output_vars : list(string)
            keys to output after merge
    """

    # Create columns with necessary and desired fields
    data1[["necessary", "desired"]] = data1.swifter.apply(
        lambda row: tokenize(row[to_tokenize1], row["postcode"]),
        axis=1,
        result_type="expand",
    )
    data2[["necessary", "desired"]] = data2.swifter.apply(
        lambda row: tokenize(row[to_tokenize2], row["postcode"]),
        axis=1,
        result_type="expand",
    )

    # List of variables
    keys = ["postcode", "necessary", "desired"] + exact_ids

    # Create DF in which to store matches
    matches = []
    unmatched1 = data1
    unmatched2 = data2

    # First, merge on exact IDs
    for merge_key in exact_ids + ["desired", "necessary"]:
        print(f"\nMerging on {merge_key}:")
        total_merge = unmatched1[~unmatched1[merge_key].isna()].merge(
            unmatched2,
            left_on=merge_key,
            right_on=merge_key,
            how="outer",
            indicator=True,
        )

        match = total_merge[total_merge["_merge"] == "both"].copy()
        match[f"{merge_key}_x"] = match[merge_key]
        match[f"{merge_key}_y"] = match[merge_key]
        match["merged_on"] = merge_key
        print("Num Matched:", len(match.index))

        if len(match.index) == 0:
            print("No match")
            continue

        # Add matches to data frame
        matches.append(match[output_vars])

        # Keys to keep
        lhs_keys = [f"{key}_x" for key in keys if key != merge_key] + [merge_key]
        if pid1 not in lhs_keys and pid1.replace("_x", "") != merge_key:
            lhs_keys.append(pid1)

        rhs_keys = [f"{key}_y" for key in keys if key != merge_key] + [merge_key]
        if pid2 not in rhs_keys and pid2.replace("_y", "") != merge_key:
            rhs_keys.append(pid2)

        # Keep track of unmatched data for future merges
        unmatched1_new = total_merge[total_merge["_merge"] == "left_only"][
            lhs_keys
        ].rename(columns={f"{key}_x": key for key in keys})
        unmatched1 = pd.concat(
            [unmatched1_new, unmatched1[unmatched1[merge_key].isna()]]
        )
        # print("Unmatched (all))", len(unmatched1.index))

        unmatched2 = total_merge[total_merge["_merge"] == "right_only"][
            rhs_keys
        ].rename(columns={f"{key}_y": key for key in keys})

    matches = pd.concat(matches)
    # Get number of words in common for merges, in case of duplicates
    matches["common_words"] = matches.apply(
        lambda row: get_common_words(row[pid1], row[pid2]), axis=1
    )
    return matches[output_vars + ["common_words"]], unmatched1, unmatched2


def pick_best_match(match, pid1, pid2, only_pid1=False):
    """
    Function to pick best match when there are duplicate matches
    """
    match = match.drop_duplicates(subset=[pid1, pid2], keep="first")
    len_before = len(match.index)

    for pid in [pid1, pid2]:

        if only_pid1 and pid == pid2:
            continue

        if pid == pid1:
            other_pid = pid2
        else:
            other_pid = pid1

        match["dup"] = match[pid].duplicated(keep=False)
        match["max_common_words"] = match.groupby(pid)["common_words"].transform("max")
        match = match[match.common_words == match.max_common_words]

        # print(f"Dropped {len_before - len(match.index)} entries by using common words.")
        len_before = len(match.index)

        i = 2
        match["dup"] = match[pid].duplicated(keep=False)
        match = match[
            (match.dup == False)
            | (
                match[pid].str.split().str[:i].apply(" ".join)
                == match[other_pid].str.split().str[:i].apply(" ".join)
            )
        ]

        # print(f"Dropped {len_before - len(match.index)} entries by using start of the sentence.")
        len_before = len(match.index)

        match = match.drop_duplicates(subset=[pid], keep=False)

        # print(f"Dropped {len_before - len(match.index)} remaining duplicates of {pid}.")
        len_before = len(match.index)

    return match
