from typing import Union

import numpy as np
import pandas as pd

from config import ECOTOX_CLEANED_FILE_PATH, ECOTOX_FILE_PATH, UNITS_TO_EXCLUDE
from dataset_creation.pubchem import add_cid_numbers, get_pesticide_type
from dataset_creation.utils import (
    get_compounds_from_pubchem,
    run_in_parallel,
    save_excluded_data,
)


def load_ecotox_data() -> pd.DataFrame:
    df = pd.read_csv(
        ECOTOX_FILE_PATH,
        sep="|",
        usecols=[
            " Chemical Name",
            "CAS Number ",
            "Exposure Type",
            " Observed Response Mean ",
            " Observed Response Units",
        ],
    )
    df = df.rename(
        columns={
            " Chemical Name": "name",
            "CAS Number ": "CAS",
            "Exposure Type": "exposure_type",
            " Observed Response Mean ": "observed_response_mean",
            " Observed Response Units": "observed_response_unit",
        }
    )
    return df


def remove_invalid_units_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with units that cannot be reasonably transformed to ug/org.
    """
    invalid_units_mask = df["observed_response_unit"].isin(UNITS_TO_EXCLUDE)

    # log removed data
    excluded_data = df[invalid_units_mask]
    save_excluded_data(excluded_data, reason="Invalid units")

    valid_data = df[~invalid_units_mask]
    return valid_data


def standardize_cas_number(cas_ecotox: int) -> str:
    """
    Transform ECOTOX CAS number, which are only integers, into the standard format.
    For example: 108952 -> 108-95-2
    """
    cas_ecotox = str(cas_ecotox)
    cas_standard = f"{cas_ecotox[:-3]}-{cas_ecotox[-3:-1]}-{cas_ecotox[-1]}"
    return cas_standard


def exposure_to_toxicity_type(exposure_types: pd.Series) -> pd.Series:
    """
    Map various exposure types to typical toxicity types: "Oral", "Contact" or "Other".
    """
    toxicity_types_map = {
        "Diet, unspecified": "Oral",
        "Drinking water": "Oral",
        "Food": "Oral",
        "Dermal": "Contact",
        "Direct application": "Contact",
        "Topical, general": "Contact",
        "Multiple routes between application groups": "Other",
        "Oral via capsule": "Other",
        "Spray, unspecified": "Other",
        "Environmental, unspecified": "Other",
    }
    toxicity_types = exposure_types.map(toxicity_types_map)
    return toxicity_types


def responses_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert observed response means to numerical type.
    We remove unknown values ("NR") and fix numbers with errors (slash at the end).
    """
    unknown_response_mask = df["observed_response_mean"] == "NR"

    excluded_data = df[unknown_response_mask]
    save_excluded_data(excluded_data, reason="Unknown observed response mean")

    df = df[~unknown_response_mask]
    df["observed_response_mean"] = df["observed_response_mean"].str.replace("/", "")
    df["observed_response_mean"] = df["observed_response_mean"].astype(float)

    return df


def convert_units_to_ug_per_bee(row: pd.Series) -> float:
    """
    Convert any unit that we reasonably can to ug/bee.
    """
    unit_to_convert = row["observed_response_unit"]
    value = row["observed_response_mean"]

    if unit_to_convert in ["AI ug/org", "AI ug/org/d", "ug/bee", "ug/org", "ug/org/d"]:
        return value
    if unit_to_convert in ["AI ng/org", "AI ng/org/d", "ng/org"]:
        return value / 1000
    elif unit_to_convert in ["AI mg/org", "mg/bee", "mg/org"]:
        return value * 1000
    elif unit_to_convert == "pg/org":
        return value / 1000000
    else:
        # wrong unit - should never happen at this point
        msg = f"Got wrong unit while converting to ug/bee: {unit_to_convert}"
        raise ValueError(msg)


def preprocess_ecotox_data(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_invalid_units_data(df)
    df = responses_to_numeric(df)
    df["observed_response_mean"] = df.apply(convert_units_to_ug_per_bee, axis=1)
    df["CAS"] = df["CAS"].apply(standardize_cas_number)
    df["toxicity_type"] = exposure_to_toxicity_type(df["exposure_type"])
    return df


def get_toxicity_label(measurements: np.ndarray) -> Union[int, str]:
    """
    Transform toxicity measurements for a given molecule into a single label.

    The threshold of LD50 is 11 ug/org. Only if all measurements agree, i.e. all
    are either below or above 11, we assign a label. Otherwise, we assume that
    they vary too much and are inconclusive, and toxicity cannot be determined.
    """
    min_val = measurements.min()
    max_val = measurements.max()
    if min_val <= 11 and max_val <= 11:
        return 1
    elif min_val >= 11 and max_val >= 11:
        return 0
    else:
        return "Unspecified"


def get_ppdb_toxicity_level(measurements: np.ndarray) -> int:
    """
    Transform toxicity measurements for a given molecule into a single level
    according to PPDB methodology. We use median value of measurements.

    Source: https://sitem.herts.ac.uk/aeru/ppdb/en/docs/5_2.pdf
    """
    measurements_median = np.median(measurements)
    if measurements_median > 100:
        return 0
    elif 1 < measurements_median <= 100:
        return 1
    else:
        return 2


def get_toxicity_values_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    We have multiple measurements for each molecule, represented by CAS number.
    First, we calculate statistics for toxicity measurements, and then assign
    """
    records = []
    for (cas_number, tox_type), row in df.groupby(["CAS", "toxicity_type"]):
        measurements = row["observed_response_mean"].values
        label = get_toxicity_label(measurements)
        ppdb_toxicity_level = get_ppdb_toxicity_level(measurements)
        record = {
            "CAS": cas_number,
            "median_measurement": np.median(measurements),
            "toxicity_type": tox_type,
            "toxicity_label": label,
            "ppdb_toxicity_level": ppdb_toxicity_level,
        }
        records.append(record)

    df_toxicity_stats = pd.DataFrame.from_records(records)

    return df_toxicity_stats


def remove_unspecified_data(
    df_tox_data: pd.DataFrame, df_tox_stats: pd.DataFrame
) -> pd.DataFrame:
    # get masks for data with unspecified toxicity level
    mask_toxicity_stats = df_tox_stats["toxicity_label"] == "Unspecified"
    unspecified_cas_numbers = df_tox_stats[mask_toxicity_stats]["CAS"]
    mask_toxicity_data = df_tox_data["CAS"].isin(unspecified_cas_numbers)

    df_tox_unspecified = df_tox_data[mask_toxicity_data]
    save_excluded_data(df_tox_unspecified, reason="Unspecifed toxicity level")

    df_tox_stats = df_tox_stats[~mask_toxicity_stats]
    df_tox_stats["toxicity_label"] = df_tox_stats["toxicity_label"].astype(int)
    return df_tox_stats


def get_toxicity_labels(df_toxicity_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For each exposure type (oral, contact, other) get two toxicity labels:
    - binary toxicity label (EPA guideline)
    - 3-level toxicity level (PPDB specification)

    We take the worst, i.e. strongest, with the lowest median LD50, toxicity
    among 3 exposure ways (oral/contact/other).
    """
    results = []
    for cas_number, data in df_toxicity_stats.groupby("CAS"):
        worst_measurement = np.inf
        worst_type = None
        for tox_type in ["Contact", "Oral", "Other"]:
            row = data[data["toxicity_type"] == tox_type].squeeze()

            if not len(row):
                # no data with particular toxicity type
                continue

            if row["median_measurement"] < worst_measurement:
                worst_measurement = row["median_measurement"]
                worst_type = tox_type

        worst_tox_row = data[data["toxicity_type"] == worst_type].squeeze()

        row = {
            "CAS": cas_number,
            "toxicity_type": worst_type,
            "label": worst_tox_row["toxicity_label"],
            "ppdb_level": worst_tox_row["ppdb_toxicity_level"],
        }
        results.append(row)

    df_toxicity_labels = pd.DataFrame.from_records(results)
    return df_toxicity_labels


def cas_to_smiles(cas: str) -> str:
    compounds = get_compounds_from_pubchem(cas=cas)
    if not compounds:
        return "NOT FOUND"

    smiles_list = [mol.canonical_smiles for mol in compounds]

    if len(set(smiles_list)) == 1:
        return smiles_list[0]  # all SMILES are equal
    else:
        return "AMBIGUOUS"


def add_smiles(df_toxicity_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Add SMILES strings for each molecule based on its CAS number, using PubChem.

    We exclude molecules for which there are many different SMILES results, and for
    which we can't find any SMILES.
    """
    smiles = run_in_parallel(cas_to_smiles, df_toxicity_labels["CAS"])
    df_toxicity_labels["SMILES"] = smiles

    ambiguous_mask = df_toxicity_labels["SMILES"] == "AMBIGUOUS"
    excluded_data = df_toxicity_labels[ambiguous_mask]
    save_excluded_data(excluded_data, reason="Ambiguous SMILES")

    not_found_mask = df_toxicity_labels["SMILES"] == "NOT FOUND"
    excluded_data = df_toxicity_labels[not_found_mask]
    save_excluded_data(excluded_data, reason="No SMILES found")

    good_smiles_mask = (~ambiguous_mask) & (~not_found_mask)
    df_toxicity_labels = df_toxicity_labels[good_smiles_mask]

    return df_toxicity_labels


def create_ecotox_dataset():
    df_ecotox = load_ecotox_data()
    df_ecotox = preprocess_ecotox_data(df_ecotox)

    df_toxicity_stats = get_toxicity_values_stats(df_ecotox)
    df_toxicity_stats = remove_unspecified_data(df_ecotox, df_toxicity_stats)

    df_toxicity_labels = get_toxicity_labels(df_toxicity_stats)

    # add CID numbers and SMILES strings
    df_ecotox_cleaned = add_smiles(df_toxicity_labels)
    df_ecotox_cleaned = add_cid_numbers(df_ecotox_cleaned)

    # add pesticide type information
    pesticide_types = run_in_parallel(get_pesticide_type, df_ecotox_cleaned["CID"])
    df_pesticide_types = pd.DataFrame.from_records(pesticide_types).astype(int)

    df_ecotox_cleaned = pd.merge(
        df_ecotox_cleaned, df_pesticide_types, on="CID", suffixes=("", "")
    )

    # add chemical names and data source information
    df_ecotox_cleaned = pd.merge(
        df_ecotox_cleaned,
        df_ecotox[["CAS", "name"]],
        on="CAS",
        suffixes=("", ""),
    )
    df_ecotox_cleaned["source"] = "ECOTOX"

    # finally, make sure we have no duplicate rows and save results
    df_ecotox_cleaned = df_ecotox_cleaned.drop_duplicates()
    df_ecotox_cleaned.to_csv(ECOTOX_CLEANED_FILE_PATH, index=False)

    return df_ecotox_cleaned
