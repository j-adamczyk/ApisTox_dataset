import json
import time
from typing import Optional

import pandas as pd
import requests

from dataset_creation.utils import (
    get_compounds_from_pubchem,
    run_in_parallel,
    save_excluded_data,
)


def get_cid(row: pd.Series) -> str | int:
    """
    Get CID (Compound IDentifier) from PubChem, based on CAS number or SMILES string.

    For CAS 39148-24-8 API doesn't return CID, but there is a PubChem page for it, so
    we handle this manually.
    """
    cas = row["CAS"]
    smiles = row["SMILES"]

    if cas == "39148-24-8":
        return 6328269

    compounds = get_compounds_from_pubchem(cas=cas)

    if not compounds or len(compounds) > 1:
        compounds = get_compounds_from_pubchem(smiles=smiles)

    if not compounds:
        return "NOT FOUND"
    elif len(compounds) > 1:
        return "AMBIGUOUS"
    else:
        cid = compounds[0].cid
        return cid if cid is not None else "NOT FOUND"


def add_cid_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Chemical IDentifier (CID) numbers to molecules which do not have it,
    using PubChem data. Lookup is based CAS numbers, or SMILES as backup.

    Most molecules from PPDB and BPDB should already have CID numbers.
    """
    if "CID" not in df.columns:
        df["CID"] = None

    no_cid_mask = df["CID"].isna()
    rows = [row for idx, row in df[no_cid_mask].iterrows()]
    df.loc[no_cid_mask, "CID"] = run_in_parallel(get_cid, rows)

    cid_not_found_mask = df["CID"] == "NOT FOUND"
    excluded_data = df[cid_not_found_mask]
    save_excluded_data(excluded_data, reason="CID not found")

    cid_ambiguous_mask = df["CID"] == "AMBIGUOUS"
    excluded_data = df[cid_ambiguous_mask]
    save_excluded_data(excluded_data, reason="CID ambiguous")

    good_cid_mask = (~cid_not_found_mask) & (~cid_ambiguous_mask)
    df = df[good_cid_mask]

    df["CID"] = df["CID"].astype(int)

    return df


def get_earliest_publication_date(cid: str) -> Optional[str]:
    """
    Get the date of the earliest publication from PubChem where a given molecule appears.
    There are molecules without publications, for which we return None.
    """
    query = {
        "download": "*",
        "collection": "literature",
        "where": {"ands": [{"cid": str(cid)}]},
        "order": ["articlepubdate,asc"],
        "start": 0,
        "limit": 5,  # fetch a few, just in case
        "downloadfilename": f"pubchem_cid_{cid}_literature",
        "nullatbottom": 1,
    }
    base_url = "https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi"
    params = {"infmt": "json", "outfmt": "json", "query": json.dumps(query)}

    def get_publication_date(response: list[dict]) -> Optional[int]:
        for entry in response:
            if "articlepubdate" in entry:
                # skip known erroneous articles
                if entry["articletitle"] in [
                    "A rapid method for the estimation of the environmental parameters octanol/water partition coefficient, soil sorption constant, water to air ratio, and water solubility",
                    "Atmospheric processes",
                    "The assessment of bioaccumulation",
                ]:
                    continue
                return entry["articlepubdate"]
        # if we get here, we could not find any publication date

    while True:
        try:
            response = requests.get(base_url, params=params).json()
            if isinstance(response, list):
                if len(response) >= 1:
                    date = get_publication_date(response)
                    return date
                else:
                    # no literature found
                    return None
            else:
                # most probably "Server too busy" error
                time.sleep(1)
        except requests.ConnectionError:
            # most probably unstable PubChem network
            time.sleep(1)


def get_pubchem_full_json(cid: str) -> dict:
    """
    Returns full information about a compound from PubChem in JSON format. Note that
    it's large and complex.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
    params = {"response_type": "save", "response_basename": f"COMPOUND_CID_{cid}"}
    while True:
        response = requests.get(url, params=params).json()
        if "Record" in response:
            return response["Record"]
        else:
            time.sleep(1)


def get_pubchem_creation_date(cid: str) -> str:
    """
    Fetch "Create date" from PubChem page for a given molecule. This, unfortunately,
    requires parsing raw output from PUG REST API.
    """
    data = get_pubchem_full_json(cid)

    for elem in data["Section"]:
        if elem["TOCHeading"] == "Names and Identifiers":
            data = elem
            break

    for elem in data["Section"]:
        if elem["TOCHeading"] == "Create Date":
            return elem["Information"][0]["Value"]["DateISO8601"][0]

    # we should never get here
    raise ValueError(f"Date could not be found for CID {cid}")


def get_first_publication_date(cid: str) -> str:
    """
    Get first date when molecule was published in literature. If such date is
    not available, we take creation date from PubChem.
    """
    date = get_earliest_publication_date(cid)
    if date:
        return date
    else:
        # no literature found, extract creation date of PubChem page
        return get_pubchem_creation_date(cid)


def get_pesticide_type(cid: str) -> dict[str, int]:
    """
    Get information about agrochemical application of a pesticide, i.e. whether
    it is a herbicide, fungicide, insecticide.

    In other cases typically PubChem notes "Pesticide active substances", so we mark
    all others uniformly as "other". If there is no agrochemical information section,
    we return all zeros (such substances can still be informative in terms of bee
    toxicity).
    """
    data = get_pubchem_full_json(cid)

    info = None
    for section in data["Section"]:
        if section["TOCHeading"] == "Agrochemical Information":
            info = [item["Information"] for item in section["Section"]]
            info = json.dumps(info).lower()
            break

    if not info:
        herbicide, fungicide, insecticide, other = 0, 0, 0, 0
    else:
        herbicide = int("herbicide" in info)
        fungicide = int("fungicide" in info)
        insecticide = int("insecticide" in info)
        other = int(not (herbicide or fungicide or insecticide))

    return {
        "CID": cid,
        "herbicide": herbicide,
        "fungicide": fungicide,
        "insecticide": insecticide,
        "other_agrochemical": other,
    }


def add_first_publication_year(df_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Add first molecule publication year from PubChem.
    """
    dates = run_in_parallel(get_first_publication_date, df_combined["CID"])
    years = [int(date[:4]) for date in dates]
    df_combined["year"] = years

    return df_combined
