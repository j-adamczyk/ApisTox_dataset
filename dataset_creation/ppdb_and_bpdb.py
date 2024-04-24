import math
import re
from typing import Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

from config import BPDB_FILE_PATH, PPDB_FILE_PATH
from dataset_creation.pubchem import add_cid_numbers
from dataset_creation.utils import save_excluded_data


def get_toxicity_value(html_text: str) -> Tuple[float | str | None, float | str | None]:
    """
    Parses HTML output from PPDB/BPDB page and returns toxicity value, if it's available.

    We take the worst (i.e. the most toxic) of three toxicities: contact, oral, other mode.
    """
    worst_tox_value = math.inf
    worst_tox_type = None
    descriptive_value = None
    for tox_type in ["Contact", "Oral", "Unknown mode"]:
        tox_value = re.search(
            rf"{tox_type} acute LD[\s\S]*?<td class=\"data3\">(.*?)</td>", html_text
        ).group(1)

        # normalize values, e.g. ">= 22.0" -> "22.0"
        tox_value = tox_value.lstrip("<>=").strip()

        # skip unavailable data
        if tox_value == "-":
            continue

        # handle special, weirdly formed cases
        if tox_value in {"Low", "Non-toxic"} or "10<sup>" in tox_value:
            tox_value = 1000
        elif tox_value.startswith("K3 <i>Apis mellifera</i>"):
            tox_value = tox_value.removeprefix("K3 <i>Apis mellifera</i>").strip()
            tox_value = float(tox_value)
        else:
            # default case
            try:
                tox_value = float(tox_value)
            except ValueError:
                # other malformed cases, which could not be handled
                descriptive_value = tox_value
                continue

        if tox_value < worst_tox_value:
            worst_tox_value = tox_value
            worst_tox_type = tox_type

    # normalize toxicity type, to be uniform with ECOTOX
    if worst_tox_type == "Unknown mode":
        worst_tox_type = "Other"

    if worst_tox_value < math.inf:
        # regular toxicity value
        return worst_tox_value, worst_tox_type
    elif descriptive_value is not None:
        # descriptive value, e.g. "Toxic", was the only available data
        return descriptive_value, ""
    else:
        # no data available
        return None, None


def get_smiles(html_text: str) -> Optional[str]:
    smiles = re.search(
        r'Canonical SMILES[\s\S]*?<td class="data1">(.*?)</td>', html_text
    ).group(1)

    # fix malformed cases from BPDB
    smiles = smiles.removeprefix("Major constituent: ")
    smiles = smiles.removeprefix("Anethole: ")
    if smiles == "Al+3].[Al+3].[O-][Si]([O-])([O-])O[Si]([O-])([O-])[O-]":
        smiles = "[" + smiles

    if smiles != "-":
        return smiles
    else:
        return None


def get_pesticide_types(html_text: str) -> tuple[int, int, int, int]:
    """
    Get information about agrochemical application of a pesticide, i.e. whether
    it is a herbicide, fungicide, insecticide.

    If we can find no such information, we return all zeros, since such substances
    can still be used as agrochemicals.
    """
    summary = re.findall(f"report_summary[\s\S]*?<tr>\n<td>(.*)</td>", html_text)
    summary = summary[0] if len(summary) > 0 else ""

    description = re.findall(
        r"Description[\s\S]*?<td class=\"data1\">(.*)</td>", html_text
    )
    description = description[0] if len(description) > 0 else ""

    pesticide_type = re.findall(
        r"[pP]esticide type[\s\S]*?<td class=\"data1\">(.*)</td>", html_text
    )
    pesticide_type = pesticide_type[0] if len(pesticide_type) > 0 else ""

    info = summary + description + pesticide_type

    herbicide = int("herbicide" in info)
    fungicide = int("fungicide" in info)
    insecticide = int("insecticide" in info)
    other = int(not (herbicide or fungicide or insecticide))

    return herbicide, fungicide, insecticide, other


def create_aeru_dataset(database: str, recreate: bool) -> pd.DataFrame:
    """
    Downloads selected data from one of AERU databases: PPDB or BPDB. We download all
    molecules which we have bee toxicity data available.

    If recreate option is False, assumes that files are already downloaded, and just
    loads and returns them.
    """
    if not recreate:
        if database == "PPDB":
            return pd.read_csv(PPDB_FILE_PATH)
        elif database == "BPDB":
            return pd.read_csv(BPDB_FILE_PATH)
        else:
            raise ValueError(f"Database '{database}' not recognized")

    if database == "PPDB":
        url_db = "ppdb/en"
    elif database == "BPDB":
        url_db = "bpdb"
    else:
        raise ValueError(f"Database '{database}' not recognized")

    resp = requests.get(f"https://sitem.herts.ac.uk/aeru/{url_db}/atoz.htm")
    pesticide_ids = re.findall(r"Reports/(\d+).htm", resp.text)

    results = []
    descriptive_value_results = []

    for pesticide_id in tqdm(pesticide_ids):
        url = f"https://sitem.herts.ac.uk/aeru/{url_db}/Reports/{pesticide_id}.htm"
        html_text = requests.get(url).text

        tox_value, tox_type = get_toxicity_value(html_text)
        if not tox_value:
            continue

        name = re.search(r"<title>(.*)</title>", html_text).group(1)
        smiles = get_smiles(html_text)
        cas = re.search(
            r"CAS RN[\s\S]*?<td class=\"data1\">(.*)</td>", html_text
        ).group(1)

        # fix malformed case
        if smiles == "OP([O-])[O-].[K+].[K+]":
            cas = "13492-26-7"

        cid = re.search(
            r"PubChem CID[\s\S]*?<td class=\"data1\">(.*)</td>",
            html_text,
        ).group(1)
        if not cid.isnumeric():
            cid = None

        herbicide, fungicide, insecticide, other = get_pesticide_types(html_text)

        # if we only have descriptive value of toxicity, e.g. "Toxic", we save it separately
        if isinstance(tox_value, str):
            row = {
                "name": name,
                "CID": cid,
                "CAS": cas,
                "SMILES": smiles,
                "source": database,
                "toxicity_type": tox_type,
                "herbicide": herbicide,
                "fungicide": fungicide,
                "insecticide": insecticide,
                "other_agrochemical": other,
                "value": tox_value,
            }
            descriptive_value_results.append(row)
            continue

        label = int(tox_value <= 11)
        if tox_value > 100:
            level = 0
        elif 1 < tox_value <= 100:
            level = 1
        else:
            level = 2

        row = {
            "name": name,
            "CID": cid,
            "CAS": cas,
            "SMILES": smiles,
            "source": database,
            "toxicity_type": tox_type,
            "herbicide": herbicide,
            "fungicide": fungicide,
            "insecticide": insecticide,
            "other_agrochemical": other,
            "label": label,
            "ppdb_level": level,
        }
        results.append(row)

    df = pd.DataFrame.from_records(results)

    invalid_smiles_mask = pd.Series(df["SMILES"]).isna()
    excluded_data = df[invalid_smiles_mask]
    save_excluded_data(excluded_data, reason="No SMILES found")

    descriptive_value_results = pd.DataFrame.from_records(descriptive_value_results)
    save_excluded_data(
        descriptive_value_results, reason="Only descriptive toxicity available"
    )

    df = df[~invalid_smiles_mask]

    # fill missing CID numbers
    df = add_cid_numbers(df)

    if database == "PPDB":
        df.to_csv(PPDB_FILE_PATH, index=False)
    else:
        df.to_csv(BPDB_FILE_PATH, index=False)

    return df
