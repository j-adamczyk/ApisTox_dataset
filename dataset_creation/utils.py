import os
import time
from typing import Callable, Optional
from urllib.error import URLError

import pandas as pd
import pubchempy as pcp
from joblib import Parallel, delayed
from tqdm import tqdm

from config import EXCLUDED_DATA_FILE_PATH


def save_excluded_data(df: pd.DataFrame, reason: str) -> None:
    if len(df) == 0:
        # no rows were actually excluded
        return

    df["Exclusion reason"] = reason

    if os.path.exists(EXCLUDED_DATA_FILE_PATH):
        with open(EXCLUDED_DATA_FILE_PATH, "a") as file:
            file.write("\n")
        df.to_csv(EXCLUDED_DATA_FILE_PATH, index=False, mode="a")
    else:
        df.to_csv(EXCLUDED_DATA_FILE_PATH, index=False)


class ProgressParallel(Parallel):
    def __init__(self, total: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total = total

    def __call__(self, *args, **kwargs):
        with tqdm(total=self.total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def run_in_parallel(function: Callable, data: list) -> list:
    """
    Runs provided function in parallel on provided data. Returns results in the same
    order as inputs, with the same length.

    We run 5 jobs in parallel, since typically each one takes 1 second, and PUG REST
    API has limit of 5 requests per second.
    """
    tasks = (delayed(function)(item) for item in data)
    results = ProgressParallel(n_jobs=5, total=len(data))(tasks)
    return results


def get_compounds_from_pubchem(
    smiles: Optional[str] = None, cid: Optional[str] = None, cas: Optional[str] = None
) -> list[pcp.Compound]:
    """
    Retrieve compounds from PubChem using one of:
    - CID number
    - CAS  number
    - SMILES string

    Implements retry mechanism to handle 503 PUGREST.ServerBusy errors (overloaded
    PubChem servers).
    """
    if cid is not None:
        key = cid
        namespace = "cid"
    elif cas is not None:
        key = cas
        namespace = "name"
    elif smiles is not None:
        key = smiles
        namespace = "smiles"
    else:
        raise ValueError("One of: CID, CAS or SMILES must be provided")

    while True:
        try:
            return pcp.get_compounds(key, namespace=namespace)
        except URLError as e:
            print(e)
            time.sleep(1)
        except Exception as e:
            if "PUGREST.ServerBusy" in str(e):
                time.sleep(1)
            else:
                input_type = "CAS" if namespace == "name" else namespace.upper()
                msg = f"Problem getting compounds for {input_type} {key}: {e}"
                raise ValueError(msg)
