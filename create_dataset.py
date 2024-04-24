import argparse
import os
import shutil

import pandas as pd

from config import DATASET_FINAL_FILE_PATH, OUTPUTS_DIR
from dataset_creation.ecotox import create_ecotox_dataset
from dataset_creation.ppdb_and_bpdb import create_aeru_dataset
from dataset_creation.processing import (
    combine_all_datasets,
    deduplicate_dataset,
    smiles_to_canonical_rdkit,
)
from dataset_creation.pubchem import add_first_publication_year


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate_ppdb", default=False, type=bool)
    parser.add_argument("--recreate_bpdb", default=False, type=bool)
    return parser.parse_args()


def create_dataset() -> None:
    args = parse_args()

    if os.path.exists(OUTPUTS_DIR):
        shutil.rmtree(OUTPUTS_DIR)
    os.mkdir(OUTPUTS_DIR)

    # turn off SettingWithCopyWarning
    pd.options.mode.chained_assignment = None

    df_ecotox = create_ecotox_dataset()

    df_ppdb = create_aeru_dataset("PPDB", recreate=args.recreate_ppdb)
    df_bpdb = create_aeru_dataset("BPDB", recreate=args.recreate_bpdb)

    df_combined = combine_all_datasets(df_ecotox, df_ppdb, df_bpdb)
    df_combined = smiles_to_canonical_rdkit(df_combined)
    df_combined = deduplicate_dataset(df_combined, column="SMILES")
    df_combined = deduplicate_dataset(df_combined, column="CAS")
    df_combined = add_first_publication_year(df_combined)

    df_final = df_combined[
        [
            "name",
            "CID",
            "CAS",
            "SMILES",
            "source",
            "year",
            "toxicity_type",
            "herbicide",
            "fungicide",
            "insecticide",
            "other_agrochemical",
            "label",
            "ppdb_level",
        ]
    ]
    df_final = df_final.sort_values(by="year")

    df_final.to_csv(DATASET_FINAL_FILE_PATH, index=False)


if __name__ == "__main__":
    create_dataset()
