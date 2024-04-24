from typing import Optional

import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles

from dataset_creation.utils import save_excluded_data


def combine_all_datasets(
    df_ecotox: pd.DataFrame,
    df_ppdb: pd.DataFrame,
    df_bpdb: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge datasets together.
    """
    df_combined = pd.concat([df_ecotox, df_ppdb, df_bpdb], ignore_index=True)

    # remove obvious duplicates
    df_combined = df_combined.drop_duplicates(["CAS", "SMILES", "label"])
    df_combined = df_combined.reset_index(drop=True)

    return df_combined


def select_mol_data(mol_group: pd.DataFrame) -> Optional[dict]:
    """
    We select one molecule from a group of molecules with the same CAS or SMILES.

    If they have the same toxicity label across all 3 datasets, we take the
    row from datasets in order of preference: PPDB, BPDB, ECOTOX.
    """
    for source in ["PPDB", "BPDB", "ECOTOX"]:
        row = mol_group[mol_group["source"] == source]
        if not row.empty:
            return row.iloc[0].to_dict()


def deduplicate_dataset(df_combined: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove molecules with given column duplicated across datasets.
    """
    deduplicated_data = []
    for mol_id, mol_group in df_combined.groupby(column):
        if len(mol_group) == 1:
            mol_data = mol_group.iloc[0].to_dict()
        else:
            mol_data = select_mol_data(mol_group)
        deduplicated_data.append(mol_data)

    df_combined = pd.DataFrame.from_records(deduplicated_data)

    return df_combined


def smiles_to_canonical_rdkit(df_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all SMILES strings to canonical SMILES using RDKit.

    Note that the notion of "canonical" SMILES is not universal, but instead
    unambiguous for a given framework. This means that for RDKit those SMILES
    will always result in the same molecules, but they may be different from,
    for example, canonical PubChem SMILES.
    """
    smiles = df_combined["SMILES"]
    canonical_smiles = []
    for smi in smiles:
        mol = MolFromSmiles(smi)
        if mol:
            canonical_smi = MolToSmiles(mol)
            canonical_smiles.append(canonical_smi)
        else:
            canonical_smiles.append(None)

    invalid_smiles_mask = pd.Series(canonical_smiles).isna()
    excluded_data = df_combined[invalid_smiles_mask]
    save_excluded_data(excluded_data, reason="Invalid SMILES")

    df_combined["SMILES"] = canonical_smiles
    df_combined = df_combined[~invalid_smiles_mask]

    return df_combined
