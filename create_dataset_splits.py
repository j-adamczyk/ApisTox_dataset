import os
import shutil

import pandas as pd
from deepchem.data import NumpyDataset
from deepchem.splits import MaxMinSplitter
from sklearn.model_selection import train_test_split

from config import DATASET_FINAL_FILE_PATH, SPLITS_DIR


def split_dataset(
    df: pd.DataFrame,
    split_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert split_type in {"random", "time", "maxmin"}

    if split_type == "random":
        df_train, df_test = train_test_split(
            df, train_size=0.8, random_state=0, stratify=df["label"]
        )
    elif split_type == "time":
        df = df.sort_values(by="year")
        df_train = df[: int(0.8 * len(df))]
        df_test = df[int(0.8 * len(df)) :]
    else:  # maxmin
        dataset = NumpyDataset(X=df, ids=df["SMILES"])
        splitter = MaxMinSplitter()
        dataset_train, dataset_test = splitter.train_test_split(
            dataset, frac_train=0.8, seed=0
        )
        df_train = pd.DataFrame(dataset_train.X, columns=df.columns)
        df_test = pd.DataFrame(dataset_test.X, columns=df.columns)

    return df_train, df_test


if __name__ == "__main__":
    if os.path.exists(SPLITS_DIR):
        shutil.rmtree(SPLITS_DIR)
    os.mkdir(SPLITS_DIR)

    df = pd.read_csv(DATASET_FINAL_FILE_PATH)

    for split in ["random", "time", "maxmin"]:
        df_train, df_test = split_dataset(df, split_type=split)
        df_train.to_csv(os.path.join(SPLITS_DIR, f"{split}_train.csv"), index=False)
        df_test.to_csv(os.path.join(SPLITS_DIR, f"{split}_test.csv"), index=False)
