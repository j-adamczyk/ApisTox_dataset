import itertools
import os
import shutil
from inspect import getmembers, isfunction

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from rdkit.Chem import Fragments, MolFromSmiles
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import (
    CalcNumHBA,
    CalcNumHBD,
    CalcNumRotatableBonds,
    CalcTPSA,
    GetMorganFingerprintAsBitVect,
)
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
from rdkit.DataStructs import TanimotoSimilarity
from venny4py.venny4py import venny4py

from config import (
    BEETOX_AI_ANNOTATED_DATA,
    BEETOX_AI_DATA,
    BEETOX_DATA,
    BPDB_FILE_PATH,
    CROP_CSM_DATA,
    ECOTOX_CLEANED_FILE_PATH,
    MOSS_DATA_DIR,
    OUTPUTS_DIR,
    PLOTS_DIR,
    PPDB_FILE_PATH,
    SPLITS_DIR,
)
from dataset_creation.processing import smiles_to_canonical_rdkit


def create_timeline_plot(df: pd.DataFrame) -> None:
    # cumulative timeline plot, i.e. total number of pesticides in each year
    df_years = pd.DataFrame({"year": df["year"], "count": 1})
    df_years = df_years.groupby("year", as_index=False).count()
    df_years["count"] = df_years["count"].cumsum()
    df_years.plot.line(
        x="year",
        y="count",
        xlabel="Year",
        ylabel="Cumulative count",
        legend=False,
    )
    plt.savefig(os.path.join(PLOTS_DIR, "timeline.pdf"))
    plt.clf()


def analyze_overall_class_distributions(df: pd.DataFrame) -> None:
    labels = pd.DataFrame(
        {
            "label": ["non-toxic", "toxic"],
            "value": [(df["label"] == i).sum() for i in [0, 1]],
        }
    )
    ppdb_levels = pd.DataFrame(
        {
            "level": ["non-toxic", "moderately toxic", "highly toxic"],
            "value": [(df["ppdb_level"] == i).sum() for i in [0, 1, 2]],
        }
    )

    print("> Class labels distributions:")

    print(f"Non-toxic count: {(df['label'] == 0).sum()}")
    print(f"Toxic count: {(df['label'] == 1).sum()}")
    print("Binary labels percentages:")
    labels["value"] = labels["value"] * 100 / len(df)
    print(labels)
    print()

    print(f"Non-toxic count: {(df['ppdb_level'] == 0).sum()}")
    print(f"Moderately toxic count: {(df['ppdb_level'] == 1).sum()}")
    print(f"Highly toxic count: {(df['ppdb_level'] == 2).sum()}")
    print("Ternary PPDB levels percentages:")
    ppdb_levels["value"] = ppdb_levels["value"] * 100 / len(df)
    print(ppdb_levels)
    print()

    ax = sns.barplot(
        data=labels,
        x="label",
        y="value",
        hue="label",
        palette=["lightgreen", "red"],
        alpha=0.8,
    )
    ax.set_ylabel("Percentage")
    ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "label_distribution.pdf"))
    plt.clf()

    ax = sns.barplot(
        data=ppdb_levels,
        x="level",
        y="value",
        hue="level",
        palette=["lightgreen", "gold", "red"],
        alpha=0.8,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "levels_distribution.pdf"))
    plt.clf()


def analyze_mol_properties(df: pd.DataFrame) -> None:
    molecules = [MolFromSmiles(smi) for smi in df["SMILES"]]
    property_function = {
        "Molecular Weight": MolWt,
        "LogP": MolLogP,
        "TPSA": CalcTPSA,
        "HB Donors": CalcNumHBD,
        "HB Acceptors": CalcNumHBA,
        "Rotatable Bonds": CalcNumRotatableBonds,
    }
    property_data = df[["CAS", "label"]]
    for name, function in property_function.items():
        property_data[name] = [function(mol) for mol in molecules]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))

    sns.histplot(
        property_data,
        x="Molecular Weight",
        hue="label",
        kde=True,
        ax=axes[0, 0],
        palette=["lightgreen", "red"],
        linewidth=0.05,
        edgecolor="darkgray",
    )
    axes[0, 0].set_title("Molecular Weight")
    axes[0, 0].set_xlim(left=0)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend([], [], frameon=False)

    sns.histplot(
        property_data,
        x="LogP",
        hue="label",
        kde=True,
        ax=axes[0, 1],
        palette=["lightgreen", "red"],
        linewidth=0.05,
        edgecolor="darkgray",
    )
    axes[0, 1].set_title("LogP")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")
    axes[0, 1].legend([], [], frameon=False)

    sns.histplot(
        property_data,
        x="TPSA",
        hue="label",
        kde=True,
        ax=axes[0, 2],
        palette=["lightgreen", "red"],
        linewidth=0.05,
        edgecolor="darkgray",
    )
    axes[0, 2].set_title("TPSA")
    axes[0, 2].set_xlim(left=0)
    axes[0, 2].set_xlabel("")
    axes[0, 2].set_ylabel("")
    legend = axes[0, 2].get_legend()
    handles = legend.legend_handles
    axes[0, 2].legend(handles, ["non-toxic", "toxic"])

    sns.boxplot(
        property_data,
        y="HB Acceptors",
        hue="label",
        legend=False,
        gap=0.1,
        ax=axes[1, 0],
        boxprops=dict(alpha=0.5),
        palette=["lightgreen", "red"],
    )
    axes[1, 0].set_title("Hydrogen bond acceptors (HBA)")
    axes[1, 0].set_ylabel("Count")

    sns.boxplot(
        property_data,
        y="HB Donors",
        hue="label",
        legend=False,
        gap=0.1,
        ax=axes[1, 1],
        boxprops=dict(alpha=0.5),
        palette=["lightgreen", "red"],
    )
    axes[1, 1].set_title("Hydrogen bond donors (HBD)")
    axes[1, 1].set_ylabel("")

    sns.boxplot(
        property_data,
        y="Rotatable Bonds",
        hue="label",
        legend=False,
        gap=0.1,
        ax=axes[1, 2],
        boxprops=dict(alpha=0.5),
        palette=["lightgreen", "red"],
    )
    axes[1, 2].set_title("Number of rotatable bonds")
    axes[1, 2].set_ylabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mol_properties.pdf"), dpi=600)
    plt.clf()


def create_functional_groups_plot(df: pd.DataFrame) -> None:
    mol_data = df[["label"]]
    molecules = [MolFromSmiles(smi) for smi in df["SMILES"]]

    # RDKit calls functional groups "fragments"
    fragment_functions = [
        (name.replace("fr_", ""), function)
        for name, function in getmembers(Fragments, isfunction)
        if name.startswith("fr")
    ]

    for name, function in fragment_functions:
        mol_data[name] = [1 if function(mol) else 0 for mol in molecules]

    frags_data = []
    for name, group in mol_data.groupby("label"):
        counts = group.sum() / len(group)
        counts = counts.drop("label")
        counts.name = "toxic" if name else "non-toxic"
        frags_data.append(counts)

    merged_data = pd.merge(
        frags_data[0], frags_data[1], left_index=True, right_index=True
    )

    merged_data = merged_data.loc[
        (merged_data["toxic"] - merged_data["non-toxic"])
        .sort_values(ascending=False)
        .index[:10],
        :,
    ]
    merged_data = merged_data.sort_values("toxic", ascending=False)
    merged_data = merged_data.reset_index().rename(columns={"index": "Fragment"})

    data_to_plot = merged_data.melt(
        id_vars="Fragment", var_name="Variable", value_name="Value"
    )

    # rescale from [0, 1] to percentages [0, 100]
    data_to_plot["Value"] = 100 * data_to_plot["Value"]

    plt.figure(figsize=(10, 6))

    # bar plot for the first data frame
    ax = sns.barplot(
        data=data_to_plot,
        x="Fragment",
        y="Value",
        hue="Variable",
        alpha=0.8,
        edgecolor="black",
        palette=["lightgreen", "red"],
    )

    # set plot labels and title
    ax.set_xlabel("")
    ax.set_ylabel("Percentage")
    ax.set_title("")

    handler, label = ax.get_legend_handles_labels()
    ax.legend(handler, ["non-toxic", "toxic"])

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "functional_groups.pdf"), dpi=600)
    plt.clf()


def analyze_common_scaffolds(df: pd.DataFrame) -> None:
    data_to_scaffold = df[["CAS", "SMILES", "label"]]
    data_to_scaffold["scaffold"] = data_to_scaffold["SMILES"].apply(
        MurckoScaffoldSmilesFromSmiles
    )
    data_to_scaffold = data_to_scaffold.replace("", "No scaffold")
    num_unique_scaffolds = data_to_scaffold["scaffold"].nunique()
    print("> Common scaffolds analysis")
    print(f"Number of scaffolds: {num_unique_scaffolds}")

    num_single_mol_scaffolds = len(
        data_to_scaffold.groupby("scaffold").filter(lambda x: len(x) == 1)
    )
    print(f"Number of single molecule scaffolds: {num_single_mol_scaffolds}")

    num_no_scaffold_mols = (data_to_scaffold["scaffold"] == "No scaffold").sum()
    print(f"Number of molecules with no scaffold: {num_no_scaffold_mols}")

    num_multi_fragment_mols = (data_to_scaffold["SMILES"].str.contains("\.")).sum()
    num_no_ring_mols = num_no_scaffold_mols - num_multi_fragment_mols
    print(f"Number of multi-fragment molecules: {num_multi_fragment_mols}")
    print(f"Number of molecules with no ring systems: {num_no_ring_mols}")

    most_common_toxic_scaffolds = (
        data_to_scaffold[data_to_scaffold["label"] == 1]["scaffold"]
        .value_counts()
        .index[:10]
    )

    print("Most common toxic scaffolds:")
    print(most_common_toxic_scaffolds.tolist())
    print()


def moss_analysis(df: pd.DataFrame, most_common: int = 10) -> None:
    moss_sub_data_path = os.path.join(MOSS_DATA_DIR, "moss.sub")
    moss_ids_data_path = os.path.join(MOSS_DATA_DIR, "moss.ids")

    moss_sub_df = pd.read_csv(moss_sub_data_path)
    moss_ids_df = pd.read_csv(moss_ids_data_path, sep=":")
    print("> Frequent Subgraph Mining analysis by MoSS:")
    print(f"Number of identified frequent subgraphs: {len(moss_ids_df)}")
    print()

    index_pairs = []
    for _, row in moss_ids_df.iterrows():
        for mol_idx in row["list"].split(","):
            index_pairs.append((row.id, mol_idx))

    df_subgraphs = pd.DataFrame(index_pairs, columns=["subgraph_id", "CAS"])
    df_subgraphs = df_subgraphs.merge(df[["CAS", "label"]], on="CAS")
    df_subgraphs = df_subgraphs.merge(
        moss_sub_df[["id", "description"]], left_on="subgraph_id", right_on="id"
    )
    df_subgraphs = df_subgraphs[["CAS", "label", "description"]]

    toxic_mols_num = (df["label"] == 1).sum()
    nontoxic_mols_num = (df["label"] == 0).sum()

    toxic_mol_subgraphs = df_subgraphs[df_subgraphs["label"] == 1]
    toxic_subgraph_counts = toxic_mol_subgraphs["description"].value_counts()
    toxic_subgraph_counts = toxic_subgraph_counts.reset_index()
    toxic_subgraph_counts["frac"] = toxic_subgraph_counts["count"] / toxic_mols_num
    toxic_subgraph_counts = toxic_subgraph_counts.drop(columns="count")

    nontoxic_mol_subgraphs = df_subgraphs[df_subgraphs["label"] == 0]
    nontoxic_subgraph_counts = nontoxic_mol_subgraphs["description"].value_counts()
    nontoxic_subgraph_counts = nontoxic_subgraph_counts.reset_index()
    nontoxic_subgraph_counts["frac"] = (
        nontoxic_subgraph_counts["count"] / nontoxic_mols_num
    )
    nontoxic_subgraph_counts = nontoxic_subgraph_counts.drop(columns="count")

    subgraph_fraction = pd.merge(
        toxic_subgraph_counts,
        nontoxic_subgraph_counts,
        on="description",
        suffixes=("_toxic", "_nontoxic"),
    )

    # take most_common most discriminative subgraphs, i.e. with the largest difference
    # in frequency among toxic and non-toxic molecules
    subgraph_fraction["diff"] = np.abs(
        subgraph_fraction["frac_toxic"] - subgraph_fraction["frac_nontoxic"]
    )
    subgraph_fraction = subgraph_fraction.sort_values("diff", ascending=False)
    data_to_plot = subgraph_fraction.head(most_common)

    data_to_plot["frac_nontoxic"] = -data_to_plot["frac_nontoxic"]
    data_to_plot = data_to_plot.drop(columns="diff")
    data_to_plot = pd.melt(data_to_plot, id_vars="description")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=data_to_plot,
        x="value",
        y="description",
        hue="variable",
        orient="h",
        dodge=False,
        palette="Set1",
    )
    plt.title("")
    plt.xlabel("Subgraph Frequency")
    plt.ylabel("Subgraph")
    handler, label = ax.get_legend_handles_labels()
    ax.legend(handler, ["toxic", "non-toxic"])

    ticks = [
        -0.7,
        -0.6,
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
    ]
    plt.xticks(ticks=ticks, labels=[abs(tick) for tick in ticks])
    plt.xlim((-0.7, 0.7))
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend([], [], frameon=False)
    plt.savefig(os.path.join(PLOTS_DIR, "moss_frequent_subgraphs_counts.pdf"), dpi=600)


def create_venn_diagram(df: pd.DataFrame) -> None:
    ecotox_data = pd.read_csv(ECOTOX_CLEANED_FILE_PATH)
    ecotox_data = smiles_to_canonical_rdkit(ecotox_data)

    ppdb_data = pd.read_csv(PPDB_FILE_PATH)
    ppdb_data = smiles_to_canonical_rdkit(ppdb_data)

    bpdb_data = pd.read_csv(BPDB_FILE_PATH)
    bpdb_data = smiles_to_canonical_rdkit(bpdb_data)

    smiles_sets = {
        "ApisTox": set(df["SMILES"]),
        "ECOTOX": set(ecotox_data["SMILES"]),
        "PPDB": set(ppdb_data["SMILES"]),
        "BPDB": set(bpdb_data["SMILES"]),
    }

    venny4py(smiles_sets, out=PLOTS_DIR, ext="pdf", dpi=600, size=10)
    os.remove(os.path.join(PLOTS_DIR, "Intersections_4.txt"))
    os.rename(
        src=os.path.join(PLOTS_DIR, "Venn_4.pdf"),
        dst=os.path.join(PLOTS_DIR, "venn_diagram.pdf"),
    )
    plt.clf()


def analyze_statistics(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    # overall size
    print(f"> Split sizes")
    print(f"Train: {len(df_train)}")
    print(f"Test: {len(df_test)}")
    print()

    # class proportions (binary EPA label and 3-level PPDB)
    get_values = (
        lambda x: x.value_counts(normalize=True).mul(100).round().values.astype(int)
    )

    y_train_bin = get_values(df_train["label"])
    y_test_bin = get_values(df_test["label"])

    y_train_ppdb = get_values(df_train["ppdb_level"])
    y_test_ppdb = get_values(df_test["ppdb_level"])

    print("> Class distributions")
    print(f"Binary (EPA): train {y_train_bin}, test {y_test_bin}")
    print(f"3-level (PPDB): train {y_train_ppdb}, test {y_test_ppdb}")
    print()

    # pesticide type distributions (herbicide / insecticide / fungicide / other agrochemical)
    print("> Pesticide types distributions")
    pesticide_types = ["herbicide", "fungicide", "insecticide", "other_agrochemical"]
    for pesticide_type in pesticide_types:
        train_perc = round(100 * df_train[pesticide_type].sum() / len(df_train))
        test_perc = round(100 * df_test[pesticide_type].sum() / len(df_test))
        pesticide = pesticide_type.capitalize().replace("_", " ")
        print(f"{pesticide}: train {train_perc}, test {test_perc}")

    # if all four pesticide types are zero (false), it means that the type is unknown
    train_unknown_type = df_train[df_train[pesticide_types].sum(axis=1) == 0]
    test_unknown_type = df_test[df_test[pesticide_types].sum(axis=1) == 0]
    train_perc = round(100 * len(train_unknown_type) / len(df_train))
    test_perc = round(100 * len(test_unknown_type) / len(df_test))
    print(f"Unknown pesticide type: train {train_perc}, test {test_perc}")


def check_mol_filer_rules(df: pd.DataFrame) -> None:
    mols = [MolFromSmiles(smi) for smi in df["SMILES"]]
    property_function = {
        "Molecular Weight": MolWt,
        "LogP": MolLogP,
        "TPSA": CalcTPSA,
        "HB Donors": CalcNumHBD,
        "HB Acceptors": CalcNumHBA,
        "Rotatable Bonds": CalcNumRotatableBonds,
        "Aromatic Bonds": lambda x: [
            str(bond.GetBondType()) for bond in x.GetBonds()
        ].count("AROMATIC"),
    }
    df_properties = df[["CAS", "label", "herbicide", "fungicide", "insecticide"]]

    for name, function in property_function.items():
        df_properties[name] = [function(mol) for mol in mols]

    def lipinski(x: pd.Series, violations_allowed: int) -> bool:
        return (
            x["Molecular Weight"] <= 500,
            x["HB Acceptors"] <= 10,
            x["HB Donors"] <= 5,
            x["LogP"] <= 5,
        ).count(True) >= 4 - violations_allowed

    def hao(x: pd.Series, violations_allowed: int) -> bool:
        return (
            x["Molecular Weight"] <= 435,
            x["HB Acceptors"] <= 6,
            x["HB Donors"] <= 2,
            x["LogP"] <= 6,
            x["Rotatable Bonds"] <= 9,
            x["Aromatic Bonds"] <= 17,
        ).count(True) >= 6 - violations_allowed

    def tice_herbicides(x: pd.Series, violations_allowed: int) -> bool:
        return (
            150 <= x["Molecular Weight"] <= 500,
            2 <= x["HB Acceptors"] <= 12,
            x["HB Donors"] <= 3,
            x["LogP"] <= 3.5,
            x["Rotatable Bonds"] < 12,
        ).count(True) >= 5 - violations_allowed

    def tice_insecticides(x: pd.Series, violations_allowed: int) -> bool:
        return (
            150 <= x["Molecular Weight"] <= 500,
            1 <= x["HB Acceptors"] <= 8,
            x["HB Donors"] <= 2,
            0 < x["LogP"] <= 5,
            x["Rotatable Bonds"] < 12,
        ).count(True) >= 5 - violations_allowed

    df_herbicides = df_properties[df_properties["herbicide"] == 1]
    df_fungicides = df_properties[df_properties["fungicide"] == 1]
    df_insecticides = df_properties[df_properties["insecticide"] == 1]

    data_subsets_and_rules = [
        ("ApisTox", df_properties, [lipinski, hao]),
        ("Herbicides", df_herbicides, [lipinski, hao, tice_herbicides]),
        ("Fungicides", df_fungicides, [lipinski, hao]),
        ("Insecticides", df_insecticides, [lipinski, hao, tice_insecticides]),
    ]

    print("> Molecular filter rules")
    for name, df_subset, rules in data_subsets_and_rules:
        print(f"Subset {name}")
        for rule in rules:
            print(f"\t{rule.__name__}")
            for violations in [0, 1]:
                print(f"\tAllowed violations: {violations}")
                results = df_subset.apply(lambda x: rule(x, violations), axis=1)
                passed = round(100 * results.sum() / len(results), 1)
                print(f"\tPassed: {passed}%")
                print()


def analyze_distances(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    # ECFP4 fingerprints
    get_ecfp = lambda smi: GetMorganFingerprintAsBitVect(MolFromSmiles(smi), 2, 1024)
    train_fps = [get_ecfp(smi) for smi in df_train["SMILES"]]
    test_fps = [get_ecfp(smi) for smi in df_test["SMILES"]]

    # average Tanimoto distance from test molecule to the nearest training molecule
    avg_test_train_dist = np.mean(
        [
            min(
                TanimotoSimilarity(fp, train_fp, returnDistance=True)
                for train_fp in train_fps
            )
            for fp in test_fps
        ]
    )
    print(f"> Average minimal test-train Tanimoto distance: {avg_test_train_dist:.3f}")

    # average Tanimoto distance in the test set
    avg_test_dist = np.mean(
        [
            TanimotoSimilarity(fp_a, fp_b, returnDistance=True)
            for fp_a, fp_b in itertools.combinations(test_fps, 2)
        ]
    )
    print(f"> Average Tanimoto distance in the test set: {avg_test_dist:.3f}")


def compare_apistox_to_other_sources(df: pd.DataFrame) -> None:
    crop_csm_raw = pd.read_csv(CROP_CSM_DATA)
    beetox_raw = pd.read_csv(BEETOX_DATA)
    beetox_ai_raw = pd.read_csv(BEETOX_AI_DATA)

    dataset_comparison = pd.DataFrame(
        index=[
            "Initial number of molecules",
            "Invalid entries",
            "Duplicated molecules",
            "Cleaned dataset size",
        ],
        columns=["ApisTox", "CropCSM", "BeeTOX", "BeeToxAI"],
    )
    dataset_comparison.loc["Initial number of molecules"] = [
        len(df),
        len(crop_csm_raw),
        len(beetox_raw),
        len(beetox_ai_raw),
    ]

    crop_csm_valid = smiles_to_canonical_rdkit(crop_csm_raw)
    beetox_valid = smiles_to_canonical_rdkit(beetox_raw)
    beetox_ai_valid = smiles_to_canonical_rdkit(beetox_ai_raw)

    dataset_comparison.loc["Invalid entries"] = [
        0,
        len(crop_csm_raw) - len(crop_csm_valid),
        len(beetox_raw) - len(beetox_valid),
        len(beetox_ai_raw) - len(beetox_ai_valid),
    ]

    crop_csm_clean = crop_csm_valid.drop_duplicates("SMILES")
    beetox_clean = beetox_valid.drop_duplicates("SMILES")
    beetox_ai_clean = beetox_ai_valid.drop_duplicates("SMILES")

    dataset_comparison.loc["Duplicated molecules"] = [
        0,
        len(crop_csm_valid) - len(crop_csm_clean),
        len(beetox_valid) - len(beetox_clean),
        len(beetox_ai_valid) - len(beetox_ai_clean),
    ]

    dataset_comparison.loc["Cleaned dataset size"] = [
        len(df),
        len(crop_csm_clean),
        len(beetox_clean),
        len(beetox_ai_clean),
    ]

    apistox_tox_dist = df["label"].value_counts()
    apistox_tox_dist.index = ["Non-toxic molecules", "Toxic molecules"]
    apistox_tox_dist.name = "ApisTox"

    crop_csm_tox_dist = crop_csm_clean["Datasetc"].value_counts()
    crop_csm_tox_dist.index = ["Non-toxic molecules", "Toxic molecules"]
    crop_csm_tox_dist.name = "CropCSM"

    beetox_tox_dist = beetox_clean["Dataset"].value_counts()
    beetox_tox_dist.index = ["Non-toxic molecules", "Toxic molecules"]
    beetox_tox_dist.name = "BeeTOX"

    beetox_ai_annotated_mols = pd.read_csv(BEETOX_AI_ANNOTATED_DATA)
    beetox_ai_tox_dist = beetox_ai_annotated_mols["Outcome"].value_counts()
    beetox_ai_tox_dist.index = ["Non-toxic molecules", "Toxic molecules"]
    beetox_ai_tox_dist.name = "BeeToxAI"

    tox_dist_combined = pd.concat(
        [apistox_tox_dist, crop_csm_tox_dist, beetox_tox_dist, beetox_ai_tox_dist],
        axis=1,
    )

    dataset_comparison = pd.concat([dataset_comparison, tox_dist_combined])
    print("> Comparison of ApisTox to other data sources:")
    print(dataset_comparison)
    print()


if __name__ == "__main__":
    if os.path.exists(PLOTS_DIR):
        shutil.rmtree(PLOTS_DIR)
    os.mkdir(PLOTS_DIR)

    # turn off warnings - data is small, so copying is not a problem
    pd.options.mode.copy_on_write = True

    df = pd.read_csv(os.path.join(OUTPUTS_DIR, "dataset_final.csv"))

    analyze_overall_class_distributions(df)
    analyze_mol_properties(df)
    create_timeline_plot(df)
    check_mol_filer_rules(df)
    create_functional_groups_plot(df)
    analyze_common_scaffolds(df)
    create_venn_diagram(df)
    moss_analysis(df)
    compare_apistox_to_other_sources(df)

    for split in ["random", "time", "maxmin"]:
        print(f"{split.upper()} SPLIT")

        df_train = pd.read_csv(os.path.join(SPLITS_DIR, f"{split}_train.csv"))
        df_test = pd.read_csv(os.path.join(SPLITS_DIR, f"{split}_test.csv"))

        analyze_statistics(df_train, df_test)
        print()
        analyze_distances(df_train, df_test)
        print("\n")
