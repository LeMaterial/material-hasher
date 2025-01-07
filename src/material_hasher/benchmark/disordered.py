from collections import defaultdict
from typing import List, Dict
from material_hasher.similarity.eqv2 import EquiformerV2Embedder
from pymatgen.core import Structure

from datasets import load_dataset, Dataset
import tqdm

import numpy as np
import pandas as pd

from material_hasher.types import StructureEquivalenceChecker

HF_DISORDERED_PATH = "LeMaterial/sqs_materials"


def get_group_structures_from_data(
    hf_data: Dataset, data_groupby: pd.DataFrame, group_column: str, no_unique: int = 2
) -> Dict[str, List[Structure]]:
    """Get the structures grouped by a column in the data.

    Parameters
    ----------
    hf_data : Dataset
        Hugging Face dataset containing the structures.
    data_groupby : pd.DataFrame
        Dataframe containing the column to group the structures by and ordered
        the same way as the hf_data.
    group_column : str
        Column to group the structures by.
    no_unique : int
        Minimum number of unique structures to consider a group.

    Returns
    -------
    groups_dict : dict
        Dictionary containing the structures grouped by the column.
    """

    assert (
        group_column in data_groupby.columns
    ), f"Column {group_column} not found in data_groupby."

    hf_data = hf_data.select_columns(
        ["lattice_vectors", "species_at_sites", "cartesian_site_positions"]
    )

    groups = data_groupby.groupby(group_column).indices

    groups = {k: v for k, v in groups.items() if len(v) > no_unique}

    print(f"Found {len(groups)} groups with more than {no_unique} structures.")

    hf_data = hf_data.select(np.concatenate(list(groups.values()))).to_pandas()

    new_groups = {}
    cumsum = 0
    for group, indices in groups.items():
        new_groups[group] = np.arange(cumsum, cumsum + len(indices))
        cumsum += len(indices)

    groups_dict = {}
    for group, indices in (
        pbar := tqdm.tqdm(new_groups.items(), desc="Loading groups")
    ):
        pbar.set_postfix_str(str(len(indices)))
        group_rows = hf_data.loc[indices]
        rows = [
            Structure(
                lattice=[x for y in row["lattice_vectors"] for x in y],
                species=row["species_at_sites"],
                coords=row["cartesian_site_positions"],
                coords_are_cartesian=True,
            )
            for _, row in group_rows.iterrows()
        ]
        groups_dict[group] = rows

    return groups_dict


def download_disordered_structures(
    hf_disordered_path: str = HF_DISORDERED_PATH,
) -> Dict[str, List[Structure]]:
    """Download disordered structures from the HF dataset.

    Parameters
    ----------
    hf_disordered_path : str
        Path to the HF dataset containing disordered structures.

    Returns
    -------
    groups_dict : dict
        Dictionary containing the structures grouped by chemical formula.
    """

    hf_data = load_dataset(hf_disordered_path, split="train")
    dataset = hf_data.to_pandas()

    return get_group_structures_from_data(
        hf_data, dataset, "chemical_formula_descriptive"
    )


def get_dissimilar_structures(
    groups_dict: Dict[str, List[Structure]],
) -> Dict[str, List[Structure]]:
    """Get dissimilar structures from the groups dictionary.

    Parameters
    ----------
    groups_dict : dict
        Dictionary containing the structures grouped by chemical formula.

    Returns
    -------
    dissimilar_structures : dict
        Dictionary containing the dissimilar structures.
    """

    dissimilar_structures = defaultdict(list)
    for group, structures in groups_dict.items():
        if len(structures) < 2:
            continue
        dissimilar_structures[group] = structures

    return dissimilar_structures


def get_classification_results(equivalence: np.ndarray) -> dict:
    """Get classification metrics from the pairwise equivalence matrix.
    The metrics are precision, recall, and f1 score.

    Parameters
    ----------
    equivalence : np.ndarray
        Pairwise equivalence matrix.

    Returns
    -------
    metrics : dict
        Dictionary containing the classification metrics.
    """

    TP = np.sum(equivalence)
    FP = np.sum(equivalence == 0)
    FN = np.sum(equivalence == 0)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    metrics = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def benchmark_disordered_structures(
    hasher: StructureEquivalenceChecker,
) -> pd.DataFrame:
    """Benchmark the disordered structures using the given hasher.

    Parameters
    ----------
    hasher : StructureEquivalenceChecker
        Hasher to use for benchmarking.

    Returns
    -------
    df_results : pd.DataFrame
        Dataframe containing the results of the benchmarking.
    """
    print("Downloading disordered structures...")
    structures = download_disordered_structures()
    results = defaultdict(dict)

    print("Benchmarking disordered structures...")
    for group, structures in structures.items():
        print(f"\n\n-- Group: {group} --")
        pairwise_equivalence = hasher.get_pairwise_equivalence(structures)
        triu_indices = np.triu_indices(len(structures), k=1)
        equivalence = np.array(pairwise_equivalence)[triu_indices].astype(int)
        metrics = get_classification_results(equivalence)
        results[group] = metrics
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1: {metrics['f1']}")
        print()

    df_results = pd.DataFrame(results).T
    print(df_results)

    return df_results


if __name__ == "__main__":
    embedder = EquiformerV2Embedder(trained=True, cpu=False, threshold=0.01)
    df_results = benchmark_disordered_structures(embedder)

    import ipdb

    ipdb.set_trace()
