from typing import List, Dict
from material_hasher.types import StructureEquivalenceChecker
from pymatgen.core import Structure
from material_hasher.benchmark.utils import get_structure_from_dict

from datasets import load_dataset, Dataset
import tqdm

import numpy as np
import pandas as pd


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
        rows = [get_structure_from_dict(row) for _, row in group_rows.iterrows()]
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
    groups_dict: Dict[str, List[Structure]], n_picked_per_pair=10, seed=0
) -> List[Structure]:
    """Get dissimilar structures from the groups dictionary.

    Parameters
    ----------
    groups_dict : dict
        Dictionary containing the structures grouped by chemical formula.
    n_picked_per_pair : int
        Number of pairs of structure to pick for two disjoint groups of structures.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    dissimilar_structures : list
        List of couples of dissimilar structures.
    """
    from itertools import combinations

    n_picked_per_pair = 10
    np.random.seed(seed)
    dissimilar_structures = []

    all_group_names = list(groups_dict.keys())
    all_pairs = list(combinations(all_group_names, 2))

    for pair in all_pairs:
        group1 = groups_dict[pair[0]]
        group2 = groups_dict[pair[1]]
        for _ in range(n_picked_per_pair):
            structure1 = np.random.choice(list(range(len(group1))))
            structure1 = group1[structure1]
            structure2 = np.random.choice(list(range(len(group2))))
            structure2 = group2[structure2]
            dissimilar_structures.append((structure1, structure2))

    return dissimilar_structures


def get_classification_results(equivalence: np.ndarray) -> dict:
    """Get classification metrics from the pairwise equivalence matrix.
    Since all samples are labeled similar in this case, only the recall is interesting

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
    FN = np.sum(equivalence == 0)
    recall = TP / (TP + FN)
    metrics = {"recall": recall}
    return metrics


def get_classification_results_dissimilar(
    dissimilar_structures: List[Structure],
    structure_checker: StructureEquivalenceChecker,
) -> Dict[str, float]:
    """Get classification metrics from the dissimilar structures.
    Only the recall is interesting in this case because all samples are labeled dissimilar (so positive in this case).

    Parameters
    ----------
    dissimilar_structures : List[Structure]
        List of dissimilar structures.
    structure_checker : StructureEquivalenceChecker
        Structure equivalence checker.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing the classification metrics.
    """
    TP = 0
    FN = 0
    for structure1, structure2 in tqdm.tqdm(dissimilar_structures, desc="Dissimilar"):
        is_equivalent = structure_checker.is_equivalent(structure1, structure2)
        TP += int(
            not is_equivalent
        )  # The structures are not equivalent, so the prediction is correct
        FN += int(is_equivalent)

    recall = TP / (TP + FN)
    return {"recall": recall}
