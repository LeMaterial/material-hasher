from collections import defaultdict
from material_hasher.similarity.eqv2 import EquiformerV2Embedder
from pymatgen.core import Structure

from datasets import load_dataset
import tqdm

import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser

HF_DISORDERED_PATH = "LeMaterial/sqs_materials"


def parse_cif(cif_file):
    parser = CifParser(cif_file)
    structure = parser.get_structures()[0]
    return structure


def download_disordered_structures():
    dataset = load_dataset(HF_DISORDERED_PATH, split="train").to_pandas()

    groups = dataset.groupby("chemical_formula_descriptive").indices
    groups_dict = {group: dataset.loc[indices] for group, indices in groups.items()}

    for group, group_rows in tqdm.tqdm(groups_dict.items(), desc="Downloading CIFs"):
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


def get_classification_results(equivalence):
    TP = np.sum(equivalence)
    FP = np.sum(equivalence == 0)
    FN = np.sum(equivalence == 0)
    TN = np.sum(equivalence)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    metrics = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def benchmark_hasher(hasher, structures):
    results = defaultdict(dict)

    for group, structures in structures.items():
        pairwise_equivalence = hasher.get_pairwise_equivalence(structures)
        triu_indices = np.triu_indices(len(structures), k=1)
        equivalence = np.array(pairwise_equivalence)[triu_indices].astype(int)
        metrics = get_classification_results(equivalence)
        results[group] = metrics

    df_results = pd.DataFrame(results)

    return df_results


if __name__ == "__main__":
    structures = download_disordered_structures()

    # structures = get_disordered_structures()

    embedder = EquiformerV2Embedder(trained=True, cpu=False, threshold=0.01)
    df_results = benchmark_hasher(embedder, structures)

    import ipdb

    ipdb.set_trace()
