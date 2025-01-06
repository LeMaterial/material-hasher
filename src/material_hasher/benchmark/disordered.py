from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser


def parse_cif(cif_file):
    parser = CifParser(cif_file)
    structure = parser.get_structures()[0]
    return structure


def get_disordered_structures():
    structures_path = Path("data/examples")
    groups = {}
    groups_names = [f for f in structures_path.iterdir() if f.is_dir()]
    for group in groups_names:
        structures = []
        for cif_file in (group / "results").glob("*.cif"):
            structure = parse_cif(cif_file)
            structures.append(structure)
        groups[group.name] = structures

    return groups


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
    structures = get_disordered_structures()

    from material_hasher.similarity.eqv2 import EquiformerV2Embedder

    embedder = EquiformerV2Embedder(trained=True, cpu=False, threshold=0.01)
    df_results = benchmark_hasher(embedder, structures)
