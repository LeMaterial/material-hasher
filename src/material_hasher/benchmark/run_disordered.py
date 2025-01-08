from collections import defaultdict
import datetime
from pathlib import Path
import yaml
import os
import tqdm

from typing import Tuple
import time
import numpy as np
import pandas as pd

from material_hasher.benchmark.disordered import (
    download_disordered_structures,
    get_classification_results,
    get_dissimilar_structures,
    get_classification_results_dissimilar,
)
from material_hasher.hasher import HASHERS
from material_hasher.similarity import SIMILARITY_MATCHERS
from material_hasher.types import StructureEquivalenceChecker

STRUCTURE_CHECKERS = {**HASHERS, **SIMILARITY_MATCHERS}


def benchmark_disordered_structures(
    structure_checker: StructureEquivalenceChecker,
) -> Tuple[pd.DataFrame, float]:
    """Benchmark the disordered structures using the given hasher or similarity matcher.

    Parameters
    ----------
    hasher : StructureEquivalenceChecker
        Hasher to use for benchmarking.

    Returns
    -------
    df_results : pd.DataFrame
        Dataframe containing the results of the benchmarking.
    total_time : float
        Total time taken for the benchmark
    """
    print("Downloading disordered structures...")
    structures = download_disordered_structures()
    dissimilar_structures = get_dissimilar_structures(structures)
    results = defaultdict(dict)

    start_time = time.time()
    print("\n\n-- Dissimilar Structures --")
    dissimilar_metrics = get_classification_results_dissimilar(
        dissimilar_structures, structure_checker
    )
    results["dissimilar_case"] = dissimilar_metrics
    print(f"Success rate: {(dissimilar_metrics['recall'] * 100):.2f}%")

    print("Benchmarking disordered structures...")
    for group, structures in tqdm.tqdm(structures.items()):
        print(f"\n\n-- Group: {group} with {len(structures)} structures --")
        pairwise_equivalence = structure_checker.get_pairwise_equivalence(structures)
        triu_indices = np.triu_indices(len(structures), k=1)
        equivalence = np.array(pairwise_equivalence)[triu_indices].astype(int)
        metrics = get_classification_results(equivalence)
        results[group] = metrics
        print(f"Success rate: {(metrics['recall'] * 100):.2f}%")
    total_time = time.time() - start_time
    results["total_time"] = {"time": total_time}

    df_results = pd.DataFrame(results).T
    print(df_results)

    return df_results, total_time


def main():
    """
    Run the benchmark for the disordered structures.

    This function provides a command-line interface to benchmark hashers and similarity matchers.

    Get help with:

    .. code-block:: bash

        $ python -m material_hasher.benchmark.run_disordered --help
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Benchmark hashers and similarity matchers for disordered structures."
    )
    parser.add_argument(
        "--algorithm",
        choices=list(STRUCTURE_CHECKERS.keys()) + ["all"],
        help=f"The name of the structure checker to benchmark. One of: {list(STRUCTURE_CHECKERS.keys()) + ['all']}",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for the results. Default: 'results/'",
        default="results/",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Name of the .yaml configuration file to use for the hyperparameters of the hasher. Defaults to default.yaml",
        default="default.yaml",
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(Path("configs") / args.config, "r"))
    output_path = Path(args.output_path) / datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    os.makedirs(output_path, exist_ok=True)
    yaml.dump(config, open(output_path / "config.yaml", "w"))

    if args.algorithm not in STRUCTURE_CHECKERS and args.algorithm != "all":
        raise ValueError(
            f"Invalid algorithm: {args.algorithm}. Must be one of: {list(STRUCTURE_CHECKERS.keys()) + ['all']}"
        )

    for structure_checker_name, structure_checker_class in STRUCTURE_CHECKERS.items():
        if args.algorithm != "all" and structure_checker_name != args.algorithm:
            continue

        structure_checker = structure_checker_class(
            **config.get(structure_checker_name, {})
        )
        df_results, structure_checker_time = benchmark_disordered_structures(
            structure_checker
        )
        df_results.to_csv(
            output_path / f"{structure_checker_name}_results_disordered.csv"
        )
        print(f"{structure_checker_name}: {structure_checker_time:.3f} s")


if __name__ == "__main__":
    main()
