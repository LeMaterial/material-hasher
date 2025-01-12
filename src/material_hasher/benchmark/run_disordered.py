from collections import defaultdict
import datetime
from pathlib import Path
import yaml
import os
import tqdm

from typing import Tuple, List, Dict
import time
import numpy as np
import pandas as pd
from pymatgen.core import Structure

from material_hasher.benchmark.disordered import (
    download_disordered_structures,
    get_group_structure_results,
    get_dissimilar_structures,
    get_classification_results_dissimilar,
)
from material_hasher.hasher import HASHERS
from material_hasher.similarity import SIMILARITY_MATCHERS
from material_hasher.types import StructureEquivalenceChecker

STRUCTURE_CHECKERS = {**HASHERS, **SIMILARITY_MATCHERS}


def run_group_structures_benchmark(
    structure_checker: StructureEquivalenceChecker,
    group: str,
    structures: List[Structure],
    n_pick_random: int = 30,
    n_random_structures: int = 30,
    seeds: List[int] = [0, 1, 2, 3, 4],
) -> Dict[str, List[float]]:
    """Run the benchmark for a group of structures.
    If the group has more than n_pick_random structures, pick n_random_structures random structures for all seed in seeds.
    Otherwise, pick all the structures and there is only one success rate.

    Parameters
    ----------
    structure_checker : StructureEquivalenceChecker
        Structure equivalence checker.
    group : str
        Group name.
    structures : List[Structure]
        List of structures in the group.
    n_pick_random : int
        Number of structures to pick randomly.
    n_random_structures : int
        Number of random structures to pick.
    seeds : List[int]
        Seeds for the random number generator.
    """
    if len(structures) > n_pick_random:
        print(
            f"Group {group} has {len(structures)} structures. Taking {min(n_random_structures, len(structures))} random for seeds {seeds}"
        )
        metrics = {"success_rate": []}
        for seed in seeds:
            np.random.seed(seed)
            np.random.shuffle(structures)
            structures_seed = structures[: min(n_random_structures, len(structures))]
            metrics_seed = get_group_structure_results(
                structure_checker, structures_seed
            )
            metrics["success_rate"].append(metrics_seed["success_rate"])
    else:
        print(f"\n\n-- Group: {group} with {len(structures)} structures --")
        metrics = get_group_structure_results(structure_checker, structures)
        metrics["success_rate"] = [metrics["success_rate"]]
    return metrics


def benchmark_disordered_structures(
    structure_checker: StructureEquivalenceChecker,
    seeds: List[int] = [0, 1, 2, 3, 4],
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
    dissimilar_structures = [
        get_dissimilar_structures(structures, seed) for seed in seeds
    ]
    results = defaultdict(dict)

    start_time = time.time()
    print("\n\n-- Dissimilar Structures --")
    dissimilar_metrics = get_classification_results_dissimilar(
        dissimilar_structures, structure_checker
    )
    results["dissimilar_case"] = dissimilar_metrics
    print(
        f"Success rate: {np.mean(dissimilar_metrics['success_rate']) * 100:.2f}%"
        + r" $\pm$ "
        + f"{np.std(dissimilar_metrics['success_rate']) * 100:.2f}%"
    )

    print("Benchmarking disordered structures...")
    for group, structures in tqdm.tqdm(structures.items()):
        metrics = run_group_structures_benchmark(structure_checker, group, structures)
        results[group] = metrics
        print(
            f"Success rate: {(np.mean(metrics['success_rate']) * 100):.2f}%"
            + r" $\pm$ "
            + f"{(np.std(metrics['success_rate']) * 100):.2f}%"
        )
    total_time = time.time() - start_time
    results["total_time (s)"] = total_time

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

    all_results = {}
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
        all_results[structure_checker_name] = df_results
        print(f"{structure_checker_name}: {structure_checker_time:.3f} s")

    if args.algorithm == "all":
        all_results = pd.concat(all_results, names=["algorithm"])
        all_results.to_csv(output_path / "all_results_disordered.csv")


if __name__ == "__main__":
    main()
