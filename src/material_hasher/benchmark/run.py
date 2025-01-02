from time import time
from typing import Callable, Iterable, Optional

from pymatgen.core import Structure

from material_hasher.benchmark.test_cases import get_test_case, make_test_cases
from material_hasher.hasher.cloud import CloudHasher
from material_hasher.hasher.entalpic import EntalpicMaterialsHasher
from material_hasher.hasher.example import SimpleCompositionHasher

HASHERS = {
    "Entalpic": EntalpicMaterialsHasher,
    "SimpleComposition": SimpleCompositionHasher,
    "CLOUD": CloudHasher,
}


def load_structures():
    structures = []

    return structures


def benchmark_hasher(
    hasher_func: Callable,
    test_cases: Optional[Iterable[str]] = None,
    ignore_test_cases: Optional[Iterable[str]] = None,
    structure_data: Optional[Iterable[Structure]] = None,
) -> dict[str, float]:
    """Measure the performance of a hasher function based on test cases listed in the :mod:`material_hasher.benchmark.test_cases` module.

    Parameters
    ----------
    hasher_func : Callable
        A function that takes a single argument, a dictionary of test data, and returns a hash.
    test_cases : Optional[Iterable[str]], optional
        _description_, by default None
    ignore_test_cases : Optional[Iterable[str]], optional
        _description_, by default None

    Returns
    -------
    dict[str, float]
        A dictionary of test case names and their corresponding execution times.

    Raises
    ------
    ValueError
        If no test cases are provided.
    """

    test_cases = make_test_cases(test_cases, ignore_test_cases)
    test_data = structure_data or load_structures()

    results = {}
    i = 0
    for structure in test_data:
        i = i + 1
        structure_key = f"structure {i}"
        results[structure_key] = {}

        case_results = {}
        all_hash = []
        start_time = time()

        original_hash = hasher_func(structure)
        all_hash.append(original_hash)

        for test_case in test_cases:
            print(test_case)
            func, params = get_test_case(test_case)

            test_case_key = f"{test_case}"
            case_results[test_case_key] = {}
            for param_name, param_values in params.items():
                for param_value in param_values:
                    kwargs = {param_name: param_value}

                    case_hashes = [
                        hasher_func().get_material_hash(func(structure, **kwargs))
                        for _ in range(100)
                    ]

                    all_hash = all_hash + case_hashes

                    param_key = f"{param_name}={param_value}"

                    case_results[test_case_key][param_key] = {
                        "hashes": case_hashes,
                        "different_hashes": len(set(case_hashes)),
                    }

        end_time = time()

        print("case_results", case_results)

        results[structure_key] = {
            "different_hashes": len(set(all_hash)),
            "compute_time": end_time - start_time,
        }

    return results


def main():
    """
    Run the benchmark for hashers.

    This function provides a command-line interface to benchmark hashers.

    Get help with:

    .. code-block:: bash

        $ python -m material_hasher.benchmark.run --help
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Benchmark hashers.")
    parser.add_argument(
        "--hasher",
        choices=list(HASHERS.keys()) + ["all"],
        help="The name of the hasher to benchmark.",
    )
    parser.add_argument(
        "--test-cases",
        nargs="+",
        help="The test cases to run. If not provided, all test cases will be run.",
    )
    parser.add_argument(
        "--ignore-test-cases",
        nargs="+",
        help="The test cases to ignore. If not provided, all test cases will be run.",
    )
    args = parser.parse_args()

    for hasher_name, hasher_class in HASHERS.items():
        if args.hasher != "all" and hasher_name != args.hasher:
            continue

        hasher = hasher_class()
        hasher_time = benchmark_hasher(
            hasher.hash, args.test_cases, args.ignore_test_cases
        )
        print(f"{hasher_name}: {hasher_time:.3f} s")


if __name__ == "__main__":
    main()
