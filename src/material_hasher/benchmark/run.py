from time import time
from typing import Callable, Iterable, Optional

from pymatgen.core import Structure

from material_hasher.benchmark.test_cases import (
    make_test_cases,
    get_test_case,
)
from material_hasher.hasher.entalpic import EntalpicMaterialsHasher
from material_hasher.hasher.example import SimpleCompositionHasher

HASHERS = {
    "Entalpic": EntalpicMaterialsHasher,
    "SimpleComposition": SimpleCompositionHasher,
}


def load_structures():
    structures = []

    return structures


def count_duplicates(hashes: list[str]) -> int:
    """
    Count the number of duplicate entries in a list of hashes.

    A duplicate is defined as any hash that occurs more than once in the list.
    This function counts the total number of duplicate occurrences.

    Parameters
    ----------
    hashes : list of str
        A list of hash strings to analyze for duplicates.

    Returns
    -------
    int
        The total count of duplicate entries in the list. Each duplicate is counted
        for the number of times it exceeds one occurrence.

    Examples
    --------
    >>> hashes = ["abc123", "def456", "abc123", "xyz789", "abc123"]
    >>> count_duplicates(hashes)
    3
    """
    from collections import Counter

    hash_counts = Counter(hashes)
    duplicates = sum(count for count in hash_counts.values() if count > 1)
    return duplicates


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
    for test_case in test_cases:
        func, params = get_test_case(test_case)

        case_results = {}
        for param_name, param_values in params.items():
            for param_value in param_values:
                kwargs = {param_name: param_value}
                start_time = time()
                case_hashes = []
                for structure in test_data:
                    transformed_structure = func(structure, **kwargs)
                    case_hashes.append(
                        hasher_func().get_material_hash(transformed_structure)
                    )
                end_time = time()
                param_key = f"{param_name}={param_value}"

                duplicates = count_duplicates(case_hashes)

                case_results[param_key] = {
                    "execution_time": end_time - start_time,
                    "duplicates": duplicates,
                }

        results[test_case] = {
            "parameters": case_results,
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
