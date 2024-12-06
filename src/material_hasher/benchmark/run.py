from time import time
from typing import Callable, Iterable, Optional

from pymatgen.core import Structure

from material_hasher.benchmark.test_cases import make_test_cases
from material_hasher.hasher.entalpic import EntalpicMaterialsHasher
from material_hasher.hasher.example import SimpleCompositionHasher

HASHERS = {
    "Entalpic": EntalpicMaterialsHasher,
    "SimpleComposition": SimpleCompositionHasher,
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

    times = {"total": 0.0}
    for test_case in test_cases:
        start_time = time()
        for structure in test_data:
            hasher_func().get_material_hash(structure)
        end_time = time()
        times[test_case] = end_time - start_time
        times["total"] += times[test_case]

    return times


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
