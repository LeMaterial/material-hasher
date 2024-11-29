from pymatgen.core import Structure

from material_hasher.benchmark.run import HASHERS, benchmark_hasher
from material_hasher.benchmark.test_cases import ALL_TEST_CASES
from material_hasher.hasher.entalpic import EntalpicMaterialsHasher


class TestBenchmark:
    def test_run(self):

        structure_data = [
            Structure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["Si"], [[0, 0, 0]]),
            Structure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["Si"], [[0, 0, 0]]),
            Structure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["Si"], [[0, 0, 0]]),
            Structure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], ["C"], [[0, 0, 0]]),
            Structure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], ["C"], [[0, 0, 0]]),
            Structure([[3, 2, 9], [1, 3, 0], [1, 0, 3]], ["Au"], [[0, 0, 0]]),
        ]

        results = benchmark_hasher(
            hasher_func=EntalpicMaterialsHasher,
            test_cases=ALL_TEST_CASES,
            ignore_test_cases=None,
            structure_data=structure_data,
        )
        print("hello, All_test_cases", ALL_TEST_CASES)
        return results
