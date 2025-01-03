from pymatgen.core import Structure
from datasets import load_dataset, VerificationMode



from material_hasher.benchmark.run import HASHERS, get_data_from_hugging_face, diagram_sensitivity
from material_hasher.benchmark.test_cases import ALL_TEST_CASES, get_test_case





# Example usage for TestBenchmark_HG
class TestBenchmark_HG:
    def test_run(self):
        hg_structures = get_data_from_hugging_face("token")
        diagram_sensitivity(hg_structures, "gaussian_noise", "TestDataset", "coordinates", "./output")



        

    

"""
class TestBenchmark:
    def test_run(self):

        structure_data = [
            Structure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["Si"], [[0, 0, 0]]),
            Structure([[3, 2, 9], [1, 3, 0], [1, 0, 3]], ["Au"], [[0, 0, 0]]),
        ]

        # Iterate over hashers in the HASHERS dictionary
        for hasher_name, hasher_func in HASHERS.items():
            print(f"Running benchmark for hasher: {hasher_name}")
            results = benchmark_hasher(
                hasher_func=hasher_func,
                test_cases=ALL_TEST_CASES,
                ignore_test_cases=None,
                structure_data=structure_data,
            )
            print(f"Results for {hasher_name}:")
            print(results)
"""