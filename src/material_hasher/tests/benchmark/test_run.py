from datasets import VerificationMode, load_dataset
from pymatgen.core import Structure

from material_hasher.benchmark.run_transformations import (
    HASHERS,
    diagram_sensitivity,
    get_data_from_hugging_face,
)
from material_hasher.benchmark.transformations import ALL_TEST_CASES, get_test_case


# Example usage for TestBenchmark_HG
class TestBenchmark_HG:
    def test_run(self):
        hg_structures = get_data_from_hugging_face("token")
        diagram_sensitivity(
            hg_structures, "symm_ops", "TestDataset", "symmetries", "./output"
        )

    def recup_mat(self):
        return get_data_from_hugging_face("token")
