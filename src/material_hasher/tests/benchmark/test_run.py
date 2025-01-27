from material_hasher.benchmark.run_transformations import (
    diagram_sensitivity,
    get_data_from_hugging_face,
)
from material_hasher.benchmark.transformations import ALL_TEST_CASES


# Example usage for TestBenchmark_HG
class TestAllTransformationsBenchmark:
    def test_run(self):
        hg_structures = [get_data_from_hugging_face("token")[0]]
        for test_case in ALL_TEST_CASES:
            diagram_sensitivity(
                hg_structures, test_case, "Test Dataset", test_case, "./output"
            )

    def recup_mat(self):
        return get_data_from_hugging_face("token")
