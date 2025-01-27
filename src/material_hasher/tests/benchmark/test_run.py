from material_hasher.benchmark.run_transformations import (
    diagram_sensitivity,
    get_data_from_hugging_face,
)


# Example usage for TestBenchmark_HG
class TestBenchmark_HG:
    def test_run(self):
        hg_structures = get_data_from_hugging_face("token")
        diagram_sensitivity(
            hg_structures, "symm_ops", "TestDataset", "symmetries", "./output"
        )

    def recup_mat(self):
        return get_data_from_hugging_face("token")
