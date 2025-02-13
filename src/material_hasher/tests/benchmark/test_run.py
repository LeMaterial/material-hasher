# Copyright 2025 Entalpic
from material_hasher.benchmark.run_transformations import (
    diagram_sensitivity,
    get_data_from_hugging_face,
)
from material_hasher.benchmark.transformations import ALL_TEST_CASES


class TestAllTransformationsBenchmark:
    """This test class runs all transformation tests for a single structure from LeMat-Bulk with all parameters.

    To initiate, use test_transformation_hashes().
    """
    def test_transformation_hashes(self):
        """Runs all transformation test for single structure from LeMat-Bulk.

        Returns
        -------
            Pyplot: outputs one Matplotlib plot per transformation test for all hashers. Next transformation freezes until plot it closed.
        """
        hg_structures = [get_data_from_hugging_face("token")[0]]
        for test_case in ALL_TEST_CASES:
            diagram_sensitivity(
                hg_structures, test_case, "Test Dataset", test_case, "./output"
            )

    def test_dataset_download(self):
        """Tests ability to get data from HuggingFace for LeMat-Bulk.

        Returns:
            datasets: LeMat-Bulk datasets object
        """
        return get_data_from_hugging_face("token")
