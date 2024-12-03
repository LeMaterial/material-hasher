from pymatgen.core import Structure
from datasets import load_dataset, VerificationMode

from material_hasher.benchmark.run import HASHERS, benchmark_hasher
from material_hasher.benchmark.test_cases import ALL_TEST_CASES


class TestBenchmark_HG:
    def test_run_parquet(self):
        # Load dataset using Hugging Face `datasets` library
        ds = load_dataset(
            "LeMaterial/leDataset",
            token="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",
            cache_dir=".",
            data_files=["data/train-00000-of-00018.parquet"],
            verification_mode=VerificationMode.NO_CHECKS,
        )

        # Convert dataset to Pandas DataFrame
        df = ds["train"]
        print("Loaded dataset:", len(df))
        #df = df.select(range(3))

        # Transform dataset int pymatgen Structure objects
        structure_data = []
        for row in df:
            try:
                # Construct the Structure object
                struct = Structure(
                    lattice=row["lattice_vectors"],
                    species=row["species_at_sites"],
                    coords=row["cartesian_site_positions"],
                    coords_are_cartesian=True,
                )
                structure_data.append(struct)

            except Exception as e:
                print(f"Error processing row : {e}")

        print(f"Total structures loaded: {len(structure_data)}")

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
