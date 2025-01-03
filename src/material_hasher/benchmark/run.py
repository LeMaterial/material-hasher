from time import time
from typing import Callable, Iterable, Optional
import matplotlib.pyplot as plt
import os

from pymatgen.core import Structure
from datasets import load_dataset, VerificationMode


from material_hasher.benchmark.test_cases import (
    make_test_cases,
    get_test_case,
)
from material_hasher.hasher.entalpic import EntalpicMaterialsHasher
from material_hasher.hasher.example import SimpleCompositionHasher
from material_hasher.hasher.pdd import PDDMaterialsHasher

HASHERS = {
    "Entalpic": EntalpicMaterialsHasher,
    "SimpleComposition": SimpleCompositionHasher,
    #"PDD": PDDMaterialsHasher,  # Ajout du PDD hasher
}


def get_data_from_hugging_face(token: str):
    """
    Downloads and processes structural data from the Hugging Face `datasets` library.

    This function fetches a dataset from Hugging Face, extracts relevant structural information,
    and converts it into a list of pymatgen Structure objects.

    Parameters
    ----------
    token : str
        The authentication token required to access the dataset.

    Returns
    -------
    list[Structure]
        A list of pymatgen Structure objects extracted and processed from the dataset.

    Raises
    ------
    ValueError
        If the dataset fails to load or the structures cannot be processed.

    Notes
    -----
    - The dataset is fetched from the `LeMaterial/LeMat-Bulk` repository using the 
      `compatible_pbe` configuration.
    - Only the first entry of the dataset is selected for processing.
    - Errors during the transformation process are logged but do not halt execution.
    """
    ds = load_dataset(
        "LeMaterial/LeMat-Bulk",
        "compatible_pbe",
        token=token,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    # Convert dataset to Pandas DataFrame
    df = ds["train"]
    print("Loaded dataset:", len(df))
    df = df.select(range(2))

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
            # Log errors without interrupting processing
            print(f"Error processing row : {e}")

    # Display the total number of successfully loaded structures
    print(f"structure_data size: {len(structure_data)}")
    
    # Return the list of pymatgen Structure objects
    return structure_data

def apply_transformation(
    structure: Structure,
    test_case: str,
    parameter: tuple[str, any],
) -> list[Structure]:
    """
    Applies transformations to a structure using a specified test case and parameter.

    Parameters
    ----------
    structure : Structure
        Input structure to be transformed.
    test_case : str
        Test case to be applied.
    parameter : tuple[str, any]
        Name and value of the parameter to be used for the transformation.

    Returns
    -------
    list[Structure]
        List of transformed structures.

    Raises
    ------
    ValueError
        If no valid test case is provided.
    """

    if not structure:
        raise ValueError("No structure was provided.")

    if not test_case:
        raise ValueError("No test case was provided.")

    if not parameter:
        raise ValueError("No parameter was provided.")

    # Load the test case
    func, params = get_test_case(test_case)

    transformed_structures = []

    # Extract parameter name and value
    param_name, param_value = parameter
    kwargs = {param_name: param_value}

    # Apply the transformation and add the transformed structures
    for _ in range(100):
        transformed_structure = func(structure, **kwargs)
        transformed_structures.append(transformed_structure)

    return transformed_structures

def hasher_sensitivity(structure: Structure, transformed_structures: list[Structure], hasher_name: str) -> float:
    """
    Computes the proportion of transformed structures with hashes equal to the original structure's hash.

    Parameters
    ----------
    structure : Structure
        The original structure.
    transformed_structures : list[Structure]
        List of transformed structures.
    hasher_name : str
        Name of the hasher to be used.

    Returns
    -------
    float
        Proportion of hashes equal to the original hash.
    """
    if hasher_name not in HASHERS:
        raise ValueError(f"Unknown hasher: {hasher_name}")

    hasher = HASHERS[hasher_name]()

    # Compute hash for the original structure
    original_hash = hasher.get_material_hash(structure)
    print('original hash:', original_hash)
    # Compute hashes for transformed structures
    transformed_hashes = [hasher.get_material_hash(s) for s in transformed_structures]

    # Calculate the proportion of hashes matching the original hash
    matching_hashes = sum(1 for h in transformed_hashes if h == original_hash)
    return matching_hashes / len(transformed_structures)


def mean_sensitivity(structure_data: list[Structure], test_case: str, parameter: tuple[str, any], hasher_name: str) -> float:
    """
    Computes the mean sensitivity for all structures in the dataset.

    Parameters
    ----------
    structure_data : list[Structure]
        List of structures to be processed.
    test_case : str
        Test case to be applied.
    parameter : tuple[str, any]
        Name and value of the parameter to be used for the transformation.
    hasher_name : str
        Name of the hasher to be used.

    Returns
    -------
    float
        Mean sensitivity across all structures.
    """
    sensitivities = []

    for structure in structure_data:
        print('new material in process !')
        # Apply transformation
        transformed_structures = apply_transformation(structure, test_case, parameter)
        # Compute sensitivity
        sensitivity = hasher_sensitivity(structure, transformed_structures, hasher_name)
        print('sensitivity:', sensitivity)
        sensitivities.append(sensitivity)

    # Calculate and return mean sensitivity
    return sum(sensitivities) / len(sensitivities)


def sensitivity_over_parameter_range(structure_data: list[Structure], test_case: str, hasher_name: str) -> dict[float, float]:
    """
    Computes mean sensitivity for a range of parameter values from PARAMETERS.

    Parameters
    ----------
    structure_data : list[Structure]
        List of structures to be processed.
    test_case : str
        Test case to be applied.
    hasher_name : str
        Name of the hasher to be used.

    Returns
    -------
    dict[float, float]
        Dictionary mapping each parameter value to its mean sensitivity.
    """
    # Load parameters from test cases
    _, params = get_test_case(test_case)
    param_name = list(params.keys())[0]  # Generalize to fetch the first parameter name
    print('param_name:', param_name)

    param_range = params[param_name]
    print('param_range:', param_range)


    results = {}
    for param_value in param_range:
        parameter = (param_name, param_value)
        mean_sens = mean_sensitivity(structure_data, test_case, parameter, hasher_name)
        results[param_value] = mean_sens
        print('results', results)

    

    return results


def benchmark_hasher(structure_data: list[Structure], test_case: str) -> dict[str, dict[float, float]]:
    """
    Benchmarks all hashers over parameter ranges.

    Parameters
    ----------
    structure_data : list[Structure]
        List of structures to be processed.
    test_case : str
        Test case to be applied.

    Returns
    -------
    dict[str, dict[float, float]]
        Dictionary containing results for each hasher.
    """
    results = {}
    for hasher_name in HASHERS.keys():
        results[hasher_name] = sensitivity_over_parameter_range(structure_data, test_case, hasher_name)
    return results

def diagram_sensitivity(structure_data: list[Structure], test_case: str, dataset_name: str, noise_type: str, output_dir: str):
    """
    Generates and saves sensitivity diagrams for all hashers.

    Parameters
    ----------
    structure_data : list[Structure]
        List of structures to be processed.
    test_case : str
        Test case to be applied.
    dataset_name : str
        Name of the dataset.
    noise_type : str
        Type of noise added.
    output_dir : str
        Directory to save the output plot.
    """
    results = benchmark_hasher(structure_data, test_case)

    plt.figure(figsize=(10, 6))
    for hasher_name, data in results.items():
        param_range = list(data.keys())
        sensitivities = list(data.values())
        plt.plot(param_range, sensitivities, label=hasher_name, marker="o")

    plt.xlabel("Parameter Value")
    plt.ylabel("Mean Sensitivity")
    plt.title(f"{dataset_name} with noise on {noise_type}")
    plt.legend()
    plt.grid(True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{dataset_name}_{noise_type}_sensitivity_diagram.png")
    plt.savefig(output_path, dpi=600, bbox_inches="tight", format="png")
    plt.show()



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
