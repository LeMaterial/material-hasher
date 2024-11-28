import random
from typing import Optional, Union

import numpy as np
from pymatgen.core import Structure, SymmOp

ALL_TEST_CASES = ["gaussian_noise"]
"""List of all test cases available in the benchmark as ``list[str]``."""


def get_new_structure_with_gaussian_noise(
    structure: Structure, sigma: float = 0.001
) -> Structure:
    """Returns new structure with gaussian noise on atomic positions

    Args:
        structure (Structure): Input structure to modify
        sigma (float, optional): Noise applied to atomic position. Defaults to 0.001.

    Returns:
        Structure: new structure with modification
    """
    return Structure(
        structure.lattice,
        structure.species,
        structure.cart_coords
        + np.random.normal(np.zeros(structure.cart_coords.shape), sigma),
        coords_are_cartesian=True,
    )


def get_new_structure_with_isometric_strain(
    structure: Structure, pct: float = 0.01
) -> Structure:
    """_summary_

    Args:
        structure (Structure): Input structure to modify
        pct (float): Strain applied to lattice in all direction

    Returns:
        Structure: new structure with modification
    """
    s = structure.copy()
    return s.scale_lattice(structure.volume * pct)


def get_new_structure_with_strain(
    structure: Structure, sigma: float = 0.01
) -> Structure:
    """_summary_

    Args:
        structure (Structure): Input structure to modify
        sigma (float): Percent noise applied to lattice vectors

    Returns:
        Structure: new structure with modification
    """
    s = structure.copy()
    return s.apply_strain(np.random.normal(np.zeros(3), sigma))


def get_new_structure_with_translation(
    structure: Structure, sigma: float = 0.1
) -> Structure:
    """_summary_

    Args:
        structure (Structure): Input structure to modify
        sigma (float): Noise to apply to lattice vectors

    Returns:
        Structure: new structure with modification
    """
    s = structure.copy()
    return s.translate_sites(
        [n for n in range(len(structure))], np.random.normal(np.zeros(3), sigma)
    )


def get_new_structure_with_symm_ops(
    structure: Structure,
    symm_ops: Union[SymmOp] = SymmOp.from_rotation_and_translation(),
) -> Structure:
    """_summary_

    Args:
        structure (Structure): Input structure to modify
        symm_ops (Union[SymmOp]): List of symmetry operation to test on structure

    Returns:
        Structure: new structure with modification
    """
    s = structure.copy()
    symm_op = random.choices(symm_ops)
    return s.apply_operation(symm_op)


def make_test_cases(
    test_cases: Optional[list[str]] = None,
    ignore_test_cases: Optional[list[str]] = None,
) -> list[str]:
    """Utility function to generate a list of test cases to run based on the specified test cases and ignored test cases.

    The procedure is as follows:

    1. If ``test_cases`` is not ``None``, include only the specified test cases.
    2. Otherwise, if ``test_cases`` is ``None``, include all test cases (from :const:`ALL_TEST_CASES`).
    3. If ``ignore_test_cases`` is not ``None``, filter the list of test cases to exclude the specified test cases.

    Parameters
    ----------
    test_cases : Optional[list[str]], optional
        List of test cases the user wants, by default ``None``
    ignore_test_cases : Optional[list[str]], optional
        List of test to ignore, by default ``None``

    Returns
    -------
    list[str]
        List of test cases to run.

    Raises
    ------
    ValueError
        If an unknown test case is specified in ``test_cases`` or ``ignore_test_cases``.
    ValueError
        If the resulting list of test cases is empty.
    """
    all_test_cases = ALL_TEST_CASES.copy()

    if test_cases is None:
        test_cases = all_test_cases

    if ignore_test_cases is None:
        ignore_test_cases = []

    for t in test_cases + ignore_test_cases:
        if t not in all_test_cases:
            raise ValueError(f"Unknown test case: {t}")

    if test_cases is not None:
        all_test_cases = [tc for tc in all_test_cases if tc in test_cases]
    if ignore_test_cases is not None:
        all_test_cases = [tc for tc in all_test_cases if tc not in ignore_test_cases]

    if not all_test_cases:
        raise ValueError("No test cases to run.")

    return all_test_cases


def get_test_case(test_case: str) -> dict:
    """Utility function to get test data for a given test case.

    Parameters
    ----------
    test_case : str
        Name of the test case.

    Returns
    -------
    dict
        Dictionary of test data.
    """
    if test_case == "gaussian_noise":
        return get_new_structure_with_gaussian_noise
    else:
        raise ValueError(f"Unknown test case: {test_case}")
