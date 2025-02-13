# Copyright 2025 Entalpic
from typing import Dict

from pymatgen.core.structure import Structure


def get_structure_from_hf_row(row: Dict) -> Structure:
    """Get a pymatgen Structure from a dictionary.
    The dictionary should contain the following keys:
        - lattice_vectors: list of lists containing the lattice vectors
        - species_at_sites: list of species at each site
        - cartesian_site_positions: list of cartesian site positions

    Parameters
    ----------
    row : dict
        Dictionary containing the structure information.

    Returns
    -------
    pymatgen.Structure
        Pymatgen Structure object.
    """

    return Structure(
        lattice=[x for y in row["lattice_vectors"] for x in y],
        species=row["species_at_sites"],
        coords=row["cartesian_site_positions"],
        coords_are_cartesian=True,
    )
