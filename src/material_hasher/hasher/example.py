# Copyright 2025 Entalpic
from pymatgen.core import Structure

from material_hasher.hasher.base import HasherBase


class SimpleCompositionHasher(HasherBase):
    """A simple hasher that always returns the composition hash.

    This is just a demo.
    """

    def __init__(self) -> None:
        pass

    def get_material_hash(self, structure: Structure) -> str:
        """Returns a hash of the structure.

        Parameters
        ----------
        structure : Structure
            Structure to hash.

        Returns
        -------
        str
            Hash of the structure.
        """
        return structure.composition.reduced_formula
