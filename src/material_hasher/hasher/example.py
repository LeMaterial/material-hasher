from typing import Optional

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

    def is_equivalent(
        self,
        structure1: Structure,
        structure2: Structure,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if two structures are similar based on the StructureMatcher of
        pymatgen. The StructureMatcher uses a similarity algorithm based on the
        maximum common subgraph isomorphism and the Jaccard index of the sites.

        Parameters
        ----------
        structure1 : Structure
            First structure to compare.
        structure2 : Structure
            Second structure to compare.

        Returns
        -------
        bool
            True if the two structures are similar, False otherwise.
        """

        hash_structure1 = self.get_material_hash(structure1)
        hash_structure2 = self.get_material_hash(structure2)

        return hash_structure1 == hash_structure2
