from pymatgen.core import Structure
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from amd import PDD, PeriodicSet, periodicset_from_gemmi_block, ParseError
from collections import Counter
import numpy as np
from typing import Optional


from hashlib import sha256
from material_hasher.hasher.base import HasherBase

class PointwiseDistanceDistributionHasher(HasherBase):
    def __init__(self, cutoff: float = 100.0):
        """
        Initialize the PDD Generator.

        Parameters:
        cutoff (float): Cutoff distance for PDD calculation. Default is 100.
        """
        self.cutoff = int(cutoff)  # Ensure cutoff is an integer

    def periodicset_from_structure(
            self, structure: Structure
    ) -> PeriodicSet:
        """Convert a pymatgen Structure object to a PeriodicSet.

        Parameters
        ----------
        structure : pymatgen.Structure
            A pymatgen Structure object representing a crystal.

        Returns
        -------
        :class:`amd.PeriodicSet`
            Represents the crystal as a periodic set, consisting of a finite
            set of points (motif) and lattice (unit cell).

        Raises
        ------
        ValueError
            Raised if the structure has no valid sites.
        """

        # Unit cell
        cell = np.array(structure.lattice.matrix)

        # Coordinates and atomic numbers
        coords = np.array(structure.cart_coords)
        atomic_numbers = np.array([site.specie.number for site in structure.sites])

        # Check if the resulting motif is valid
        if len(coords) == 0:
            raise ValueError('The structure has no valid sites after filtering.')

        # Map coordinates to the unit cell (fractional positions mod 1)
        frac_coords = np.mod(structure.lattice.get_fractional_coords(coords), 1)

        motif=frac_coords

        return PeriodicSet(
            motif=motif,
            cell=cell,
            types=atomic_numbers,
        )

    def get_material_hash(self, structure: Structure) -> str:
        """
        Generate a hashed string for a single pymatgen structure based on its
        Point-wise Distance Distribution (PDD).

        Parameters
        ----------
        structure : pymatgen.Structure
            A pymatgen Structure object representing the crystal structure.

        Returns
        -------
        str
            A SHA256 hash string generated from the calculated PDD.
        """
        periodic_set = self.periodicset_from_structure(structure)

        pdd = PDD(periodic_set, int(self.cutoff), collapse=False)  # Ensure cutoff is an integer, without collapsing similar rows
        
        # Round the PDD values to 4 decimal places for numerical stability and consistency.
        pdd = np.round(pdd, decimals=4)
        #print(f"PDD shape: {pdd.shape}")

        # PDD hash array to PDD hash string
        string_pdd = pdd.tobytes()
        string_pdd = sha256(string_pdd).hexdigest()

        return string_pdd

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

        if self.shorten_hash:
            hash_structure1 = shorten_hash(hash_structure1)
            hash_structure2 = shorten_hash(hash_structure2)

        return hash_structure1 == hash_structure2
