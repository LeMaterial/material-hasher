# Copyright 2025 Entalpic
from typing import Optional

import numpy as np
from amd import PDD, PeriodicSet
from pymatgen.core import Structure

from material_hasher.hasher.base import HasherBase


class PointwiseDistanceDistributionHasher(HasherBase):
    def __init__(self, cutoff: float = 100.0, threshold: float = 1e-3):
        """
        Initialize the PDD Generator.

        Parameters:
        cutoff (float): Cutoff distance for PDD calculation. Default is 100.
        threshold (float): Threshold for PDD comparison of two crystals. Default is 1e-3.
        """
        self.cutoff = int(cutoff)  # Ensure cutoff is an integer
        self.threshold = threshold

    def periodicset_from_structure(self, structure: Structure) -> PeriodicSet:
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
            raise ValueError("The structure has no valid sites after filtering.")

        # Map coordinates to the unit cell (fractional positions mod 1)
        frac_coords = np.mod(structure.lattice.get_fractional_coords(coords), 1)

        motif = frac_coords

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

        pdd = PDD(
            periodic_set, int(self.cutoff), collapse=False
        )

        return pdd

    def is_hash_equivalent(
        self,
        hash1: np.ndarray,
        hash2: np.ndarray,
        threshold: Optional[float] = None,
    ) -> bool:
        """Check if two PDD hashes are equivalent.
        PDD hashes are numpy arrays and are considered equivalent if the
        Euclidean distance between them is less than a given threshold.

        Parameters
        ----------
        hash1 : np.ndarray
            First hash to compare.
        hash2 : str
            Second hash to compare.
        threshold : float, optional
            Threshold to determine similarity, by default None and the
            algorithm's default threshold is used if it exists.
            Some algorithms may not have a threshold.

        Returns
        -------
        bool
            True if the two hashes are equivalent, False otherwise.
        """
        # TODO: Should we use euclidean distance or something else?
        if hash1.shape != hash2.shape:
            return False

        if threshold is None:
            threshold = self.threshold

        return np.allclose(hash1, hash2, atol=threshold)

    def get_pairwise_equivalence(
        self, structures: list[Structure], threshold: Optional[float] = None
    ) -> np.ndarray:
        """Returns a matrix of equivalence between structures.

        Parameters
        ----------
        structures : list[Structure]
            List of structures to compare.
        threshold : float, optional
            Threshold to determine similarity, by default None and the
            algorithm's default threshold is used if it exists.

        Returns
        -------
        np.ndarray
            Matrix of equivalence between structures.
        """

        all_hashes = np.array(self.get_materials_hashes(structures))
        equivalence_matrix = np.zeros((len(all_hashes), len(all_hashes)), dtype=bool)

        # Fill triu + diag
        for i, hash1 in enumerate(all_hashes):
            for j, hash2 in enumerate(all_hashes):
                if i <= j:
                    equivalence_matrix[i, j] = self.is_hash_equivalent(
                        hash1, hash2, threshold
                    )

        # Fill tril
        equivalence_matrix = equivalence_matrix | equivalence_matrix.T

        return equivalence_matrix
