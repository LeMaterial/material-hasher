from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from pymatgen.core import Structure

from material_hasher.types import StructureEquivalenceChecker


class HasherBase(ABC, StructureEquivalenceChecker):
    """Abstract class for matching of the hashes between structures."""

    @abstractmethod
    def is_equivalent(
        self,
        structure1: Structure,
        structure2: Structure,
        threshold: Optional[float] = None,
    ) -> bool:
        """Returns True if the two structures are similar according to the
        implemented algorithm.
        Uses a threshold to determine similarity if provided and the algorithm
        does not have a built-in threshold.

        Parameters
        ----------
        structure1 : Structure
            First structure to compare.
        structure2 : Structure
            Second structure to compare.
        threshold : float, optional
            Threshold to determine similarity, by default None and the
            algorithm's default threshold is used if it exists.

        Returns
        -------
        bool
            True if the two structures are similar, False otherwise.
        """
        pass

    @abstractmethod
    def get_material_hash(
        self,
        structure: Structure,
    ) -> str:
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
        pass

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

        n = len(structures)
        equivalence_matrix = np.zeros((n, n), dtype=bool)

        # Fill triu + diag
        for i, structure1 in enumerate(structures):
            for j, structure2 in enumerate(structures):
                if i <= j:
                    equivalence_matrix[i, j] = self.is_equivalent(
                        structure1, structure2, threshold
                    )

        # Fill tril
        equivalence_matrix = equivalence_matrix | equivalence_matrix.T

        return equivalence_matrix
