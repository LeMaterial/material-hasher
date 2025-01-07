from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pymatgen.core import Structure

from material_hasher.types import StructureEquivalenceChecker


class SimilarityMatcherBase(ABC, StructureEquivalenceChecker):
    """Abstract class for similarity matching between structures."""

    @abstractmethod
    def get_similarity_score(
        self, structure1: Structure, structure2: Structure
    ) -> float:
        """Returns a similarity score between two structures.

        Parameters
        ----------
        structure1 : Structure
            First structure to compare.
        structure2 : Structure

        Returns
        -------
        float
            Similarity score between the two structures.
        """
        pass

    @abstractmethod
    def are_similar(
        self,
        structure1: Structure,
        structure2: Structure,
        threshold: Optional[float] = None,
    ) -> bool:
        """Returns True if the two structures are equivalent according to the
        implemented algorithm.
        Uses a threshold to determine equivalence if provided and the algorithm
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

    def get_pairwise_similarity_scores(
        self,
        structures: list[Structure],
    ) -> np.ndarray:
        """Returns a matrix of similarity scores between structures.

        Parameters
        ----------
        structures : list[Structure]
            List of structures to compare.

        Returns
        -------
        np.ndarray
            Matrix of similarity scores between structures.
        """

        n = len(structures)
        scores = np.zeros((n, n))

        for i, structure1 in enumerate(structures):
            for j, structure2 in enumerate(structures):
                scores[i, j] = self.get_similarity_score(structure1, structure2)

        return scores

    @abstractmethod
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
        pass


class HashingMatcherBase(ABC):
    """Abstract class for matching of the hashes between structures."""

    @abstractmethod
    def are_similar(
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
    def get_hash(
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
