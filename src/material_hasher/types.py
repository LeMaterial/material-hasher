# Copyright 2025 Entalpic
from typing import Optional, Protocol

import numpy as np
from pymatgen.core import Structure


class StructureEquivalenceChecker(Protocol):
    def is_equivalent(
        self,
        structure1: Structure,
        structure2: Structure,
        threshold: Optional[float] = None,
    ) -> bool: ...

    def get_pairwise_equivalence(
        self,
        structures: list[Structure],
        threshold: Optional[float] = None,
    ) -> np.ndarray: ...
