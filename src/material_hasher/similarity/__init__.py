# Copyright 2025 Entalpic
# from .eqv2 import EquiformerV2Similarity
from .structure_matchers import PymatgenStructureSimilarity

# __all__ = ["EquiformerV2Similarity", "PymatgenStructureSimilarity"]

SIMILARITY_MATCHERS = {
    # "eqv2": EquiformerV2Similarity,
    "pymatgen": PymatgenStructureSimilarity,
}
