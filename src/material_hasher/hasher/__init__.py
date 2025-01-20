from material_hasher.hasher.entalpic import EntalpicMaterialsHasher, ShortenedEntalpicMaterialsHasher
from material_hasher.hasher.example import SimpleCompositionHasher
from material_hasher.hasher.pdd import PointwiseDistanceDistributionHasher
from material_hasher.hasher.slices import SLICESHasher

__all__ = ["EntalpicMaterialsHasher"]

HASHERS = {
    "Entalpic": EntalpicMaterialsHasher,
    "Shortened Entalpic": ShortenedEntalpicMaterialsHasher,
    "PDD": PointwiseDistanceDistributionHasher,
    "SLICES": SLICESHasher,
}
