from material_hasher.hasher.entalpic import EntalpicMaterialsHasher
from material_hasher.hasher.example import SimpleCompositionHasher
from material_hasher.hasher.pdd import PointwiseDistanceDistributionHasher
from material_hasher.hasher.slices import SLICESHasher

__all__ = ["EntalpicMaterialsHasher"]

HASHERS = {
    "Entalpic": EntalpicMaterialsHasher,
    "SimpleComposition": SimpleCompositionHasher,
    "PDD": PointwiseDistanceDistributionHasher,
    "SLICES": SLICESHasher,
}
