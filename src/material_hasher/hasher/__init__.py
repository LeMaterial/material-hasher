from .entalpic import EntalpicMaterialsHasher
from .example import SimpleCompositionHasher
from .pdd import PointwiseDistanceDistributionHasher
from .slices import SLICESHasher

__all__ = ["EntalpicMaterialsHasher"]

HASHERS = {
    "Entalpic": EntalpicMaterialsHasher,
    "SimpleComposition": SimpleCompositionHasher,
    "PDD": PointwiseDistanceDistributionHasher,
    "SLICES": SLICESHasher,
}
