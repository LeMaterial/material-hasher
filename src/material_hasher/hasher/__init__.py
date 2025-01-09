from .entalpic import EntalpicMaterialsHasher
from .example import SimpleCompositionHasher
from .pdd import PointwiseDistanceDistributionHasher

__all__ = ["EntalpicMaterialsHasher"]

HASHERS = {
    #"Entalpic": EntalpicMaterialsHasher,
    #"SimpleComposition": SimpleCompositionHasher,
    "PDD": PointwiseDistanceDistributionHasher,  # Ajout du PDD hasher
}
