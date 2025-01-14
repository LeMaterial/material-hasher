from .entalpic import EntalpicMaterialsHasher
from .example import SimpleCompositionHasher
from .pdd import PointwiseDistanceDistributionHasher
from .slices import SLICESHasher

__all__ = ["EntalpicMaterialsHasher"]

HASHERS = {
    "Entalpic": EntalpicMaterialsHasher,
<<<<<<< HEAD
    #"SimpleComposition": SimpleCompositionHasher,
=======
    "SimpleComposition": SimpleCompositionHasher,
>>>>>>> bb0e7a1d5479deb6beb98dd74de83cdc9264be85
    "PDD": PointwiseDistanceDistributionHasher,  # Ajout du PDD hasher
    "SLICES": SLICESHasher,
}
