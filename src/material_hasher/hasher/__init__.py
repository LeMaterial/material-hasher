from .entalpic import EntalpicMaterialsHasher
from .example import SimpleCompositionHasher


__all__ = ["EntalpicMaterialsHasher"]

HASHERS = {
    "Entalpic": EntalpicMaterialsHasher,
    "SimpleComposition": SimpleCompositionHasher,
    # "PDD": PDDMaterialsHasher,  # Ajout du PDD hasher
}
