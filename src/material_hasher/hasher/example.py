from pymatgen.core import Structure


class SimpleCompositionHasher:
    """A simple hasher that always returns the composition hash.

    This is just a demo.
    """

    def __init__(self) -> None:
        pass

    def get_material_hash(self, structure: Structure) -> str:
        return structure.composition.reduced_formula
