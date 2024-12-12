from pymatgen.analysis.local_env import EconNN, NearNeighbors
from pymatgen.core import Structure

from material_hasher.hasher.utils.graph import get_weisfeiler_lehman_hash
from material_hasher.hasher.utils.graph_structure import get_structure_graph
from material_hasher.hasher.utils.symmetry import AFLOWSymmetry, SPGLibSymmetry


class EntalpicMaterialsHasher:
    """Materials fingerprint method proposed by Entalpic.
    Returns hash based on bonding graph structure, composition,
    and symmetry.
    """

    def __init__(
        self,
        graphing_algorithm: str = "WL",
        bonding_algorithm: NearNeighbors = EconNN,
        bonding_kwargs: dict = {},
        include_composition: bool = False,
        symmetry_labeling: str = "SPGLib",
    ):
        """Generate fingerprint for given Pymatgen structure.

        Method proposed by Entalpic.

        Args:
            graphing_algorithm (str, optional): Graph hashing algorithm.
                Currently only Weisfeiler-Lehman (WL) is implemented.
                Defaults to "WL".
            bonding_algorithm (NearNeighbors, optional): Pymatgen
                NearNeighbors class to compute bonds and create
                bonded structure. Defaults to EconNN.
            bonding_kwargs (dict, optional): kwargs passed to the
                Near Neighbor class. Defaults to {}.
            include_composition (bool, optional): Whether to
                include composition in the hash. Defaults to False.
            symmetry_labeling (str, optional): Method for deciding symmetry
                label. Only AFLOW or SPGLib implemented. AFLOW requires
                AFLOW python packages. Defaults to "SPGLib".
        """
        self.graphing_algorithm = graphing_algorithm
        self.bonding_algorithm = bonding_algorithm
        self.bonding_kwargs = bonding_kwargs
        self.include_composition = include_composition
        self.symmetry_labeling = symmetry_labeling

    def get_entalpic_materials_data(self, structure: Structure) -> dict:
        """Gets various hash component for given
        Pymatgen structure.

        Args:
            structure (Structure): pymatgen structure object

        Raises:
            ValueError: if non-implemented hash method
                is request.
            ValueError: if non implemented symmetry method
                is requested.

        Returns:
            dict: data dictionary with all hash components
        """
        data = dict()
        if self.graphing_algorithm == "WL":
            graph = get_structure_graph(
                structure,
                bonding_kwargs=self.bonding_kwargs,
                bonding_algorithm=self.bonding_algorithm,
            )
            data["bonding_graph_hash"] = get_weisfeiler_lehman_hash(graph)
        else:
            raise ValueError(
                "Graphing algorithm {} not implemented".format(self.graphing_algorithm)
            )
        if self.symmetry_labeling == "AFLOW":
            data["symmetry_label"] = AFLOWSymmetry().get_symmetry_label(structure)
        elif self.symmetry_labeling == "SPGLib":
            data["symmetry_label"] = SPGLibSymmetry().get_symmetry_label(structure)
        else:
            raise ValueError(
                "Symmetry algorithm {} not implemented".format(self.symmetry_labeling)
            )
        if self.include_composition:
            data["composition"] = structure.composition.formula.replace(" ", "")
        return data

    def get_material_hash(self, structure: Structure) -> str:
        """Get hash for a given Pymatgen Structure

        Args:
            structure (Structure): Pymatgen Structure to hash

        Returns:
            str: hash
        """
        data = self.get_entalpic_materials_data(structure)
        return "_".join([str(v) for k, v in data.items()])
