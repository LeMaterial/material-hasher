from pymatgen.analysis.local_env import EconNN, NearNeighbors

from material_hasher.hasher.utils.graph import get_weisfeiler_lehman_hash
from material_hasher.hasher.utils.graph_structure import get_structure_graph
from material_hasher.hasher.utils.symmetry import AFLOWSymmetry, SPGLibSymmetry


class EntalpicMaterialsHasher:
    def __init__(
        self,
        graphing_algorithm: str = "WL",
        bonding_algorithm: NearNeighbors = EconNN,
        bonding_kwargs: dict = {},
        include_composition: bool = False,
        symmetry_labeling: str = "SPGLib",
    ):
        self.graphing_algorithm = graphing_algorithm
        self.bonding_algorithm = bonding_algorithm
        self.bonding_kwargs = bonding_kwargs
        self.include_composition = include_composition
        self.symmetry_labeling = symmetry_labeling

    def get_entalpic_materials_data(self, structure):
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

    def get_material_hash(self, structure):
        data = self.get_entalpic_materials_data(structure)
        return "_".join([str(v) for k, v in data.items()])
