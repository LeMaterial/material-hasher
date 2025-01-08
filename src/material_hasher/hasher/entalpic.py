from pymatgen.analysis.local_env import EconNN, NearNeighbors

from material_hasher.hasher.utils.graph import get_weisfeiler_lehman_hash
from material_hasher.hasher.utils.graph_structure import get_structure_graph
from material_hasher.hasher.utils.symmetry import AFLOWSymmetry, SPGLibSymmetry
from pymatgen.core.structure import Structure
from typing import Optional

from material_hasher.hasher.base import HasherBase


def shorten_hash(hash):
    split = hash.split("_")
    return split[0] + "_" + split[2]


class EntalpicMaterialsHasher(HasherBase):
    def __init__(
        self,
        graphing_algorithm: str = "WL",
        bonding_algorithm: NearNeighbors = EconNN,
        bonding_kwargs: dict = {"tol": 0.2, "cutoff": 10, "use_fictive_radius": True},
        include_composition: bool = True,
        symmetry_labeling: str = "SPGLib",
        shorten_hash: bool = False,
    ):
        self.graphing_algorithm = graphing_algorithm
        self.bonding_algorithm = bonding_algorithm
        self.bonding_kwargs = bonding_kwargs
        self.include_composition = include_composition
        self.symmetry_labeling = symmetry_labeling
        self.shorten_hash = shorten_hash

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

    def get_material_hash(self, structure: Structure) -> str:
        """Returns a hash of the structure.

        Parameters
        ----------
        structure : Structure
            Structure to hash.

        Returns
        -------
        str
            Hash of the structure.
        """
        data = self.get_entalpic_materials_data(structure)
        return "_".join([str(v) for k, v in data.items()])

    def is_equivalent(
        self,
        structure1: Structure,
        structure2: Structure,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if two structures are similar based on the StructureMatcher of
        pymatgen. The StructureMatcher uses a similarity algorithm based on the
        maximum common subgraph isomorphism and the Jaccard index of the sites.

        Parameters
        ----------
        structure1 : Structure
            First structure to compare.
        structure2 : Structure
            Second structure to compare.

        Returns
        -------
        bool
            True if the two structures are similar, False otherwise.
        """

        hash_structure1 = self.get_material_hash(structure1)
        hash_structure2 = self.get_material_hash(structure2)

        if self.shorten_hash:
            hash_structure1 = shorten_hash(hash_structure1)
            hash_structure2 = shorten_hash(hash_structure2)

        return hash_structure1 == hash_structure2
