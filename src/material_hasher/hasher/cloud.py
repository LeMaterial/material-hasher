import argparse
import glob
import json
import os
from multiprocessing import Pool
from time import perf_counter

import numpy as np
import pandas as pd
from pymatgen.core.structure import Composition, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

wyckoff_multiplicity_dict = json.load(
    open(os.path.join(os.path.dirname(__file__), "utils/wyckoff-cloud-data.json"))
)
gen_str = json.load(
    open(os.path.join(os.path.dirname(__file__), "utils/generator-cloud-data.json"))
)


class CloudHasher:
    """
    from: https://github.com/ChangwenXu98/CLOUD
    """

    def __init__(
        self,
        sga_kwargs=None,
    ):
        self.sga_kwargs = sga_kwargs or {}

    def get_material_hash(self, structure):
        analyzer = SpacegroupAnalyzer(structure, *self.sga_kwargs)
        symm_dataset = analyzer.get_symmetry_dataset()
        wyckoff_positions = symm_dataset["wyckoffs"]

        spg_num = str(analyzer.get_space_group_number())
        seq = " ".join(gen_str[spg_num])

        wyckoff_ls = []
        for i in range(len(wyckoff_positions)):
            multiplicity = wyckoff_multiplicity_dict[spg_num][wyckoff_positions[i]]
            wyckoff_symbol = multiplicity + wyckoff_positions[i]
            if wyckoff_symbol not in wyckoff_ls:
                wyckoff_ls.append(wyckoff_symbol)
        seq = seq + " | " + " ".join(wyckoff_ls)

        comp_ls = []
        for (
            element,
            ratio,
        ) in structure.composition.fractional_composition.get_el_amt_dict().items():
            ratio = str(np.round(ratio, 2))
            comp_ls.append(element)
            comp_ls.append(ratio)

        seq = seq + " | " + " ".join(comp_ls)

        return seq
