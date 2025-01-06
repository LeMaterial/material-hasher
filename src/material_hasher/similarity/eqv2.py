from pathlib import Path
from typing import Optional, Union

import ase
import numpy as np
import yaml
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from fairchem.core import OCPCalculator
from huggingface_hub import hf_hub_download
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from material_hasher.similarity.base import SimilarityMatcherBase

HF_MODEL_REPO_ID = "fairchem/OMAT24"
HF_MODEL_PATH = "eqV2_31M_omat_mp_salex.pt"


class EquiformerV2Embedder(SimilarityMatcherBase):
    """EquiformerV2 Embedder for structure similarity comparison.
    Designed for EquiformerV2 models trained on the OMAT24 dataset.
    These models can be found on the Hugging Face model hub at
    https://huggingface.co/fairchem/OMAT24

    Parameters
    ----------
    trained : bool
        Whether the model was trained or not
    cpu : bool
        Whether to use the cpu to run inference on or the gpu if one is found
    threshold : float, optional
        Threshold to determine similarity, by default 0.01
    n_relaxation_steps : int, optional
        Number of relaxation steps to perform on the atoms object before computing the embeddings of the atoms, by default 0 (no relaxations).
    model_path : Optional[str], optional
        Path to the model checkpoint if downloaded, by default None
    load_from_hf : bool, optional
        Whether to download the model from the Hugging Face model hub, by default True. Note that you need to have access to the model on the Hugging Face model hub to download it.
    """

    def __init__(
        self,
        trained: bool,
        cpu: bool,
        threshold: float = 0.01,
        n_relaxation_steps: int = 0,
        model_path: Optional[Union[str, Path]] = None,
        load_from_hf: bool = True,
    ):
        self.model_path = model_path
        self.load_from_hf = load_from_hf

        self.trained = trained
        self.cpu = cpu

        self.threshold = threshold
        self.n_relaxation_steps = n_relaxation_steps

        self.calc = None
        self.features = {}
        self.load_model()

    def load_model(self):
        if self.load_from_hf:
            try:
                self.model_path = hf_hub_download(
                    repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_PATH
                )
            except Exception as e:
                print(
                    f"Failed to download the model from the Hugging Face model hub: {e}"
                )

        if not self.trained:
            print("⚠️ Loading an untrained model because trained is set to False.")
            calc = OCPCalculator(checkpoint_path=self.model_path, cpu=self.cpu)
            config = calc.trainer.config

            config["dataset"] = {
                "train": {"src": "dummy"}
            }  # for compatibility with yaml loading

            yaml.dump(config, open("/tmp/config.yaml", "w"))
            self.calc = OCPCalculator(config_yml="/tmp/config.yaml", cpu=self.cpu)
        else:
            self.calc = OCPCalculator(checkpoint_path=self.model_path, cpu=self.cpu)

        self.add_model_hook()

    def add_model_hook(self):
        assert self.calc is not None, "Model not loaded"

        def hook_norm_block(m, input_embeddings, output_embeddings):
            self.features["sum_norm_embeddings"] = (
                output_embeddings.narrow(1, 0, 1)
                .squeeze(1)
                .sum(0)
                .detach()
                .cpu()
                .numpy()
            )

        self.calc.trainer.model.backbone.norm.register_forward_hook(hook_norm_block)

    def relax_atoms(self, atoms: ase.Atoms) -> ase.Atoms:
        """Relax the atoms using the FIRE optimizer
        WARNING: This function modifies the atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object to relax
        """
        atoms.calc = self.calc

        dyn = FIRE(FrechetCellFilter(atoms))
        dyn.run(steps=self.n_relaxation_steps)

        return atoms

    def get_structure_embeddings(self, structure: Structure) -> np.ndarray:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms = self.relax_atoms(atoms)

        return self.features["sum_norm_embeddings"]

    def get_similarity_score(
        self, structure1: Structure, structure2: Structure
    ) -> float:
        embeddings1 = self.get_structure_embeddings(structure1)
        embeddings2 = self.get_structure_embeddings(structure2)

        return np.linalg.norm(embeddings1 - embeddings2)

    def is_equivalent(
        self,
        structure1: Structure,
        structure2: Structure,
        threshold: Optional[float] = None,
    ) -> bool:
        score = self.get_similarity_score(structure1, structure2)

        if threshold is None:
            return score < 0.01
        else:
            return score < threshold
