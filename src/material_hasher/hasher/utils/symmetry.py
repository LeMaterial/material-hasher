from shutil import which

from monty.tempfile import ScratchDir
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure


class SPGLibSymmetry:
    """
    Symmetry space group number using SPGLib via Pymatgen
    """

    def __init__(self, symprec: float = 0.01):
        """Class to set settings for Pymatgen's symmetry detection

        Args:
            symprec (float, optional): Symmetry precision tollerance.
              Defaults to 0.01.
        """
        self.symprec = symprec

    def get_symmetry_label(self, structure: Structure) -> int:
        """Get symmetry space group number from structure

        Args:
            structure (Structure): input structure

        Returns:
            int: space group number
        """
        sga = SpacegroupAnalyzer(structure, self.symprec)
        return sga.get_symmetry_dataset().number


class AFLOWSymmetry:
    """
    AFLOW prototype label using AFLOW libary
    """

    def __init__(self, aflow_executable: str = None):
        """AFLOW Symmetry

        Args:
            aflow_executable (str, optional): AFLOW executable. If none listed tries to find aflow in system path

        Raises:
            RuntimeError: If AFLOW is not found
        """
        from aflow_xtal_finder import XtalFinder

        self.aflow_executable = aflow_executable or which("aflow")

        print("aflow found in {}".format(self.aflow_executable))

        if self.aflow_executable is None:
            raise RuntimeError(
                "Requires aflow to be in the PATH or the absolute path to "
                f"the binary to be specified via {self.aflow_executable=}.\n"
            )

    def get_symmetry_label(self, structure: Structure, tolerance: float = 0.1) -> str:
        """
        Returns AFLOW label for a given structure
        Args:
            structure (Structure): structure to run AFLOW on
            tolerance (float, optional): AFLOW symmetry tolerance. Defaults to 0.1.

        Returns:
            str: AFLOW label
        """

        xtf = XtalFinder(self.aflow_executable)
        with ScratchDir("."):
            structure.to_file("POSCAR")
            data = xtf.get_prototype_label(
                [
                    "POSCAR",
                ],
                options="--tolerance={}".format(tolerance),
            )
        return data
