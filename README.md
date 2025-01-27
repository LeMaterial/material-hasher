# material-hasher
material-hasher provide access to comprehensive benchmark for material-fingerprinting and similarity methods, as well as implementation of fingerprint and similarity methods from the community.

## Benchmarks
In this release, we include the following benchmarks:
- Transformations
- - Noise on atomistic coordinates `from material_hasher.benchmark.transformations import get_new_structure_with_gaussian_noise`
- - Noise on lattice vectors `from material_hasher.benchmark.transformations import get_new_structure_with_strain`
- - Isometric strain on lattice `from material_hasher.benchmark.transformations import get_new_structure_with_isometric_strain`
- - Translations `from material_hasher.benchmark.transformations import get_new_structure_with_translation`
- - Application of symmetry operations `from material_hasher.benchmark.transformations import get_new_structure_with_symm_ops`
- Disordered materials
- - Includes comprehensive test cases of structures generated via Supercell program and from Supercell paper to test whether various fingerprint or similarity metrics recognize disordered materials `from material_hasher.benchmark.run_disordered import benchmark_disordered_structures`

## Fingerprinting methods
We include the following fingerprint methods:
- a Structure graph, hashed via Weisfeiller-Lehman with and without symmetry labeling from SPGLib and composition `from material_hasher.hasher.entalpic import EntalpicMaterialsHasher`
- SLICES `from material_hasher.hasher.slices import SLICESHasher`
- PointwiseDistanceDistributionHasher `from material_hasher.hasher.pdd import PointwiseDistanceDistributionHasher`

## Similarity methods
We include the following structure similarity methods:
- Using GNN embeddings from trained and untrained EquiformerV2 `from material_hasher.eqv2 import EquiformerV2Similarity`
- Pymatgen's StructureMatcher `from material_hasher.similarity.structure_matchers import PymatgenStructureSimilarity`

## How to run benchmarks:
### Disordered benchmark
To test all the hasher and similarity methods on disordered materials dataset, seeing if each method can match the varying amount of disordered across a set of curated materials:
#### get help:
`python -m material_hasher.benchmark.run_disordered --help`

#### typical run (test disordered materials benchmark on all algorithms):
`python -m material_hasher.benchmark.run_disordered --algorithm all`

### Transformation benchmark
To test all the hasher and similarity methods on varying transformations applied to the structures across materials sampled from LeMat-Bulk:
#### get help:
`python -m material_hasher.benchmark.run_transformations --help`

#### typical run (test Entalpic fingerprint on all test cases for a single structure):
`python -m material_hasher.benchmark.run_transformations --algorithm Entalpic  --n-test-elements 1`

## How to utilize a fingerprint method:
Here is a sample code to get a hash result:
```
from pymatgen.core import Structure
import numpy as np
structure = Structure(np.eye(3)*3, ['Si'], [[0,0,0]])
from material_hasher.hasher.entalpic import EntalpicMaterialsHasher
emh = EntalpicMaterialsHasher()
print(emh.get_material_hash(structure))
```

## Installation
We utilize uv. Please have uv installed in your environment, and then run `uv sync`.
To utilize SLICES, please run: `uv pip install -r requirements_slices.txt`


## Citation
We are working on a pre-print describing our fingerprint method.

If your work makes use of the varying fingerprint methods, please consider citing:
SLICES:
```
@article{xiao2023invertible,
  title={An invertible, invariant crystal representation for inverse design of solid-state materials using generative deep learning},
  author={Xiao, Hang and Li, Rong and Shi, Xiaoyang and Chen, Yan and Zhu, Liangliang and Chen, Xi and Wang, Lei},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={7027},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
PDD: 
```
@article{widdowson2021pointwise,
  title={Pointwise distance distributions of periodic point sets},
  author={Widdowson, Daniel and Kurlin, Vitaliy},
  journal={arXiv preprint arXiv:2108.04798},
  year={2021}
}
```

If your work makes use of varying similarity methods, please consider citing:
Pymatgen:
```
@article{ong2013python,
  title={Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis},
  author={Ong, Shyue Ping and Richards, William Davidson and Jain, Anubhav and Hautier, Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and Chevrier, Vincent L and Persson, Kristin A and Ceder, Gerbrand},
  journal={Computational Materials Science},
  volume={68},
  pages={314--319},
  year={2013},
  publisher={Elsevier}
}
```
EquiformerV2
```
@article{liao2023equiformerv2,
  title={Equiformerv2: Improved equivariant transformer for scaling to higher-degree representations},
  author={Liao, Yi-Lun and Wood, Brandon and Das, Abhishek and Smidt, Tess},
  journal={arXiv preprint arXiv:2306.12059},
  year={2023}
}
```

