# Final Dataset Structure Analysis (`final_dataset.pkl`)

This document describes the structure and contents of the `final_dataset.pkl` file located in the `data/` directory.

## Overview

- **File Path:** `data/final_dataset.pkl`
- **File Format:** Python Pickle (`.pkl`)
- **Data Type:** List of Dictionaries
- **Total Samples:** 10,641

## Data Schema

Each element in the list is a dictionary representing a single data sample (likely a crystal structure with a defect). The dictionary contains the following keys:

| Key | Type | Shape / Structure | Description |
| :--- | :--- | :--- | :--- |
| `id` | `int` | Scalar | A numerical identifier for the sample (e.g., `5`). |
| `unique_id` | `str` | Scalar | A unique hash or string identifier (e.g., `'6d34bf467b0f0f89da56863bc5c1c46e'`). |
| `numbers` | `numpy.ndarray` | `(N_atoms,)` | Atomic numbers of the atoms in the structure (e.g., `[17, 16, ..., 50]`). |
| `positions` | `numpy.ndarray` | `(N_atoms, 3)` | Cartesian coordinates (x, y, z) of each atom. |
| `cell` | `numpy.ndarray` | `(3, 3)` | Unit cell lattice vectors. |
| `pbc` | `numpy.ndarray` | `(3,)` | Periodic boundary conditions (usually boolean or 0/1 indicating periodicity along x, y, z). |
| `target` | `float` | Scalar | The target value for prediction, likely the **Defect Formation Energy**. |
| `metadata` | `dict` | Dictionary | Contains descriptive metadata about the defect system. (See [Metadata Details](#metadata-details) below). |
| `edge_index` | `numpy.ndarray` | `(2, N_edges)` | Graph edge indices (adjacency list) for graph neural networks. Row 0 is source, Row 1 is target. |
| `edge_dist` | `numpy.ndarray` | `(N_edges,)` | Euclidean distances corresponding to each edge in `edge_index`. |
| `triplet_index` | `numpy.ndarray` | `(N_triplets, 3)` | Indices of atom triplets used to calculate bond angles (i-j-k). |
| `angles` | `numpy.ndarray` | `(N_triplets,)` | Bond angle values corresponding to the triplets. |

### Metadata Details

The `metadata` dictionary provides specific chemical and structural context:

- **`formula`** (`str`): The chemical formula of the specific defect structure (e.g., `'S2Sn_Cl_int0'`).
- **`host`** (`str`): The formula of the host material (e.g., `'SnS2'`).
- **`dopant`** (`str`): The dopant or impurity element (e.g., `'Cl'`).
- **`site`** (`str`): The specific site of the defect (e.g., `'int0'` for interstitial site 0).
- **`defecttype`** (`str`): The classification of the defect (e.g., `'interstitial'`, `'substitution'`).
- **`natoms`** (`int`): The total number of atoms in the supercell structure.

## Usage Example

To load and access the data in Python:

```python
import pickle

file_path = 'data/final_dataset.pkl'

with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

# Access the first sample
sample = dataset[0]

print(f"Sample ID: {sample['id']}")
print(f"Structure Formula: {sample['metadata']['formula']}")
print(f"Target Energy: {sample['target']}")
print(f"Number of Atoms: {len(sample['numbers'])}")
```
