# Defect Formation Energy Dataset Structure

## 1. Dataset Overview
- **File Path**: `data/cleaned_dataset.pkl`
- **Total Samples**: 10,985
- **Data Type**: Python List of Dictionaries (`List[Dict]`)
- **Description**: This dataset contains structural information and formation energies for various point defects in 2D materials.

## 2. Sample Structure
Each sample in the list is a dictionary containing the following keys:

| Key | Type | Shape / Example | Description |
| :--- | :--- | :--- | :--- |
| `id` | `int` | `5` | Integer identifier for the sample. |
| `unique_id` | `str` | `'6d34bf...'` | Unique hash identifier for the defect structure. |
| `numbers` | `np.ndarray` | `(N_atoms,)` | Array of atomic numbers for each atom in the supercell. |
| `positions` | `np.ndarray` | `(N_atoms, 3)` | Cartesian coordinates of atoms (in Angstroms). |
| `cell` | `np.ndarray` | `(3, 3)` | Unit cell vectors (lattice matrix). |
| `pbc` | `np.ndarray` | `(3,)` | Periodic boundary conditions, typically `[True, True, True]`. |
| `target` | `float` | `4.801` | **Target Value**: Defect Formation Energy (eV). |
| `metadata` | `dict` | `{...}` | Dictionary containing chemical and structural metadata. |

## 3. Metadata Structure
The `metadata` dictionary provides detailed information about the defect chemistry:

| Key | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `formula` | `str` | `'S2Sn_Cl_int0'` | Chemical formula label for the defect system. |
| `host` | `str` | `'SnS2'` | Chemical formula of the host material. |
| `dopant` | `str` | `'Cl'` | Element symbol of the dopant/defect species. |
| `site` | `str` | `'int0'` | Label for the defect site (e.g., specific interstitial site). |
| `defecttype` | `str` | `'interstitial'` | Type of defect (e.g., `interstitial`, `vacancy`, `substitution`). |
| `natoms` | `int` | `28` | Total number of atoms in the supercell. |

## 4. Data Sample (JSON Representation)
Below is a simplified JSON representation of a single data sample:

```json
{
  "id": 5,
  "unique_id": "6d34bf467b0f0f89da56863bc5c1c46e",
  "numbers": [17, 16, 16, 16, 16, ...],  // Atomic numbers (e.g., Cl=17, S=16)
  "positions": [
    [-0.94, 1.47, 9.12],
    [5.31, 7.14, 10.81],
    [7.08, 4.06, 7.40],
    ...
  ],
  "cell": [
    [10.83, 0.0, 0.0],
    [-5.42, 9.38, 0.0],
    [0.0, 0.0, 15.0]
  ],
  "pbc": [true, true, true],
  "target": 4.801365,
  "metadata": {
    "formula": "S2Sn_Cl_int0",
    "host": "SnS2",
    "dopant": "Cl",
    "site": "int0",
    "defecttype": "interstitial",
    "natoms": 28
  }
}
```
