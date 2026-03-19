# Structure of `final_augmented_dataset.pkl`

The file `final_augmented_dataset.pkl` contains a list of dictionary objects, where each dictionary represents a crystal structure sample with its associated properties and graph-based features.

**Data Type:** `list`
**Length:** 32955
**Element Type:** `dict`

## Dictionary Keys and Descriptions

Each element in the list is a dictionary with the following keys:

| Key | Type | Description | Sample/Shape |
| :--- | :--- | :--- | :--- |
| `id` | `int` | An integer identifier for the sample. | `8207` |
| `unique_id` | `str` | A unique string identifier (MD5 hash). | `'1ce14887a3cd255b9b9b200f3aa4afb1'` |
| `numbers` | `numpy.ndarray` | Atomic numbers of the atoms in the structure. | `(33,)`, dtype=`int32` |
| `positions` | `numpy.ndarray` | Cartesian coordinates of the atoms. | `(33, 3)` |
| `cell` | `numpy.ndarray` | Unit cell lattice vectors (3x3 matrix). | `(3, 3)` |
| `pbc` | `numpy.ndarray` | Periodic boundary conditions (boolean array). | `[True, True, True]` |
| `target` | `float` | The target value for prediction (likely formation energy). | `-0.34518` |
| `metadata` | `dict` | Metadata about the defect and host material. | See below |
| `edge_index` | `numpy.ndarray` | Indices of connected nodes (atoms) in the crystal graph. | `(2, N_edges)` |
| `edge_dist` | `numpy.ndarray` | Euclidean distances between connected atoms. | `(N_edges,)`, dtype=`float32` |
| `triplet_index` | `numpy.ndarray` | Indices of atom triplets for angular features. | `(N_triplets, 3)` |
| `angles` | `numpy.ndarray` | Bond angles for the triplets in radians. | `(N_triplets,)`, dtype=`float32` |

## Metadata Structure

The `metadata` dictionary contains specific details about the defect:

| Key | Type | Description | Sample |
| :--- | :--- | :--- | :--- |
| `formula` | `str` | Chemical formula of the defect structure. | `'Cr2I6_Cs_int3'` |
| `host` | `str` | Host material formula. | `'Cr2I6'` |
| `dopant` | `str` | Dopant element symbol. | `'Cs'` |
| `site` | `str` | Defect site identifier. | `'int3'` |
| `defecttype` | `str` | Type of defect (e.g., interstitial, substitution). | `'interstitial'` |
| `natoms` | `int` | Number of atoms in the structure. | `33` |

## Notes

- **Graph Features**: The keys `edge_index`, `edge_dist`, `triplet_index`, and `angles` suggest that the data has been pre-processed for use with Graph Neural Networks (GNNs), possibly including edge and triplet (angle) features.
- **Data Types**: Atomic numbers are integers, while positions, cell vectors, and graph features are floats (some specifically `float32`).
