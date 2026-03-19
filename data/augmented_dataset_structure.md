# Dataset Structure Analysis

## Dataset Overview

- **File Path**: `data/augmented_dataset.pkl`
- **Total Samples**: 32955
- **Data Format**: List of dictionaries (Python pickle)
- **Description**: This dataset contains augmented defect structures with their corresponding formation energies.

## Sample Structure

Each element in the list is a dictionary representing a single data sample with the following keys:

| Key | Type | Description | Shape/Example |
| :--- | :--- | :--- | :--- |
| `id` | `int` | Unique integer identifier for the sample. | `8207` |
| `unique_id` | `str` | Unique string identifier (hash). | `"1ce14887a3cd255b9b9b200f3aa4afb1"` |
| `numbers` | `numpy.ndarray` | Atomic numbers of the atoms in the structure. | `(natoms,)` e.g., `[24, 24, ...]` |
| `positions` | `numpy.ndarray` | Cartesian coordinates of the atoms (Angstroms). | `(natoms, 3)` |
| `cell` | `numpy.ndarray` | Unit cell vectors (3x3 matrix). | `(3, 3)` |
| `pbc` | `numpy.ndarray` | Periodic boundary conditions (True/False for each dimension). | `(3,)` e.g., `[True, True, True]` |
| `target` | `float` | The target property (Defect Formation Energy). | `-0.34518` |
| `metadata` | `dict` | Metadata describing the defect properties. | See below |

## Metadata Structure

The `metadata` dictionary contains descriptive information about the material and defect:

| Key | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `formula` | `str` | Full chemical formula of the defect structure. | `"Cr2I6_Cs_int3"` |
| `host` | `str` | Formula of the host material. | `"Cr2I6"` |
| `dopant` | `str` | Symbol of the dopant element. | `"Cs"` |
| `site` | `str` | Identifier for the defect site. | `"int3"` |
| `defecttype` | `str` | Type of the defect (e.g., interstitial, substitutional). | `"interstitial"` |
| `natoms` | `int` | Total number of atoms in the supercell. | `33` |
| `augmented` | `str` | **(Optional)** Type of augmentation applied. If missing, it is an original sample. | `"rotation"` or `"perturbation"` |

## Augmentation Details

The dataset is a combination of original samples and two types of augmentations, balanced equally:

| Augmentation Type | Description | Count |
| :--- | :--- | :--- |
| **Original** | No augmentation applied (missing `augmented` key in metadata). | 10,985 |
| **Rotation** | Structure randomly rotated (`metadata['augmented'] = 'rotation'`). | 10,985 |
| **Perturbation** | Small Gaussian noise added to atomic positions (`metadata['augmented'] = 'perturbation'`). | 10,985 |
| **Total** | | **32,955** |

## Data Samples

Below are examples of different sample types.

### 1. Original Sample

```json
{
    "id": 8207,
    "unique_id": "1ce14887a3cd255b9b9b200f3aa4afb1",
    "numbers": [
        24,
        24,
        24,
        24,
        24,
        "..."
    ],
    "positions": [
        [
            3.665796123999081,
            2.045935374759883,
            9.150174519806608
        ],
        [
            3.450959209049514,
            10.08445222200866,
            9.028176514527852
        ],
        "..."
    ],
    "cell": [
        [
            14.015888202758955,
            0.0,
            0.0
        ],
        [
            -7.007944101379479,
            12.138115240191876,
            0.0
        ],
        [
            0.0,
            0.0,
            18.063536576448442
        ]
    ],
    "pbc": [
        true,
        true,
        true
    ],
    "target": -0.3451828549999985,
    "metadata": {
        "formula": "Cr2I6_Cs_int3",
        "host": "Cr2I6",
        "dopant": "Cs",
        "site": "int3",
        "defecttype": "interstitial",
        "natoms": 33
    }
}
```

### 2. Rotation Augmented Sample

Note the `augmented: "rotation"` field in metadata and the rotated cell vectors.

```json
{
    "id": 14648,
    "unique_id": "e50f69e2864da577dc763fd45ea8384d_rot_0",
    "numbers": [
        16,
        16,
        16,
        16,
        16,
        "..."
    ],
    "positions": [
        [
            7.0921082920029885,
            -0.06595901296882811,
            -9.775202004700677
        ],
        [
            2.60479480905319,
            -0.6051309714198219,
            -13.754869287488884
        ],
        "..."
    ],
    "cell": [
        [
            -5.582019740951227,
            4.418237703260277,
            -8.447032728930408
        ],
        [
            8.775911576762415,
            5.25428688039156,
            4.172286673982726
        ],
        [
            10.634495376881436,
            -8.606936410991505,
            -11.529427813847821
        ]
    ],
    "pbc": [
        true,
        true,
        true
    ],
    "target": 2.303252160000003,
    "metadata": {
        "formula": "S2Zr_Y_int0",
        "host": "ZrS2",
        "dopant": "Y",
        "site": "int0",
        "defecttype": "interstitial",
        "natoms": 28,
        "augmented": "rotation"
    }
}
```

### 3. Perturbation Augmented Sample

Note the `augmented: "perturbation"` field in metadata.

```json
{
    "id": 11826,
    "unique_id": "64240f3ab8415ac6c332a0268bd8a334_pert_0",
    "numbers": [
        26,
        42,
        42,
        42,
        42,
        "..."
    ],
    "positions": [
        [
            1.617324758475152,
            0.9560997822955967,
            13.031609313311264
        ],
        [
            4.864904292861209,
            8.437337030486303,
            9.231353644996052
        ],
        "..."
    ],
    "cell": [
        [
            13.003400858389192,
            0.0,
            0.0
        ],
        [
            -6.501700429194598,
            11.261275478957417,
            0.0
        ],
        [
            0.0,
            0.0,
            18.232485982860293
        ]
    ],
    "pbc": [
        true,
        true,
        true
    ],
    "target": 4.338038520000039,
    "metadata": {
        "formula": "MoSSe_Fe_ads0",
        "host": "MoSSe",
        "dopant": "Fe",
        "site": "ads0",
        "defecttype": "adsorbate",
        "natoms": 49,
        "augmented": "perturbation"
    }
}
```
