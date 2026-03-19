import pickle
import numpy as np
from ase import Atoms
from copy import deepcopy
import os
from tqdm import tqdm

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(data, file_path):
    print(f"Saving {len(data)} samples to {file_path}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print("Save complete.")

def dict_to_atoms(d):
    """Convert dictionary to ASE Atoms object."""
    # Ensure cell is 3x3 array
    cell = np.array(d['cell'])
    if cell.shape != (3, 3):
        # Handle cases where cell might be flattened or different shape if any
        cell = cell.reshape(3, 3)
        
    return Atoms(numbers=d['numbers'],
                 positions=d['positions'],
                 cell=cell,
                 pbc=d['pbc'])

def atoms_to_dict(atoms, original_dict):
    """Update dictionary with new atoms positions and cell."""
    new_dict = deepcopy(original_dict)
    new_dict['positions'] = atoms.get_positions()
    new_dict['cell'] = np.array(atoms.get_cell())
    return new_dict

def augment_rotation(data, num_augmentations=1):
    """
    Augment data by rotating the crystal structure.
    """
    augmented_data = []
    print(f"Applying Rotation Augmentation (x{num_augmentations})...")
    
    for entry in tqdm(data):
        try:
            atoms = dict_to_atoms(entry)
            
            for _ in range(num_augmentations):
                atoms_copy = atoms.copy()
                # Random rotation axis and angle
                axis = np.random.rand(3) - 0.5
                angle = np.random.uniform(0, 360)
                # Rotate structure and cell
                atoms_copy.rotate(angle, v=axis, rotate_cell=True)
                
                new_entry = atoms_to_dict(atoms_copy, entry)
                
                # Update metadata
                if 'metadata' not in new_entry:
                    new_entry['metadata'] = {}
                new_entry['metadata']['augmented'] = 'rotation'
                
                # Create a new unique ID to avoid duplicates if ID is used as key
                if 'unique_id' in new_entry:
                    new_entry['unique_id'] = f"{new_entry['unique_id']}_rot_{_}"
                
                augmented_data.append(new_entry)
        except Exception as e:
            print(f"Skipping entry due to error: {e}")
            continue
            
    return augmented_data

def augment_perturbation(data, sigma=0.01, num_augmentations=1):
    """
    Augment data by adding small Gaussian noise to atomic positions.
    sigma: Standard deviation of noise in Angstroms.
    """
    augmented_data = []
    print(f"Applying Perturbation Augmentation (x{num_augmentations}, sigma={sigma})...")
    
    for entry in tqdm(data):
        try:
            atoms = dict_to_atoms(entry)
            
            for _ in range(num_augmentations):
                atoms_copy = atoms.copy()
                positions = atoms_copy.get_positions()
                noise = np.random.normal(0, sigma, positions.shape)
                atoms_copy.set_positions(positions + noise)
                
                new_entry = atoms_to_dict(atoms_copy, entry)
                
                if 'metadata' not in new_entry:
                    new_entry['metadata'] = {}
                new_entry['metadata']['augmented'] = 'perturbation'
                
                if 'unique_id' in new_entry:
                    new_entry['unique_id'] = f"{new_entry['unique_id']}_pert_{_}"
                
                augmented_data.append(new_entry)
        except Exception as e:
            print(f"Skipping entry due to error: {e}")
            continue
            
    return augmented_data

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/data/cleaned_dataset.pkl'
    OUTPUT_FILE = '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/data/augmented_dataset.pkl'
    
    # Load original data
    original_data = load_data(INPUT_FILE)
    print(f"Original dataset size: {len(original_data)}")
    
    # Apply augmentations
    # 1. Rotation
    rotated_data = augment_rotation(original_data, num_augmentations=1)
    
    # 2. Perturbation (small noise)
    perturbed_data = augment_perturbation(original_data, sigma=0.02, num_augmentations=1)
    
    # Combine
    final_dataset = original_data + rotated_data + perturbed_data
    
    # Shuffle
    np.random.shuffle(final_dataset)
    
    # Save
    print(f"Final dataset size: {len(final_dataset)}")
    save_data(final_dataset, OUTPUT_FILE)
