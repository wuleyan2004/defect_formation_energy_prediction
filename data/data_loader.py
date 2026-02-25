import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os

class CrystalGraphDataset(Dataset):
    def __init__(self, data_path, feature_path, device='cpu'):
        """
        Args:
            data_path (str): Path to the processed dataset pickle file (with graphs).
            feature_path (str): Path to the atom_features.pth file.
            device (str): Device to put tensors on ('cpu' or 'cuda').
        """
        self.device = device
        
        # 1. Load Dataset
        # Note: We use the processed dataset which contains graph information
        # If the user specified 'cleaned_dataset.pkl' but meant the one with graphs,
        # we try to locate the correct one or fallback.
        real_path = data_path
        if 'cleaned_dataset.pkl' in data_path:
             # Check if the graph version exists, as cleaned_dataset usually doesn't have edges
             graph_path = data_path.replace('cleaned_dataset.pkl', 'processed_dataset_with_graphs.pkl')
             if os.path.exists(graph_path):
                 print(f"Redirecting to {graph_path} which contains graph data.")
                 real_path = graph_path
        
        if not os.path.exists(real_path):
            raise FileNotFoundError(f"Dataset not found at {real_path}")
            
        print(f"Loading dataset from {real_path}...")
        with open(real_path, 'rb') as f:
            self.data = pickle.load(f)
            
        # 2. Load Atom Features (Lookup Table)
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Atom features not found at {feature_path}")
            
        print(f"Loading atom features from {feature_path}...")
        self.atom_features = torch.load(feature_path, map_location='cpu').to(self.device) # (101, 9)
        
    def __len__(self):
        return len(self.data)
    
    def compute_pbc_distance_matrix(self, positions, cell):
        """
        Compute N*N distance matrix with PBC using Minimum Image Convention (MIC).
        
        Args:
            positions: (N, 3) tensor
            cell: (3, 3) tensor, row vectors (a, b, c)
        Returns:
            dist_matrix: (N, N) tensor
        """
        # 1. Compute pairwise difference vectors (Cartesian)
        # diff[i, j] = r_i - r_j
        # Shape: (N, N, 3)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # 2. Convert to fractional coordinates
        # r_cart = r_frac @ cell  =>  r_frac = r_cart @ cell^-1
        # Note: If cell rows are lattice vectors, then X_frac = X_cart @ inv(cell)
        try:
            cell_inv = torch.linalg.inv(cell)
        except RuntimeError:
            # Handle singular matrix (e.g. if cell is all zeros or degenerate)
            return torch.norm(diff, dim=-1)
            
        diff_frac = diff @ cell_inv
        
        # 3. Apply MIC: shift fractional coordinates to [-0.5, 0.5]
        diff_frac_mic = diff_frac - torch.round(diff_frac)
        
        # 4. Convert back to Cartesian
        diff_cart_mic = diff_frac_mic @ cell
        
        # 5. Compute Euclidean norm
        dist_matrix = torch.norm(diff_cart_mic, dim=-1)
        
        return dist_matrix

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # --- 1. Atom Features ---
        # Get atomic numbers (integers)
        numbers = torch.tensor(sample['numbers'], dtype=torch.long, device=self.device)
        # Lookup features from the table
        # atom_features is (101, 9), numbers are indices 0-100
        x = self.atom_features[numbers] # (N, 9)
        
        # --- 2. Graph Data ---
        # edge_index: (2, E) -> LongTensor
        edge_index = torch.tensor(sample['edge_index'], dtype=torch.long, device=self.device)
        
        # edge_dist: (E,) -> FloatTensor
        edge_dist = torch.tensor(sample['edge_dist'], dtype=torch.float32, device=self.device)
        
        # triplet_index: (M, 3) -> LongTensor
        triplet_index = torch.tensor(sample['triplet_index'], dtype=torch.long, device=self.device)
        
        # angles: (M,) -> FloatTensor
        angles = torch.tensor(sample['angles'], dtype=torch.float32, device=self.device)
        
        # --- 3. Distance Matrix (PBC) ---
        positions = torch.tensor(sample['positions'], dtype=torch.float32, device=self.device)
        cell = torch.tensor(sample['cell'], dtype=torch.float32, device=self.device)
        
        dist_matrix = self.compute_pbc_distance_matrix(positions, cell) # (N, N)
        
        # --- 4. Target ---
        target = torch.tensor(sample['target'], dtype=torch.float32, device=self.device)
        
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_dist': edge_dist,
            'triplet_index': triplet_index,
            'angles': angles,
            'dist_matrix': dist_matrix,
            'target': target,
            # Optional: keep original pos and cell if needed
            'pos': positions,
            'cell': cell,
            'num_atoms': len(numbers)
        }

def collate_fn(batch):
    """
    Custom collate function to batch graph data with padding.
    
    Args:
        batch: List of dictionaries returned by __getitem__
        
    Returns:
        batch_dict: Dictionary containing batched tensors
    """
    # 1. Find max number of atoms in this batch
    num_atoms_list = [sample['num_atoms'] for sample in batch]
    max_num_atoms = max(num_atoms_list)
    batch_size = len(batch)
    
    # 2. Prepare batched tensors
    # x: (Batch, N_max, 9)
    # dist_matrix: (Batch, N_max, N_max)
    # mask: (Batch, N_max) -> True for real atoms, False for padding
    # target: (Batch,)
    
    # Get feature dimension from first sample
    feature_dim = batch[0]['x'].shape[1]
    device = batch[0]['x'].device
    
    batched_x = torch.zeros((batch_size, max_num_atoms, feature_dim), device=device)
    batched_dist = torch.zeros((batch_size, max_num_atoms, max_num_atoms), device=device)
    atom_mask = torch.zeros((batch_size, max_num_atoms), dtype=torch.bool, device=device)
    batched_target = torch.zeros((batch_size,), device=device)
    
    # Lists for variable-size graph data (PyG style usually keeps these as lists or concatenated)
    # Here we just keep them as lists since they are hard to pad into a single tensor efficiently
    # without PyG's Batch object.
    edge_indices = []
    edge_dists = []
    triplet_indices = []
    angles = []
    
    for i, sample in enumerate(batch):
        n = sample['num_atoms']
        
        # Fill x
        batched_x[i, :n, :] = sample['x']
        
        # Fill dist_matrix
        batched_dist[i, :n, :n] = sample['dist_matrix']
        
        # Fill mask (True = Valid atom)
        atom_mask[i, :n] = True
        
        # Fill target
        batched_target[i] = sample['target']
        
        # Collect graph data (offset indices for concatenation if needed later, but here just raw)
        edge_indices.append(sample['edge_index'])
        edge_dists.append(sample['edge_dist'])
        triplet_indices.append(sample['triplet_index'])
        angles.append(sample['angles'])
        
    return {
        'x': batched_x,
        'dist_matrix': batched_dist,
        'atom_mask': atom_mask,
        'target': batched_target,
        'num_atoms': torch.tensor(num_atoms_list, device=device),
        # Graph data kept as lists
        'edge_index_list': edge_indices,
        'edge_dist_list': edge_dists,
        'triplet_index_list': triplet_indices,
        'angles_list': angles
    }

if __name__ == "__main__":
    # Test the DataLoader
    DATA_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/cleaned_dataset.pkl'
    FEATURE_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/atom_features.pth'
    
    try:
        dataset = CrystalGraphDataset(DATA_PATH, FEATURE_PATH)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        
        print("DataLoader initialized successfully.")
        print(f"Dataset size: {len(dataset)}")
        
        for batch in loader:
            print("\n--- Batch 0 ---")
            # Batch is now a dict
            print(f"Batched X shape: {batch['x'].shape}")
            print(f"Batched Dist Matrix shape: {batch['dist_matrix'].shape}")
            print(f"Atom Mask shape: {batch['atom_mask'].shape}")
            print(f"Batched Target shape: {batch['target'].shape}")
            print(f"Num atoms list: {batch['num_atoms']}")
            
            # Check mask correctness
            first_n = batch['num_atoms'][0]
            print(f"Sample 0 Mask sum (should be {first_n}): {batch['atom_mask'][0].sum()}")
            
            break
            
    except Exception as e:
        print(f"Error: {e}")
