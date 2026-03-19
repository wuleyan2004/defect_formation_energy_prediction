
import os
import sys
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Try to import ALIGNN and Jarvis
try:
    from jarvis.core.atoms import Atoms
    from jarvis.core.specie import atomic_numbers_to_symbols
    from alignn.graphs import Graph
    from alignn.models.alignn import ALIGNN, ALIGNNConfig
    import dgl
except ImportError:
    print("Error: ALIGNN, Jarvis-Tools, or DGL not installed.")
    # sys.exit(1) # Commented out to allow importing this module even if dependencies are missing

# --- Classes & Utils (Adapted from train_alignn.py) ---

class Normalizer:
    def __init__(self, tensor=None, mean=None, std=None):
        if tensor is not None:
            self.mean = torch.mean(tensor)
            self.std = torch.std(tensor)
        else:
            self.mean = mean
            self.std = std

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

class ALIGNNDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        symbols = atomic_numbers_to_symbols(sample['numbers'])
        atoms = Atoms(
            lattice_mat=sample['cell'],
            coords=sample['positions'],
            elements=symbols,
            cartesian=True
        )
        g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=8.0, max_neighbors=12, compute_line_graph=True)
        target = torch.tensor(sample['target'], dtype=torch.float32)
        return g, lg, target, sample

def collate_alignn(batch):
    gs, lgs, targets, samples = zip(*batch)
    batched_g = dgl.batch(gs)
    batched_lg = dgl.batch(lgs)
    targets = torch.stack(targets)
    return batched_g, batched_lg, targets

def get_base_id(uid):
    if '_rot_' in uid:
        return uid.split('_rot_')[0]
    if '_pert_' in uid:
        return uid.split('_pert_')[0]
    return uid

def is_original(sample):
    if 'metadata' in sample and 'augmented' in sample['metadata']:
        return False
    uid = sample.get('unique_id', '')
    if '_rot_' in uid or '_pert_' in uid:
        return False
    return True

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Replicate the splitting logic to ensure we test on the exact same test set.
    """
    print("Grouping samples by base ID...")
    groups = defaultdict(list)
    for idx, sample in enumerate(tqdm(dataset.data, desc="Indexing")):
        uid = sample['unique_id']
        base_id = get_base_id(uid)
        groups[base_id].append(idx)
        
    all_base_ids = list(groups.keys())
    
    random.seed(seed)
    random.shuffle(all_base_ids)
    
    n_total = len(all_base_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    test_ids = all_base_ids[n_train+n_val:]
    
    test_indices = []
    
    # Test: Add ONLY original samples
    for uid in test_ids:
        for idx in groups[uid]:
            if is_original(dataset.data[idx]):
                test_indices.append(idx)
                
    print(f"Test Set Size (Samples): {len(test_indices)} (Original only)")
    
    return Subset(dataset, test_indices)

def get_test_predictions():
    # --- Configuration ---
    CONFIG = {
        'data_path': '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/data/final_dataset.pkl',
        'checkpoint_path': '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/checkpoints/ALIGNN/best_model.pth',
        'batch_size': 32,
        'seed': 42
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Note: DGL on macOS often has issues with MPS backend, so we force CPU if CUDA is not available.
    # If you have a DGL version that supports MPS, you can uncomment the following lines:
    # if not torch.cuda.is_available() and torch.backends.mps.is_available():
    #     device = torch.device('mps')
        
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])

    # --- 1. Load Data ---
    print("Loading dataset...")
    full_dataset = ALIGNNDataset(CONFIG['data_path'])
    
    # --- 2. Reproduce Test Split ---
    print("Reproducing Test Split...")
    test_set = split_dataset(full_dataset, train_ratio=0.8, val_ratio=0.1, seed=CONFIG['seed'])
    
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_alignn)
    
    # --- 3. Load Checkpoint ---
    if not os.path.exists(CONFIG['checkpoint_path']):
        print(f"Error: Checkpoint not found at {CONFIG['checkpoint_path']}")
        return [], [], 0.0, 0.0
        
    print(f"Loading checkpoint from {CONFIG['checkpoint_path']}...")
    checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device)
    
    # Restore Normalizer
    norm_state = checkpoint['normalizer']
    if isinstance(norm_state, dict):
        normalizer = Normalizer(mean=norm_state['mean'], std=norm_state['std'])
    else:
        normalizer = norm_state
        
    print(f"Restored Normalizer: Mean={normalizer.mean:.4f}, Std={normalizer.std:.4f}")
    
    # --- Model ---
    # Using the same config as in train_alignn.py
    config = ALIGNNConfig(
        name="alignn",
        output_features=1,
        alignn_layers=4,
        gcn_layers=4,
        atom_input_features=92, 
        edge_input_features=80, 
        triplet_input_features=40, 
        embedding_features=64,
        hidden_features=256,
    )
    
    model = ALIGNN(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # --- 4. Evaluate ---
    print("Starting Evaluation...")
    all_preds = []
    all_targets = []
    
    test_mae_sum = 0
    test_mse_sum = 0
    
    with torch.no_grad():
        for g, lg, targets in tqdm(test_loader, desc="Testing"):
            g = g.to(device)
            lg = lg.to(device)
            targets = targets.to(device)
            
            preds = model((g, lg, None))
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = preds.squeeze(-1)
            
            # Denormalize
            preds_denorm = normalizer.denorm(preds)
            
            # Collect results
            all_preds.extend(preds_denorm.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Metrics
            abs_error = torch.abs(preds_denorm - targets)
            sq_error = (preds_denorm - targets) ** 2
            
            test_mae_sum += abs_error.sum().item()
            test_mse_sum += sq_error.sum().item()
            
            if device.type == 'mps':
                torch.mps.empty_cache()
                
    num_samples = len(test_set)
    mae = 0.0
    rmse = 0.0
    if num_samples > 0:
        mae = test_mae_sum / num_samples
        mse = test_mse_sum / num_samples
        rmse = np.sqrt(mse)
    else:
        print("Warning: Test set is empty!")
    
    print("\n------------------------------------------------")
    print(f"Test Results (Samples: {num_samples})")
    print(f"MAE:  {mae:.4f} eV")
    print(f"RMSE: {rmse:.4f} eV")
    print("------------------------------------------------")

    return all_preds, all_targets, mae, rmse

def main():
    all_preds, all_targets, mae, rmse = get_test_predictions()
    
    # --- 5. Visualization ---
    try:
        if len(all_targets) > 0:
            plt.figure(figsize=(6, 6))
            plt.scatter(all_targets, all_preds, alpha=0.5, s=10, c='blue', edgecolors='none')
            
            min_val = min(min(all_targets), min(all_preds))
            max_val = max(max(all_targets), max(all_preds))
            margin = (max_val - min_val) * 0.05
            
            plt.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 'k--', alpha=0.75, zorder=0)
            
            plt.xlabel('DFT Formation Energy (eV)')
            plt.ylabel('Predicted Formation Energy (eV)')
            plt.title(f'Test Set Parity Plot (ALIGNN)\nMAE={mae:.3f} eV, RMSE={rmse:.3f} eV')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.axis('square')
            
            # Use hardcoded path to avoid scope issues
            checkpoint_path = '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/checkpoints/ALIGNN/best_model.pth'
            output_dir = os.path.dirname(checkpoint_path)
            output_plot = os.path.join(output_dir, 'test_parity_plot_alignn.png')
            plt.savefig(output_plot, dpi=300, bbox_inches='tight')
            print(f"Parity plot saved to {output_plot}")
            
            # Save results to text file
            output_txt = os.path.join(output_dir, 'test_results.txt')
            with open(output_txt, 'w') as f:
                f.write(f"Test Results (Samples: {len(all_targets)})\n")
                f.write(f"MAE:  {mae:.4f} eV\n")
                f.write(f"RMSE: {rmse:.4f} eV\n")
            print(f"Results saved to {output_txt}")
            
        else:
            print("Skipping plot generation: No data points.")
        
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
