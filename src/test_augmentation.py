
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import sys
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import random
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Adjust imports based on directory structure
try:
    from data.data_loader import CrystalGraphDataset, collate_fn
except ImportError:
    # Fallback if running from a different context
    from WLY.data_loader import CrystalGraphDataset, collate_fn

try:
    from model import CrystalTransformer
except ImportError:
    from src.model import CrystalTransformer

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

def get_base_id(uid):
    """
    Extract the base unique_id from an augmented ID.
    Examples:
        'abc1234' -> 'abc1234'
        'abc1234_rot_0' -> 'abc1234'
        'abc1234_pert_0' -> 'abc1234'
    """
    if '_rot_' in uid:
        return uid.split('_rot_')[0]
    if '_pert_' in uid:
        return uid.split('_pert_')[0]
    return uid

def is_original(sample):
    """Check if a sample is original based on metadata or ID."""
    # Method 1: Check metadata (preferred if available)
    if 'metadata' in sample and 'augmented' in sample['metadata']:
        return False # It has an augmented tag
    
    # Method 2: Check ID pattern (backup)
    uid = sample.get('unique_id', '')
    if '_rot_' in uid or '_pert_' in uid:
        return False
        
    return True

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Custom split function:
    1. Group all samples by their original base ID.
    2. Split the *base IDs* into train/val/test.
    3. Train set: Includes Original + All Augmented versions of train IDs.
    4. Val/Test set: Includes ONLY Original versions of val/test IDs.
    """
    print("Grouping samples by base ID...")
    groups = defaultdict(list)
    for idx, sample in enumerate(tqdm(dataset.data, desc="Indexing")):
        uid = sample['unique_id']
        base_id = get_base_id(uid)
        groups[base_id].append(idx)
        
    all_base_ids = list(groups.keys())
    print(f"Total unique crystals (base IDs): {len(all_base_ids)}")
    
    # Shuffle base IDs
    random.seed(seed)
    random.shuffle(all_base_ids)
    
    # Calculate split sizes
    n_total = len(all_base_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    # n_test = rest
    
    train_ids = all_base_ids[:n_train]
    val_ids = all_base_ids[n_train:n_train+n_val]
    test_ids = all_base_ids[n_train+n_val:]
    
    print(f"Split sizes (Unique Crystals): Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Build final indices
    # We only need TEST indices here, but to keep logic consistent with training split, we generate all
    
    test_indices = []
    
    # Test: Add ONLY original samples
    for uid in test_ids:
        for idx in groups[uid]:
            if is_original(dataset.data[idx]):
                test_indices.append(idx)
                
    print(f"Test Set Size (Samples): {len(test_indices)} (Original only)")
    
    return Subset(dataset, test_indices)

def main():
    # --- Configuration ---
    CONFIG = {
        'data_path': '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/data/final_augmented_dataset.pkl',
        'feature_path': '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/data/atom_features.pth',
        'checkpoint_path': '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/checkpoints/augmentation/best_model.pth',
        'batch_size': 32,
        'seed': 42
    }
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed for reproducibility of the split
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])

    # --- 1. Load Data ---
    print("Loading dataset...")
    full_dataset = CrystalGraphDataset(CONFIG['data_path'], CONFIG['feature_path'], device=device)
    
    # --- 2. Reproduce Test Split ---
    print("Reproducing Test Split...")
    test_set = split_dataset(full_dataset, train_ratio=0.8, val_ratio=0.1, seed=CONFIG['seed'])
    
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # --- 3. Load Checkpoint ---
    if not os.path.exists(CONFIG['checkpoint_path']):
        raise FileNotFoundError(f"Checkpoint not found at {CONFIG['checkpoint_path']}")
        
    print(f"Loading checkpoint from {CONFIG['checkpoint_path']}...")
    checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device)
    
    # Restore Normalizer
    norm_state = checkpoint['normalizer']
    # Handle cases where norm_state is a dict or object
    if isinstance(norm_state, dict):
        normalizer = Normalizer(mean=norm_state['mean'], std=norm_state['std'])
    else:
        # If it was saved as an object
        normalizer = norm_state
        
    print(f"Restored Normalizer: Mean={normalizer.mean:.4f}, Std={normalizer.std:.4f}")
    
    # Restore Model
    # Get model config from checkpoint if available, else assume defaults or hardcode
    saved_config = checkpoint.get('config', {})
    hidden_dim = saved_config.get('hidden_dim', 64)
    n_local = saved_config.get('n_local', 2)
    n_global = saved_config.get('n_global', 1)
    
    print(f"Model Config: Hidden Dim={hidden_dim}, Local Layers={n_local}, Global Layers={n_global}")
    
    model = CrystalTransformer(
        atom_fea_len=9,
        hidden_dim=hidden_dim,
        n_local_layers=n_local,
        n_global_layers=n_global
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # --- 4. Evaluate ---
    print("Starting Evaluation...")
    all_preds = []
    all_targets = []
    
    test_mae_sum = 0
    test_mse_sum = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            preds = model(batch)
            targets = batch['target']
            
            # Denormalize predictions
            preds_denorm = normalizer.denorm(preds)
            
            # Collect results (move to CPU first)
            all_preds.extend(preds_denorm.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Compute metrics for this batch
            abs_error = torch.abs(preds_denorm - targets)
            sq_error = (preds_denorm - targets) ** 2
            
            # Handle potential NaNs (safety)
            if torch.isnan(abs_error).any():
                 abs_error = torch.nan_to_num(abs_error, nan=0.0)
            if torch.isnan(sq_error).any():
                 sq_error = torch.nan_to_num(sq_error, nan=0.0)
            
            test_mae_sum += abs_error.sum().item()
            test_mse_sum += sq_error.sum().item()
            
            if device.type == 'mps':
                torch.mps.empty_cache()
                
    num_samples = len(test_set)
    if num_samples > 0:
        mae = test_mae_sum / num_samples
        mse = test_mse_sum / num_samples
        rmse = np.sqrt(mse)
    else:
        mae = 0.0
        rmse = 0.0
        print("Warning: Test set is empty!")
    
    print("\n------------------------------------------------")
    print(f"Test Results (Samples: {num_samples})")
    print(f"MAE:  {mae:.4f} eV")
    print(f"RMSE: {rmse:.4f} eV")
    print("------------------------------------------------")
    
    # --- 5. Visualization (Optional) ---
    try:
        if len(all_targets) > 0:
            plt.figure(figsize=(6, 6))
            plt.scatter(all_targets, all_preds, alpha=0.5, s=10, c='blue', edgecolors='none')
            
            # Plot y=x line
            min_val = min(min(all_targets), min(all_preds))
            max_val = max(max(all_targets), max(all_preds))
            
            # Add some margin
            margin = (max_val - min_val) * 0.05
            plt.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 'k--', alpha=0.75, zorder=0)
            
            plt.xlabel('DFT Formation Energy (eV)')
            plt.ylabel('Predicted Formation Energy (eV)')
            plt.title(f'Test Set Parity Plot\nMAE={mae:.3f} eV, RMSE={rmse:.3f} eV')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.axis('square')
            
            output_plot = os.path.join(os.path.dirname(CONFIG['checkpoint_path']), 'test_parity_plot.png')
            plt.savefig(output_plot, dpi=300, bbox_inches='tight')
            print(f"Parity plot saved to {output_plot}")
        else:
            print("Skipping plot generation: No data points.")
        
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
