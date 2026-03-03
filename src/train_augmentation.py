
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import sys
import os
import time
import numpy as np
from tqdm import tqdm
import gc
from collections import defaultdict
import random

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

import logging
import datetime

# --- Logger Setup ---
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log file name based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class Normalizer:
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

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
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Train: Add ALL samples (original + augmented)
    for uid in train_ids:
        train_indices.extend(groups[uid])
        
    # Val: Add ONLY original samples
    for uid in val_ids:
        for idx in groups[uid]:
            if is_original(dataset.data[idx]):
                val_indices.append(idx)
                
    # Test: Add ONLY original samples
    for uid in test_ids:
        for idx in groups[uid]:
            if is_original(dataset.data[idx]):
                test_indices.append(idx)
                
    print(f"Final Dataset Sizes (Samples):")
    print(f"  Train: {len(train_indices)} (Original + Augmented)")
    print(f"  Val:   {len(val_indices)} (Original only)")
    print(f"  Test:  {len(test_indices)} (Original only)")
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

def main():
    # --- Hyperparameters ---
    CONFIG = {
        'data_path': '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/data/final_augmented_dataset.pkl',
        'feature_path': '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/data/atom_features.pth',
        'output_dir': '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/checkpoints/augmentation',
        'batch_size': 32,      
        'epochs': 50,
        'lr': 1e-4,            
        'hidden_dim': 64,      
        'n_local': 2,
        'n_global': 1,
        'seed': 42,
        'resume_path': '/Users/wuleyan/Desktop/大创_我自己的代码保留一份/Defect_Formation_Energy_Prediction/checkpoints/augmentation/latest_model.pth'
    }
    
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
    
    # Initialize Logger
    logger = setup_logger(CONFIG['output_dir'])
    logger.info(f"Training started. Config: {CONFIG}")
        
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])
    
    if device.type == 'mps':
        torch.mps.empty_cache()

    # --- 1. Load Data ---
    logger.info("Loading dataset...")
    full_dataset = CrystalGraphDataset(CONFIG['data_path'], CONFIG['feature_path'], device=device)
    
    # Calculate Normalizer based on WHOLE dataset target statistics (or just train, but here we use all for simplicity as in original)
    all_targets = [sample['target'] for sample in full_dataset.data]
    
    # Filter out extreme outliers that might cause NaN/Inf in normalization
    valid_targets = [t for t in all_targets if not (np.isnan(t) or np.isinf(t) or abs(t) > 1e6)]
    if len(valid_targets) < len(all_targets):
        logger.warning(f"Warning: Removed {len(all_targets) - len(valid_targets)} invalid targets from normalizer calculation.")
        
    target_tensor = torch.tensor(valid_targets, dtype=torch.float32, device=device)
    normalizer = Normalizer(target_tensor)
    # --- 在 main() 函数中，normalizer = Normalizer(target_tensor) 下方 ---s
    
    logger.info(f"Target Norm Stats: Mean={normalizer.mean:.4f}, Std={normalizer.std:.4f}")
    
    # --- Custom Split ---
    train_set, val_set, test_set = split_dataset(full_dataset, train_ratio=0.8, val_ratio=0.1, seed=CONFIG['seed'])
    logger.info(f"Dataset split completed.")
    
    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    # Test loader can be added if needed, but usually we just train/val here
    
    # --- 2. Initialize Model & Optimizer ---
    model = CrystalTransformer(
        atom_fea_len=9,
        hidden_dim=CONFIG['hidden_dim'],
        n_local_layers=CONFIG['n_local'],
        n_global_layers=CONFIG['n_global']
    ).to(device)
    logger.info(f"Model initialized: Hidden Dim={CONFIG['hidden_dim']}, Local Layers={CONFIG['n_local']}, Global Layers={CONFIG['n_global']}")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Changed to Huber Loss
    criterion = nn.HuberLoss(delta=1.0)
    
    start_epoch = 0
    best_val_mae = float('inf')

    # --- 3. Resume Training ---
    if CONFIG['resume_path'] and os.path.exists(CONFIG['resume_path']):
        logger.info(f"Resuming from checkpoint: {CONFIG['resume_path']}")
        checkpoint = torch.load(CONFIG['resume_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mae = checkpoint.get('val_mae', float('inf'))
        logger.info(f"Restarting from Epoch {start_epoch+1}, Best Val MAE was: {best_val_mae:.4f}")

    logger.info("Start Training...")
    
    for epoch in range(start_epoch, CONFIG['epochs']):
        start_time = time.time()
        
        # --- Train ---
        model.train()
        train_loss_sum = 0
        train_mae_sum = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", unit="batch")
        
        for batch in train_pbar:
            optimizer.zero_grad()
            preds = model(batch)
            targets = batch['target']
            
            targets_norm = normalizer.norm(targets)
            loss = criterion(preds, targets_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if device.type == 'mps':
                torch.mps.empty_cache()
            
            batch_loss = loss.item()
            train_loss_sum += batch_loss * targets.size(0)
            # MAE Calculation Fix
            # Model output 'preds' is compared against 'targets_norm' in loss function,
            # so 'preds' is in normalized scale.
            # We want to report MAE in original eV scale.
            
            # 'preds' -> Normalized prediction
            # 'targets' -> Original ground truth (eV)
            
            with torch.no_grad():
                preds_denorm = normalizer.denorm(preds)
                # Compute absolute error in eV
                abs_error = torch.abs(preds_denorm - targets)
                
                # Check for NaNs or Infs which could mess up the sum
                if torch.isnan(abs_error).any() or torch.isinf(abs_error).any():
                    logger.warning(f"Warning: NaN/Inf detected in batch MAE calculation!")
                    # Replace with zeros or handle gracefully? For now just clamp.
                    abs_error = torch.nan_to_num(abs_error, nan=0.0, posinf=0.0, neginf=0.0)
                
                mae = abs_error.mean().item()
                train_mae_sum += mae * targets.size(0)
            
            train_pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'mae': f'{mae:.4f}'})
        
        avg_train_loss = train_loss_sum / len(train_set)
        avg_train_mae = train_mae_sum / len(train_set)
        
        # --- Validation ---
        model.eval()
        val_loss_sum = 0
        val_mae_sum = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]", unit="batch")
        
        with torch.no_grad():
            for batch in val_pbar:
                preds = model(batch)
                targets = batch['target']
                targets_norm = normalizer.norm(targets)
                loss = criterion(preds, targets_norm)
                val_loss_sum += loss.item() * targets.size(0)
                
                preds_denorm = normalizer.denorm(preds)
                # Compute absolute error
                abs_error = torch.abs(preds_denorm - targets)
                
                # Check for NaNs
                if torch.isnan(abs_error).any():
                     logger.warning(f"Warning: NaN detected in val batch!")
                     abs_error = torch.nan_to_num(abs_error, nan=0.0)
                
                mae = abs_error.mean().item()
                val_mae_sum += mae * targets.size(0)
                
                if device.type == 'mps':
                    torch.mps.empty_cache()
        
        avg_val_loss = val_loss_sum / len(val_set)
        avg_val_mae = val_mae_sum / len(val_set)
        
        scheduler.step(avg_val_mae)
        
        # --- Save Checkpoints ---
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'normalizer': {'mean': normalizer.mean, 'std': normalizer.std},
            'config': CONFIG,
            'val_mae': avg_val_mae
        }
        
        latest_path = os.path.join(CONFIG['output_dir'], 'latest_model.pth')
        torch.save(checkpoint_data, latest_path)

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_path = os.path.join(CONFIG['output_dir'], 'best_model.pth')
            torch.save(checkpoint_data, best_path)
            saved_msg = "(*)"
        else:
            saved_msg = ""
            
        gc.collect()
            
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} MAE: {avg_train_mae:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} MAE: {avg_val_mae:.4f} | "
              f"Time: {epoch_time:.1f}s {saved_msg}")

        # 💡 如果需要配合 runner.sh 进行每轮重启（用于清理顽固显存），请取消下面两行的注释
        print("🚩 这一轮跑完啦，我要自杀重启来清理内存了...")
        break  # 强制跳出循环，结束 Python 进程

    logger.info("Training Complete.")
    logger.info(f"Best Validation MAE: {best_val_mae:.4f} eV")

if __name__ == "__main__":
    main()
