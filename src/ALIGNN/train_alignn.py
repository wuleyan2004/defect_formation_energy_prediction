
import os
import sys
import pickle
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm import tqdm
import time
import datetime
import logging
from collections import defaultdict
import random
import gc

# Try to import ALIGNN and Jarvis
try:
    from jarvis.core.atoms import Atoms
    from jarvis.core.specie import atomic_numbers_to_symbols
    from alignn.graphs import Graph
    from alignn.models.alignn import ALIGNN, ALIGNNConfig
    import dgl
except ImportError:
    print("Error: ALIGNN, Jarvis-Tools, or DGL not installed.")
    print("Please install them using: pip install alignn jarvis-tools dgl")
    sys.exit(1)

# --- Logger Setup (Same as train_original.py) ---
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
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

# --- Dataset ---
class ALIGNNDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Convert atomic numbers to symbols
        # sample['numbers'] is a list of integers or numpy array
        # atomic_numbers_to_symbols expects an iterable (list/array), not a scalar
        symbols = atomic_numbers_to_symbols(sample['numbers'])
        
        # Create Jarvis Atoms
        atoms = Atoms(
            lattice_mat=sample['cell'],
            coords=sample['positions'],
            elements=symbols,
            cartesian=True
        )
        
        # Create ALIGNN Graph (Atom graph)
        # cutoff=8.0 is standard for ALIGNN, max_neighbors=12
        g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=8.0, max_neighbors=12, compute_line_graph=True)
        
        # Target
        target = torch.tensor(sample['target'], dtype=torch.float32)
        
        return g, lg, target, sample

def collate_alignn(batch):
    gs, lgs, targets, samples = zip(*batch)
    batched_g = dgl.batch(gs)
    batched_lg = dgl.batch(lgs)
    targets = torch.stack(targets)
    return batched_g, batched_lg, targets

# --- Split Logic (Consistent with train_original.py) ---
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
    print("Grouping samples by base ID...")
    groups = defaultdict(list)
    for idx, sample in enumerate(tqdm(dataset.data, desc="Indexing")):
        uid = sample['unique_id']
        base_id = get_base_id(uid)
        groups[base_id].append(idx)
        
    all_base_ids = list(groups.keys())
    print(f"Total unique crystals (base IDs): {len(all_base_ids)}")
    
    random.seed(seed)
    random.shuffle(all_base_ids)
    
    n_total = len(all_base_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    train_ids = all_base_ids[:n_train]
    val_ids = all_base_ids[n_train:n_train+n_val]
    test_ids = all_base_ids[n_train+n_val:]
    
    print(f"Split sizes (Unique Crystals): Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Train: Add ONLY original samples
    for uid in train_ids:
        for idx in groups[uid]:
            if is_original(dataset.data[idx]):
                train_indices.append(idx)
        
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
    print(f"  Train: {len(train_indices)} (Original only)")
    print(f"  Val:   {len(val_indices)} (Original only)")
    print(f"  Test:  {len(test_indices)} (Original only)")
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _resolve_path(project_root: Path, p: str) -> str:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return str(path)

def parse_args():
    project_root = _default_project_root()
    parser = argparse.ArgumentParser(description="Train ALIGNN on defect formation energy dataset.")
    parser.add_argument("--data-path", default="data/final_dataset.pkl")
    parser.add_argument("--output-dir", default="checkpoints/ALIGNN")
    parser.add_argument("--resume-path", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--epochs-per-run", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    args.data_path = _resolve_path(project_root, args.data_path)
    args.output_dir = _resolve_path(project_root, args.output_dir)
    if args.resume_path:
        args.resume_path = _resolve_path(project_root, args.resume_path)
    else:
        args.resume_path = os.path.join(args.output_dir, "latest_model.pth")

    if args.epochs_per_run < 1:
        raise SystemExit("--epochs-per-run must be >= 1")
    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit("--train-ratio must be in (0, 1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise SystemExit("--val-ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio must be < 1")

    return args

# --- Main ---
def main():
    args = parse_args()
    CONFIG = {
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'epochs_per_run': args.epochs_per_run,
        'lr': args.lr,
        'seed': args.seed,
        'resume_path': args.resume_path,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'num_workers': args.num_workers,
    }
    
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
    
    logger = setup_logger(CONFIG['output_dir'])
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Training started. Config: {CONFIG}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])

    # --- Load Data ---
    logger.info("Loading dataset...")
    full_dataset = ALIGNNDataset(CONFIG['data_path'])
    
    # Normalizer
    all_targets = [sample['target'] for sample in full_dataset.data]
    valid_targets = [t for t in all_targets if not (np.isnan(t) or np.isinf(t) or abs(t) > 1e6)]
    target_tensor = torch.tensor(valid_targets, dtype=torch.float32)
    normalizer = Normalizer(target_tensor)
    logger.info(f"Target Norm Stats: Mean={normalizer.mean:.4f}, Std={normalizer.std:.4f}")
    
    # Split
    train_set, val_set, test_set = split_dataset(
        full_dataset,
        train_ratio=CONFIG['train_ratio'],
        val_ratio=CONFIG['val_ratio'],
        seed=CONFIG['seed'],
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_alignn,
        num_workers=CONFIG['num_workers'],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_alignn,
        num_workers=CONFIG['num_workers'],
    )
    
    # --- Model ---
    # ALIGNN Config
    config = ALIGNNConfig(
        name="alignn",
        output_features=1,
        # Default ALIGNN params
        alignn_layers=4,
        gcn_layers=4,
        atom_input_features=92, 
        edge_input_features=80, 
        triplet_input_features=40, 
        embedding_features=64,
        hidden_features=256,
        # Link to dataset features if needed? 
        # ALIGNN uses standard atomic embeddings usually.
    )
    model = ALIGNN(config).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss() # ALIGNN paper often uses MSE or MAE. Original script used Huber. Let's use Huber or MSE.
    # To match original script, use Huber? Or stick to ALIGNN defaults?
    # Original used HuberLoss(delta=1.0). Let's use Huber for consistency in loss scale.
    criterion = nn.HuberLoss(delta=1.0)

    start_epoch = 0
    best_val_mae = float('inf')
    
    # --- Resume ---
    if CONFIG['resume_path'] and os.path.exists(CONFIG['resume_path']):
        logger.info(f"Resuming from checkpoint: {CONFIG['resume_path']}")
        checkpoint = torch.load(CONFIG['resume_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mae = checkpoint.get('val_mae', float('inf'))
        logger.info(f"Restarting from Epoch {start_epoch+1}, Best Val MAE was: {best_val_mae:.4f}")

    # --- Training Loop ---
    logger.info("Start Training...")
    
    end_epoch = min(CONFIG['epochs'], start_epoch + CONFIG['epochs_per_run'])
    if start_epoch >= CONFIG['epochs']:
        logger.info("Already finished all epochs. Exiting.")
        return

    for epoch in range(start_epoch, end_epoch):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss_sum = 0
        train_mae_sum = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", unit="batch")
        
        for step, (g, lg, targets) in enumerate(train_pbar):
            g = g.to(device)
            lg = lg.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # ALIGNN forward(g, lg) returns tensor (N, 1)
            preds = model((g, lg, None))
            if isinstance(preds, tuple):
                preds = preds[0] # Some versions might return tuple
            preds = preds.squeeze(-1) # (B,)
            
            # Normalize targets
            targets_norm = normalizer.norm(targets)
            
            loss = criterion(preds, targets_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_sum += loss.item() * targets.size(0)
            
            with torch.no_grad():
                preds_denorm = normalizer.denorm(preds)
                mae = torch.abs(preds_denorm - targets).mean().item()
                train_mae_sum += mae * targets.size(0)
                
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{mae:.4f}'})
    

        denom_train = min(len(train_set), CONFIG['batch_size'] * (step + 1))
        avg_train_loss = train_loss_sum / max(1, denom_train)
        avg_train_mae = train_mae_sum / max(1, denom_train)
        
        # Validation
        model.eval()
        val_loss_sum = 0
        val_mae_sum = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]", unit="batch")
        
        with torch.no_grad():
            for step_v, (g, lg, targets) in enumerate(val_pbar):
                g = g.to(device)
                lg = lg.to(device)
                targets = targets.to(device)
                
                preds = model((g, lg, None))
                if isinstance(preds, tuple):
                    preds = preds[0]
                preds = preds.squeeze(-1)
                
                targets_norm = normalizer.norm(targets)
                loss = criterion(preds, targets_norm)
                
                val_loss_sum += loss.item() * targets.size(0)
                
                preds_denorm = normalizer.denorm(preds)
                mae = torch.abs(preds_denorm - targets).mean().item()
                val_mae_sum += mae * targets.size(0)
                
        
        denom_val = min(len(val_set), CONFIG['batch_size'] * (step_v + 1))
        avg_val_loss = val_loss_sum / max(1, denom_val)
        avg_val_mae = val_mae_sum / max(1, denom_val)
        
        scheduler.step(avg_val_mae)
        
        # Save
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
        
        saved_msg = ""
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_path = os.path.join(CONFIG['output_dir'], 'best_model.pth')
            torch.save(checkpoint_data, best_path)
            saved_msg = "(*)"
            
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} MAE: {avg_train_mae:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} MAE: {avg_val_mae:.4f} | "
              f"Time: {epoch_time:.1f}s {saved_msg}")

    if end_epoch < CONFIG['epochs']:
        logger.info(
            "Completed %d epoch(s) this run (epochs_per_run=%d). Exiting for an external restart.",
            end_epoch - start_epoch,
            CONFIG['epochs_per_run'],
        )

    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
