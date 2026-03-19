import os
import sys
import time
import argparse
import datetime
import random
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from data.data_loader import CrystalGraphDataset, collate_fn
except ImportError:
    from WLY.data_loader import CrystalGraphDataset, collate_fn

import logging


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


class Normalizer:
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor).clamp(min=1e-12)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean


class RBFExpansion(nn.Module):
    def __init__(self, dmin=0.0, dmax=8.0, bins=64):
        super().__init__()
        centers = torch.linspace(dmin, dmax, bins)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / ((dmax - dmin) / bins) ** 2

    def forward(self, dist):
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)


class CGCNNConv(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.fc = nn.Linear(2 * hidden_dim + edge_dim, 2 * hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index
        m = torch.cat([h[row], h[col], edge_attr], dim=-1)
        m = self.fc(m)
        gate, core = torch.chunk(m, chunks=2, dim=-1)
        gate = torch.sigmoid(gate)
        core = F.softplus(core)
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, gate * core)
        out = self.bn(h + agg)
        return F.softplus(out)


class CGCNNBaseline(nn.Module):
    def __init__(self, atom_fea_len=9, hidden_dim=128, n_conv=4, edge_bins=64):
        super().__init__()
        self.embedding = nn.Linear(atom_fea_len, hidden_dim)
        self.rbf = RBFExpansion(dmin=0.0, dmax=8.0, bins=edge_bins)
        self.convs = nn.ModuleList([CGCNNConv(hidden_dim, edge_bins) for _ in range(n_conv)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch_dict):
        x = batch_dict['x']
        mask = batch_dict['atom_mask']
        B, N, _ = x.shape
        device = x.device

        h = F.softplus(self.embedding(x))
        flat_h = h[mask]

        num_atoms = batch_dict['num_atoms'].to(device)
        offsets = torch.cumsum(
            torch.cat([torch.zeros(1, device=device, dtype=num_atoms.dtype), num_atoms[:-1]]), dim=0
        )

        all_edge_indices = []
        all_edge_dists = []

        for i in range(B):
            edge_index = batch_dict['edge_index_list'][i]
            edge_dist = batch_dict['edge_dist_list'][i]
            if edge_index.numel() == 0:
                continue
            all_edge_indices.append(edge_index + int(offsets[i].item()))
            all_edge_dists.append(edge_dist)

        if len(all_edge_indices) > 0:
            flat_edge_index = torch.cat(all_edge_indices, dim=1)
            flat_edge_dist = torch.cat(all_edge_dists, dim=0)
            edge_attr = self.rbf(flat_edge_dist)
            for conv in self.convs:
                flat_h = conv(flat_h, flat_edge_index, edge_attr)

        h_updated = torch.zeros_like(h)
        h_updated[mask] = flat_h

        mask_float = mask.float().unsqueeze(-1)
        pooled = (h_updated * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
        out = self.head(pooled).squeeze(-1)
        return out


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
    groups = defaultdict(list)
    for idx, sample in enumerate(tqdm(dataset.data, desc="Indexing")):
        base_id = get_base_id(sample['unique_id'])
        groups[base_id].append(idx)

    all_base_ids = list(groups.keys())
    random.seed(seed)
    random.shuffle(all_base_ids)

    n_total = len(all_base_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_ids = all_base_ids[:n_train]
    val_ids = all_base_ids[n_train:n_train + n_val]
    test_ids = all_base_ids[n_train + n_val:]

    train_indices = []
    val_indices = []
    test_indices = []

    for uid in train_ids:
        train_indices.extend(groups[uid])

    for uid in val_ids:
        for idx in groups[uid]:
            if is_original(dataset.data[idx]):
                val_indices.append(idx)

    for uid in test_ids:
        for idx in groups[uid]:
            if is_original(dataset.data[idx]):
                test_indices.append(idx)

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def default_project_root():
    return Path(__file__).resolve().parents[2]


def resolve_path(project_root: Path, p: str):
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return str(path)


def resolve_existing_dataset_path(data_path: str):
    path = Path(data_path)
    if path.exists():
        return str(path)

    fallback_candidates = []
    if path.name == "final_augmented_dataset.pkl":
        fallback_candidates.append(path.with_name("final_dataset.pkl"))
    fallback_candidates.append(path.with_name("processed_dataset_with_graphs.pkl"))
    fallback_candidates.append(path.with_name("cleaned_dataset.pkl"))

    for candidate in fallback_candidates:
        if candidate.exists():
            return str(candidate)

    return str(path)


def parse_args():
    project_root = default_project_root()
    parser = argparse.ArgumentParser(description="Train CGCNN baseline on defect formation energy dataset.")
    parser.add_argument("--data-path", default="data/final_dataset.pkl")
    parser.add_argument("--feature-path", default="data/atom_features.pth")
    parser.add_argument("--output-dir", default="checkpoints/CGCNN")
    parser.add_argument("--resume-path", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--epochs-per-run", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-conv", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    args.data_path = resolve_path(project_root, args.data_path)
    args.feature_path = resolve_path(project_root, args.feature_path)
    args.output_dir = resolve_path(project_root, args.output_dir)
    if args.resume_path:
        args.resume_path = resolve_path(project_root, args.resume_path)
    else:
        args.resume_path = os.path.join(args.output_dir, "latest_model.pth")

    if args.epochs_per_run < 0:
        raise SystemExit("--epochs-per-run must be >= 0")
    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit("--train-ratio must be in (0, 1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise SystemExit("--val-ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio must be < 1")
    return args


def choose_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    args = parse_args()
    config = {
        'data_path': args.data_path,
        'feature_path': args.feature_path,
        'output_dir': args.output_dir,
        'resume_path': args.resume_path,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'epochs_per_run': args.epochs_per_run,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'hidden_dim': args.hidden_dim,
        'n_conv': args.n_conv,
        'seed': args.seed,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
    }

    os.makedirs(config['output_dir'], exist_ok=True)
    logger = setup_logger(config['output_dir'])
    resolved_data_path = resolve_existing_dataset_path(config['data_path'])
    if resolved_data_path != config['data_path']:
        logger.warning(
            "Configured dataset not found, fallback to: %s",
            resolved_data_path,
        )
        config['data_path'] = resolved_data_path

    logger.info(f"Training started. Config: {config}")

    device = choose_device()
    logger.info(f"Using device: {device}")

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    if device.type == 'mps':
        torch.mps.empty_cache()

    full_dataset = CrystalGraphDataset(config['data_path'], config['feature_path'], device=device)
    all_targets = [sample['target'] for sample in full_dataset.data]
    valid_targets = [t for t in all_targets if not (np.isnan(t) or np.isinf(t) or abs(t) > 1e6)]
    if len(valid_targets) == 0:
        raise RuntimeError("No valid targets found for normalization.")
    target_tensor = torch.tensor(valid_targets, dtype=torch.float32, device=device)
    normalizer = Normalizer(target_tensor)
    logger.info(f"Target Norm Stats: Mean={normalizer.mean:.4f}, Std={normalizer.std:.4f}")

    train_set, val_set, test_set = split_dataset(
        full_dataset,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        seed=config['seed'],
    )
    logger.info(
        f"Dataset sizes | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = CGCNNBaseline(
        atom_fea_len=9,
        hidden_dim=config['hidden_dim'],
        n_conv=config['n_conv'],
        edge_bins=64,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.HuberLoss(delta=1.0)

    start_epoch = 0
    best_val_mae = float('inf')

    if config['resume_path'] and os.path.exists(config['resume_path']):
        logger.info(f"Resuming from checkpoint: {config['resume_path']}")
        checkpoint = torch.load(config['resume_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_mae = checkpoint.get('val_mae', float('inf'))
        logger.info(f"Restarting from epoch {start_epoch + 1}, best val MAE: {best_val_mae:.4f}")

    if start_epoch >= config['epochs']:
        logger.info("Already finished all epochs. Exiting.")
        return

    if config['epochs_per_run'] == 0:
        end_epoch = config['epochs']
    else:
        end_epoch = min(config['epochs'], start_epoch + config['epochs_per_run'])
    logger.info(f"Running epochs [{start_epoch + 1}, {end_epoch}] / {config['epochs']}")

    if device.type == 'mps':
        torch.mps.empty_cache()

    for epoch in range(start_epoch, end_epoch):
        start_time = time.time()
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_count = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Train]", unit="batch")
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
            bsz = targets.size(0)
            train_loss_sum += batch_loss * bsz

            with torch.no_grad():
                preds_denorm = normalizer.denorm(preds)
                abs_error = torch.abs(preds_denorm - targets)
                
                if torch.isnan(abs_error).any() or torch.isinf(abs_error).any():
                    logger.warning(f"Warning: NaN/Inf detected in batch MAE calculation!")
                    abs_error = torch.nan_to_num(abs_error, nan=0.0, posinf=0.0, neginf=0.0)

                mae = abs_error.mean().item()
                train_mae_sum += mae * bsz
            
            train_count += bsz
            train_pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'mae': f'{mae:.4f}'})

        avg_train_loss = train_loss_sum / max(1, train_count)
        avg_train_mae = train_mae_sum / max(1, train_count)

        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_count = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Val]", unit="batch")
        with torch.no_grad():
            for batch in val_pbar:
                preds = model(batch)
                targets = batch['target']
                targets_norm = normalizer.norm(targets)
                loss = criterion(preds, targets_norm)
                
                bsz = targets.size(0)
                val_loss_sum += loss.item() * bsz

                preds_denorm = normalizer.denorm(preds)
                abs_error = torch.abs(preds_denorm - targets)
                
                if torch.isnan(abs_error).any():
                     logger.warning(f"Warning: NaN detected in val batch!")
                     abs_error = torch.nan_to_num(abs_error, nan=0.0)
                
                mae = abs_error.mean().item()
                val_mae_sum += mae * bsz
                val_count += bsz
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{mae:.4f}'})

                if device.type == 'mps':
                    torch.mps.empty_cache()

        avg_val_loss = val_loss_sum / max(1, val_count)
        avg_val_mae = val_mae_sum / max(1, val_count)
        scheduler.step(avg_val_mae)

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'normalizer': {'mean': normalizer.mean, 'std': normalizer.std},
            'config': config,
            'val_mae': avg_val_mae,
        }

        latest_path = os.path.join(config['output_dir'], 'latest_model.pth')
        torch.save(checkpoint_data, latest_path)

        saved_msg = ""
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_path = os.path.join(config['output_dir'], 'best_model.pth')
            torch.save(checkpoint_data, best_path)
            saved_msg = "(*)"

        gc.collect()
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} MAE: {avg_train_mae:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} MAE: {avg_val_mae:.4f} | "
            f"Time: {epoch_time:.1f}s {saved_msg}"
        )

        print("🚩 这一轮跑完啦，我要自杀重启来清理内存了...")
        break  # 强制跳出循环，结束 Python 进程

    logger.info(f"Training Complete. Best Validation MAE: {best_val_mae:.4f} eV")


if __name__ == "__main__":
    main()