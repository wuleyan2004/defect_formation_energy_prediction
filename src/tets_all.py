import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os
import sys

# 导入你的模型和数据类
from data_loader import CrystalGraphDataset, collate_fn
from model import CrystalTransformer

def test_on_full_dataset():
    # --- 1. 手动指定关键配置 (必须与 train.py 一致) ---
    CONFIG = {
        'data_path': '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/final_dataset.pkl',
        'feature_path': '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/atom_features.pth',
        'checkpoint': '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/checkpoints/best_model.pth',
        'hidden_dim': 64,
        'n_local': 2,
        'n_global': 1,
        'seed': 42
    }
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. 加载全量数据集 ---
    print("Loading dataset...")
    full_dataset = CrystalGraphDataset(CONFIG['data_path'], CONFIG['feature_path'], device=device)
    
    # 模拟 train.py 的划分逻辑，确保我们拿到的 test_set 和训练时是一样的
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # 使用相同的随机种子进行划分
    torch.manual_seed(CONFIG['seed'])
    _, _, test_set = random_split(full_dataset, [train_size, val_size, test_size])
    
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # --- 3. 获取 Normalizer 信息 ---
    # 我们需要从 best_model.pth 中读取训练时的 Mean 和 Std，否则预测值会完全错掉
    print(f"Loading checkpoint: {CONFIG['checkpoint']}")
    checkpoint = torch.load(CONFIG['checkpoint'], map_location=device)
    
    norm_mean = checkpoint['normalizer']['mean']
    norm_std = checkpoint['normalizer']['std']
    print(f"Loaded Normalizer: Mean={norm_mean:.4f}, Std={norm_std:.4f}")

    # --- 4. 初始化模型并加载权重 ---
    model = CrystalTransformer(
        atom_fea_len=9,
        hidden_dim=CONFIG['hidden_dim'],
        n_local_layers=CONFIG['n_local'],
        n_global_layers=CONFIG['n_global']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- 5. 开始推理 ---
    all_preds = []
    all_reals = []

    print(f"Starting inference on {len(test_set)} samples...")
    with torch.no_grad():
        for batch in test_loader:
            preds_norm = model(batch)
            targets = batch['target']
            
            # 反标准化：还原真实物理单位 (eV)
            preds_denorm = preds_norm * norm_std + norm_mean
            
            all_preds.extend(preds_denorm.cpu().numpy().flatten())
            all_reals.extend(targets.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_reals = np.array(all_reals)

    # --- 6. 计算指标 ---
    mae = np.mean(np.abs(all_preds - all_reals))
    rmse = np.sqrt(np.mean((all_preds - all_reals)**2))
    
    print("\n" + "="*30)
    print(f"Final Test MAE: {mae:.4f} eV")
    print(f"Final Test RMSE: {rmse:.4f} eV")
    print("="*30)

    # --- 7. 绘制散点图 (Parity Plot) ---
    plt.figure(figsize=(8, 8))
    plt.scatter(all_reals, all_preds, alpha=0.6, edgecolors='w', label=f'Model Predictions')
    
    # 理想线 y = x
    min_val = min(min(all_reals), min(all_preds))
    max_val = max(max(all_reals), max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal (Real=Pred)')
    
    plt.xlabel('Ground Truth Energy (eV)', fontsize=12)
    plt.ylabel('Predicted Energy (eV)', fontsize=12)
    plt.title(f'Crystal Energy Prediction (MAE: {mae:.4f} eV)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存结果图
    save_path = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/test_result_plot.png'
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    # ... 前面的推理代码保持不变 ...
    all_preds = np.array(all_preds)
    all_reals = np.array(all_reals)

    # 1. 计算每个样本的绝对误差
    errors = np.abs(all_preds - all_reals)

    # 2. 获取误差最大的前 10 个样本的索引
    # argsort 会按从小到大排，我们取最后 10 个并翻转
    top_k = 50
    worst_idx = np.argsort(errors)[-top_k:][::-1]

    print(f"\n======== 误差最大的前 {top_k} 个样本分析 ========")
    print(f"{'排名':<6} | {'索引':<10} | {'真实值(eV)':<12} | {'预测值(eV)':<12} | {'误差(eV)':<10}")
    print("-" * 65)

    for i, idx in enumerate(worst_idx):
        print(f"{i+1:<8} | {idx:<10} | {all_reals[idx]:<12.4f} | {all_preds[idx]:<12.4f} | {errors[idx]:<10.4f}")
if __name__ == "__main__":
    test_on_full_dataset()