# 实验设置总结 (Experimental Settings Summary)

基于 `train_augmentation.py` 的代码分析，以下是本次实验的详细设置总结。

## 1. 基础超参数 (Hyperparameters)
- **Batch Size**: 32
- **Epochs**: 50
- **Learning Rate (LR)**: 1e-4
- **Random Seed**: 42 (用于 `torch`, `numpy`, `random`)
- **Device**: MPS (如果可用) 或 CPU

## 2. 模型架构 (Model Architecture)
- **模型**: `CrystalTransformer`
- **输入特征长度 (Atom Feature Length)**: 9
- **隐藏层维度 (Hidden Dim)**: 64
- **局部层数 (Local Layers)**: 2
- **全局层数 (Global Layers)**: 1

## 3. 优化器与调度器 (Optimizer & Scheduler)
- **优化器**: `AdamW`
  - Learning Rate: 1e-4
  - Weight Decay: 1e-4
- **学习率调度器**: `ReduceLROnPlateau`
  - Mode: `min` (监控验证集指标)
  - Factor: 0.5 (学习率衰减倍数)
  - Patience: 5 (容忍的 epoch 数)
- **损失函数**: `HuberLoss`
  - Delta: 1.0

## 4. 数据处理与划分 (Data Processing & Split)
- **数据集划分策略**:
  - 基于 **晶体唯一 ID (Base ID)** 进行划分，确保同一晶体的原始样本和增强样本都在同一集合中。
  - **训练集 (Train)**: 80% 的晶体 ID。包含原始样本 + **所有增强样本**。
  - **验证集 (Val)**: 10% 的晶体 ID。**仅包含原始样本**。
  - **测试集 (Test)**: 剩余 10% 的晶体 ID。**仅包含原始样本**。
- **归一化 (Normalization)**:
  - 对目标值 (Target) 进行标准化 (Z-score normalization)。
  - 统计量 (`mean`, `std`) 计算时过滤了极端异常值 (abs > 1e6)。

## 5. 评价指标 (Evaluation Metrics)
- **主要指标**: Mean Absolute Error (MAE)
- **单位**: eV
- **计算方式**:
  - 模型输出经过反归一化 (Denormalization) 还原为真实尺度。
  - 计算预测值与真实值之间的绝对误差平均值。
  - 在验证集上选择 MAE 最小的模型作为最佳模型 (`best_model.pth`)。

## 6. 其他设置
- **检查点 (Checkpoints)**:
  - 保存最新模型 (`latest_model.pth`) 和最佳模型 (`best_model.pth`)。
  - 支持从断点恢复训练 (`resume_path`)。
- **日志 (Logging)**:
  - 记录训练和验证的 Loss 及 MAE。
  - 记录每次 Epoch 的耗时。
