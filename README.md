# Defect Formation Energy Prediction (缺陷形成能预测)

## 📖 项目概述

本项目旨在利用深度学习方法预测晶体结构中缺陷的形成能。晶体缺陷的形成能是决定材料性能（如电子结构、光学特性和热力学稳定性）的关键物理量。通过构建和应用先进的图神经网络（Graph Neural Networks, GNN），如 ALIGNN 和 CGCNN，本项目实现了基于原子结构特征的形成能高效、高精度预测。

## ✨ 功能特性

- **多模型支持**：集成了多种经典的图神经网络架构，包括：
  - **ALIGNN** (Atomistic Line Graph Neural Network)
  - **CGCNN** (Crystal Graph Convolutional Neural Networks)
  - **Transformer / No-Transformer** 自定义模型变体
- **物理先验的数据增强**：提供了专门的数据增强策略以提升模型鲁棒性：
  - **晶体旋转 (Rotation)**：保持物理性质不变的三维空间随机旋转。
  - **坐标微扰 (Perturbation)**：添加高斯噪声模拟热振动及测量误差。
- **自动训练恢复机制**：提供了强大的 Shell 运行脚本（如 `ALIGNN_runner.sh`），可自动处理训练过程中的显存泄漏 (Memory Leak) 或意外中断问题，实现自动重启。

## 🛠️ 技术栈

- **编程语言**: Python 3.x
- **深度学习框架**: PyTorch
- **晶体/图处理库**: Pymatgen, DGL (Deep Graph Library) 或相关依赖
- **操作系统**: macOS / Linux / Windows

## 📦 安装说明

1. **克隆项目到本地**
   ```bash
   git clone https://github.com/wuleyan2004/defect_formation_energy_prediction.git
   cd defect_formation_energy_prediction
   ```

2. **创建虚拟环境** (推荐使用 `conda` 或 `venv`)
   ```bash
   conda create -n defect_env python=3.9
   conda activate defect_env
   ```

3. **安装依赖项**
   由于不同模型可能对依赖版本有特定要求，请根据您使用的 PyTorch 和 CUDA 版本安装相应的包：
   ```bash
   pip install torch torchvision torchaudio
   pip install pymatgen
   # 若需要 DGL 等库，请按需安装
   ```

## 🚀 使用方法

### 数据准备与增强
您可以利用项目提供的数据增强脚本对原始数据进行扩充：
```bash
python data/data_augmentation.py
```
*有关增强策略的详细信息，请参阅 `data/data_augmentation.md`。*

### 模型训练与评估
项目中各个模型被组织在 `src` 目录的不同子文件夹中。推荐使用项目提供的 runner 脚本启动训练，以确保稳定运行：

**以 ALIGNN 模型为例：**
```bash
cd src/ALIGNN
bash ALIGNN_runner.sh
```

**以 CGCNN 模型为例：**
```bash
cd src/CGCNN
bash cgcnn_runner.sh
```

各个模块还包含了对应的测试脚本（如 `test_alignn.py`），用于评估保存的 `.pth` 模型权重。

## 🤝 贡献指南

欢迎大家提出 Issue 或提交 Pull Request。如果您想为本项目做出贡献，请遵循以下步骤：

1. Fork 本仓库。
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)。
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)。
4. 将您的分支推送到远程仓库 (`git push origin feature/AmazingFeature`)。
5. 开启一个 Pull Request。

## 📄 许可证信息

本项目采用 [MIT License](LICENSE) 许可证。您可以自由地使用、修改和分发本项目代码，但请保留原作者的版权声明。
