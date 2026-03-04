## ALIGNN 训练脚本

本目录用于用 ALIGNN 模型训练缺陷形成能数据集。

### 依赖

- Python + PyTorch
- alignn
- jarvis-tools
- dgl

安装示例：

```bash
pip install alignn jarvis-tools dgl
```

### 训练

默认路径按“项目根目录”解析：

- 数据集：data/final_dataset.pkl
- 输出目录：checkpoints/ALIGNN
- 断点：checkpoints/ALIGNN/latest_model.pth

直接运行：

```bash
python3 src/ALIGNN/train_alignn.py
```

常用参数：

```bash
python3 src/ALIGNN/train_alignn.py \
  --data-path data/final_dataset.pkl \
  --output-dir checkpoints/ALIGNN \
  --epochs 50 \
  --epochs-per-run 1
```

其中 `--epochs-per-run` 默认为 1，便于配合 runner 进行“每次跑 1 个 epoch 然后退出”的重启训练。

### Runner

老版 runner：

```bash
PYTHON_BIN=python3 bash src/ALIGNN/ALIGNN_runner.sh
```

新版 runner（推荐，可自动读取 checkpoint 判断是否训练结束）：

```bash
PYTHON_BIN=python3 bash src/ALIGNN/ALIGNN_runner_new.sh
```
