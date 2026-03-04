import os
import torch

ckpt_path = os.environ["CKPT_FILE"]
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

epochs = int(os.environ.get("DUMMY_EPOCHS", "3"))

epoch = -1
if os.path.exists(ckpt_path):
    try:
        prev = torch.load(ckpt_path, map_location="cpu")
        epoch = int(prev.get("epoch", -1))
    except Exception:
        epoch = -1

epoch += 1
torch.save({"epoch": epoch, "config": {"epochs": epochs}}, ckpt_path)
