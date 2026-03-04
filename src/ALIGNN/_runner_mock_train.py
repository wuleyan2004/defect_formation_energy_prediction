import os
import sys

import torch


def main():
    CONFIG = {
        "output_dir": "/tmp/alignn_runner_mock_checkpoints",
        "resume_path": "/tmp/alignn_runner_mock_checkpoints/latest_model.pth",
        "epochs": 3,
    }

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    start_epoch = 0
    if CONFIG["resume_path"] and os.path.exists(CONFIG["resume_path"]):
        ckpt = torch.load(CONFIG["resume_path"], map_location="cpu")
        start_epoch = int(ckpt.get("epoch", -1)) + 1

    if start_epoch >= CONFIG["epochs"]:
        print("Already finished.")
        return 0

    ckpt = {
        "epoch": start_epoch,
        "config": {"epochs": CONFIG["epochs"]},
    }
    torch.save(ckpt, CONFIG["resume_path"])
    print(f"Mock epoch {start_epoch + 1}/{CONFIG['epochs']} saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

