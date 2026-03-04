#!/bin/bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_alignn.py}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_FILE="${CKPT_FILE:-${PROJECT_ROOT}/checkpoints/ALIGNN/latest_model.pth}"

MAX_CYCLES="${MAX_CYCLES:-500}"
SLEEP_SECS="${SLEEP_SECS:-2}"
CRASH_LIMIT="${CRASH_LIMIT:-20}"

cycle=0
crashes=0

cd "${SCRIPT_DIR}"

read_progress() {
  CKPT="${CKPT_FILE}" "${PYTHON_BIN}" - <<'PY'
import os
import sys

ckpt_path = os.environ.get("CKPT", "")
if not ckpt_path or not os.path.exists(ckpt_path):
    sys.exit(2)

try:
    import torch
except Exception:
    sys.exit(3)

try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
except Exception:
    sys.exit(4)

epoch = int(ckpt.get("epoch", -1))
cfg = ckpt.get("config", {}) or {}
epochs = int(cfg.get("epochs", -1))

print(f"{epoch} {epochs}")
PY
}

cleanup_memory() {
  "${PYTHON_BIN}" - <<'PY' || true
import gc
gc.collect()

try:
    import torch
except Exception:
    raise SystemExit(0)

try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass

try:
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
except Exception:
    pass

gc.collect()
PY
}

echo "Working dir: ${SCRIPT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Checkpoint: ${CKPT_FILE}"

while [ "${cycle}" -lt "${MAX_CYCLES}" ]; do
  echo "----------------------------------------"
  echo "Cycle ${cycle}"
  echo "----------------------------------------"

  set +e
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
  exit_code=$?
  set -e

  progress=""
  if progress="$(read_progress 2>/dev/null)"; then
    :
  else
    progress=""
  fi

  if [ -n "${progress}" ]; then
    cur_epoch="$(echo "${progress}" | awk '{print $1}')"
    total_epochs="$(echo "${progress}" | awk '{print $2}')"
    echo "Checkpoint progress: epoch=${cur_epoch} / epochs=${total_epochs}"

    if [ "${total_epochs}" -gt 0 ] && [ "${cur_epoch}" -ge $((total_epochs - 1)) ]; then
      echo "Training reached the last epoch. Exiting."
      exit 0
    fi
  else
    echo "Checkpoint progress: unavailable (file missing or unreadable)."
  fi

  if [ "${exit_code}" -ne 0 ]; then
    crashes=$((crashes + 1))
    echo "Training exited with code ${exit_code}. crashes=${crashes}/${CRASH_LIMIT}"
    if [ "${crashes}" -ge "${CRASH_LIMIT}" ]; then
      echo "Crash limit reached. Exiting."
      exit "${exit_code}"
    fi
  else
    crashes=0
    echo "Training exited with code 0. Treating as a restart boundary and continuing."
  fi

  echo "Cleaning memory..."
  cleanup_memory

  cycle=$((cycle + 1))
  sleep "${SLEEP_SECS}"
done

echo "MAX_CYCLES reached (${MAX_CYCLES}). Exiting."
exit 1
