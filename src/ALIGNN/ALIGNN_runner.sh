
#!/bin/bash

# ALIGNN Runner Script
# Runs train_alignn.py in a loop to handle memory leaks (especially on MPS)

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_alignn.py}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MAX_RETRIES=100
COUNT=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    echo "----------------------------------------"
    echo "Starting Training Session $COUNT"
    echo "----------------------------------------"
    
    # Run the training script
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
    
    # Capture exit code
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully."
        break
    else
        echo "Training script exited with code $EXIT_CODE. Restarting..."
    fi
    
    COUNT=$((COUNT+1))
    sleep 2
done
