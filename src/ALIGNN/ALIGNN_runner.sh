
#!/bin/bash

# ALIGNN Runner Script
# Runs train_alignn.py in a loop to handle memory leaks (especially on MPS)

MAX_RETRIES=100
COUNT=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    echo "----------------------------------------"
    echo "Starting Training Session $COUNT"
    echo "----------------------------------------"
    
    # Run the training script
    /usr/bin/python3 train_alignn.py
    
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
