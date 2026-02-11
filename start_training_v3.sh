#!/bin/bash

# Navigate to Athena directory
cd /root/Athena

# Reset session
echo "Resetting config server session..."
python3 reset_session.py

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="train_${TIMESTAMP}.log"

echo "Starting training..."
echo "Logs are redirected to ${LOG_FILE}"

# Run training in background with nohup
nohup python3 -u maddpg/maddpg/experiments/lc.py > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Training process started with PID: ${PID}"
