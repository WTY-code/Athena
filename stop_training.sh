#!/bin/bash
echo "Stopping training processes..."
# Find PIDs for main.py and lc.py (excluding grep itself)
pids=$(ps -ef | grep "python3.*\(main.py\|lc.py\)" | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "No training processes found."
else
    echo "Killing PIDs: $pids"
    # Kill the processes
    echo "$pids" | xargs kill
    echo "Training stopped."
fi
