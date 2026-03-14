#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/root/Athena/maddpg/maddpg:/root/Athena/maddpg/multiagent-particle-envs

echo "Starting training..."
# cd /root/Athena && nohup python3 main.py > main.log 2>&1 &

timestamp=$(date +%Y%m%d_%H%M%S)
cd /root/Athena/maddpg/maddpg/experiments && stdbuf -oL python3 lc.py > /root/Athena/train_${timestamp}.log 2>&1 &