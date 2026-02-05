#!/bin/bash

echo "Starting training..."
cd /root/Athena && nohup python3 main.py > main.log 2>&1 &

cd /root/Athena/maddpg/maddpg/experiments && nohup python3 lc.py > train.log 2>&1 &