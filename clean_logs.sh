#!/bin/bash
echo "Cleaning logs and history..."

# Clear main log
if [ -f "/root/Athena/main.log" ]; then
    > /root/Athena/main.log
    echo "Cleared main.log"
fi

# Clear training log
if [ -f "/root/Athena/maddpg/maddpg/experiments/train.log" ]; then
    > /root/Athena/maddpg/maddpg/experiments/train.log
    echo "Cleared train.log"
fi

if [ -f "/root/Athena/caliper-deploy-tool/caliper.log" ]; then
    > /root/Athena/caliper-deploy-tool/caliper.log
    echo "Cleared caliper.log"
fi

# Remove history
if [ -d "/root/Athena/caliper-deploy-tool/history" ]; then
    rm -rf /root/Athena/caliper-deploy-tool/history/*
    echo "Cleared history directory"
fi


echo "All logs cleaned."
