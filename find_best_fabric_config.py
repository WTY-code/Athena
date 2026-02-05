
import re
import ast
import json

MAIN_LOG = "/root/Athena/main.log"
TRAIN_LOG = "/root/Athena/maddpg/maddpg/experiments/train.log"

def extract_configs(logfile):
    configs = []
    with open(logfile, 'r') as f:
        for line in f:
            if "config data:  {" in line:
                try:
                    json_str = line.split("config data:  ")[1].strip()
                    config = ast.literal_eval(json_str)
                    configs.append(config)
                except Exception as e:
                    pass
    return configs

def extract_rewards(logfile):
    rewards = []
    metrics = []
    # Read entire file content to handle multiline
    with open(logfile, 'r') as f:
        content = f.read()
        
    # Regex to find tps, latency, reward
    # Pattern matches: tps: 123.4, latency: 1.23, reward: 0.99
    # It might be preceded by "], array([...])" etc.
    matches = re.findall(r"tps: ([\d\.]+), latency: ([\d\.]+), reward: ([\d\.\-]+)", content)
    
    for match in matches:
        tps = float(match[0])
        latency = float(match[1])
        reward = float(match[2])
        
        rewards.append(reward)
        metrics.append({
            "tps": tps,
            "latency": latency,
            "reward": reward
        })
            
    return rewards, metrics

def main():
    print("Extracting configs...")
    configs = extract_configs(MAIN_LOG)
    print(f"Found {len(configs)} configs.")
    
    print("Extracting rewards...")
    rewards, metrics = extract_rewards(TRAIN_LOG)
    print(f"Found {len(rewards)} rewards.")
    
    # Align
    count = min(len(configs), len(rewards))
    if count == 0:
        print("No matching data found.")
        return

    best_idx = -1
    best_reward = -float('inf')
    
    # Iterate through the aligned data
    # Note: Configs are logged BEFORE reward.
    # config[0] -> reward[0]
    for i in range(count):
        if rewards[i] > best_reward:
            best_reward = rewards[i]
            best_idx = i
            
    print("-" * 50)
    print("BEST CONFIGURATION FOUND")
    print(f"Step Index: {best_idx}")
    print(f"Reward: {metrics[best_idx]['reward']}")
    print(f"TPS: {metrics[best_idx]['tps']}")
    print(f"Latency: {metrics[best_idx]['latency']} s")
    print("-" * 50)
    print("Configuration:")
    print(json.dumps(configs[best_idx], indent=2))

if __name__ == "__main__":
    main()
