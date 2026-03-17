import matplotlib.pyplot as plt
import numpy as np
import re

log_file = "/root/Athena/train_20260315_103919.log"

steps = []
tps_success_list = []

current_step = None
has_error = False

with open(log_file, 'r') as f:
    for line in f:
        step_match = re.search(r"Train_step now:\s*(\d+)", line)
        if step_match:
            current_step = int(step_match.group(1))
            has_error = False
            continue
            
        if "Deployment exception" in line or "Failed to contact server" in line:
            if current_step is not None:
                has_error = True
                
        obs_match = re.search(r"Obs:\s*TPS=([\d\.]+).*SR=([\d\.]+)", line)
        if obs_match and current_step is not None:
            if "Rew=-100.0" in line:
                has_error = True
                
            tps = float(obs_match.group(1))
            sr = float(obs_match.group(2))
            successful_tps = tps * sr
            
            steps.append(current_step)
            if has_error:
                tps_success_list.append(np.nan)
            else:
                tps_success_list.append(successful_tps)
                
            current_step = None # reset for next step

plt.figure(figsize=(12, 6))
plt.plot(steps, tps_success_list, marker='o', linestyle='-', color='b', linewidth=2)

# Add some visual markers for errors if wanted, but standard NaN is enough
# For completeness, let's just plot the valid data points and let the line break
plt.title('Athena Training Process: Successful TPS per Step', fontsize=14)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Successful TPS (TPS * SR)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(steps) # show all steps on x-axis

# Save the plot
for i, val in enumerate(tps_success_list):
    if not np.isnan(val):
        plt.annotate(f"{val:.2f}", (steps[i], val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

output_path = '/root/Athena/successful_tps_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved to {output_path}")

# Print data to terminal to verify
for s, val in zip(steps, tps_success_list):
    print(f"Step {s}: {'ERROR (NaN)' if np.isnan(val) else f'{val:.2f}'}")

