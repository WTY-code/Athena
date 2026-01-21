# Athena Caliper Deployment Guide & Troubleshooting

This document summarizes the deployment process for the Athena/Caliper project, including critical fixes applied to the legacy codebase, common pitfalls, and a step-by-step guide to running experiments from scratch.

## 1. Critical Fixes & "Pitfalls" (Read Before Deploying)

The original codebase had several compatibility issues with modern environments. The following fixes have been applied to the current workspace:

### A. Environment Compatibility (Node.js & Dependencies)
*   **Issue:** The project uses legacy code (c. 2018) which is incompatible with modern Node.js versions. Dependencies like `@so-ric/colorspace` caused syntax errors (`||=`, `Object.hasOwn`) when installed on newer Node versions.
*   **Fix:** 
    *   Updated `docker/Dockerfile` to use **Node 10 (Buster)**.
    *   Added `sed` patches in the Dockerfile to fix syntax errors in global `node_modules`.
    *   Switched to **global installation** of `caliper-cli` to avoid issues with volume-mounted `node_modules` hiding installed packages.
    *   Updated `scripts/boot.sh` to invoke `caliper` directly (instead of `npx`).

### B. Network & DNS Conflicts
*   **Issue:** The deployment uses **CoreDNS** on port **53** to handle internal domain resolution (`*.example.com`).
*   **Pitfall:** Tools like **Clash**, `systemd-resolved`, or other proxies often bind to port 53, causing CoreDNS to fail. This leads to `Connection refused` or timeout errors when peers try to contact the CA or Orderer.
*   **Solution:** Ensure port 53 is free on the master node before starting.
    ```bash
    # Check who is using port 53
    netstat -ulpn | grep 53
    # Stop conflicting services (example)
    pkill clash
    ```

### C. Docker Image Distribution
*   **Issue:** Remote nodes often fail to pull `hyperledger/fabric-baseos` or `fabric-ccenv` due to network restrictions or Docker Hub rate limits. This causes "Chaincode instantiation failed" errors.
*   **Solution:** Images must be manually distributed to all nodes.
    *   `hyperledger/fabric-baseos:amd64-0.4.22` (Required for Peer runtime)
    *   `hyperledger/fabric-ccenv:1.4` (Required for Chaincode build)

### D. Configuration & Ansible
*   **Ansible:** Removed `miniconda` dependency from `ansible/deploy-up.yaml` and `ansible/deploy-down.yaml`. The playbook now uses the system `docker-compose`.
*   **Client Config:** Removed a hanging `wget` command in `templates/client-base.yaml` that tried to contact a hardcoded IP (`10.10.7.51`), causing the benchmark to hang at completion.
*   **NFS Stale Handle:** Added `umount -l /root/ansible/nfs` in `main.py` deployment loop to prevent "stale file handle" errors during repeated deployments.

---

## 2. Prerequisites

*   **OS:** Linux (Ubuntu/Debian recommended).
*   **Python 3:** Installed on all nodes.
*   **Docker & Docker Compose:** Installed on all nodes.
*   **Ansible:** Installed on the master node.
*   **SSH Access:** Master node must be able to SSH to all worker nodes (configured in `/etc/ansible/hosts`).
*   **Dependencies:**
    *   `pip install tensorflow==2.11.0 gym==0.10.5 pandas lxml`
    *   `pip install --upgrade Jinja2 Flask` (Fixes Ansible compatibility)

---

## 3. Step-by-Step Deployment Guide (From Scratch)

### Step 1: Prepare the Codebase
Ensure you are in the correct directory.
```bash
cd /root/Athena/caliper-deploy-tool
```

### Step 2: Configure Deployment
1.  **Ansible Hosts:** Verify `/etc/ansible/hosts` contains your cluster IPs.
2.  **Action Config:** Ensure `action.yaml` exists.
    ```bash
    cp ../action.default.yaml action.yaml
    ```

### Step 3: Build the Control Tool (CDT)
Rebuild the local Docker image to apply environment patches.
```bash
make setup-cdt
```

### Step 4: Distribute Docker Images (Crucial)
To avoid pull failures on remote nodes, save and copy the base images.

**On Master Node:**
```bash
# 1. Pull/Save BaseOS
docker pull hyperledger/fabric-baseos:amd64-0.4.22
mkdir -p ansible/images
docker save hyperledger/fabric-baseos:amd64-0.4.22 -o ansible/images/baseos.tar

# 2. Distribute & Load to all nodes via Ansible
ansible cdt -m copy -a "src=ansible/images/baseos.tar dest=/root/baseos.tar"
ansible cdt -m shell -a "docker load -i /root/baseos.tar"

# 3. Ensure ccenv tag is correct on all nodes
ansible cdt -m shell -a "docker tag hyperledger/fabric-ccenv:latest hyperledger/fabric-ccenv:1.4 || true"
```

### Step 5: Generate Configuration
Generate the crypto-config and distributed configuration files.
```bash
make generate
```

### Step 6: Start the Network
1.  **Clean up previous runs:**
    ```bash
    make deploy-fabric-down
    ```
    *If you get NFS errors, unmount explicitly:*
    ```bash
    ansible cdt -m shell -a "umount -l /root/ansible/nfs"
    ```

2.  **Setup NFS Shares:**
    ```bash
    make setup-config
    ```

3.  **Deploy Fabric Containers:**
    ```bash
    make deploy-fabric-up
    ```

### Step 7: Run the Benchmark
Ensure the `CDTIP` variable is set to the Master Node's LAN IP (the IP remote nodes use to reach the master).

```bash
# Replace with your Master Node IP
export CDTIP=192.168.0.25 

make start-cdt
```

### Step 8: View Results
Upon success, the report is generated at:
`caliper-deploy-tool/report.html`

---

## 4. DRL Training Setup (Athena)

This section covers the setup for the Deep Reinforcement Learning agent that optimizes the Fabric network.

### A. Dependencies & Migration
The original code was written for TensorFlow 1.x. The following changes were made to support TF 2.x:
1.  **Imports:** Used `tensorflow.compat.v1` and `tf.disable_v2_behavior()`.
2.  **Contrib:** Replaced `tensorflow.contrib.layers` with `tf.layers` (compat).
3.  **Dependencies:** Removed incompatible version constraints from `requirements.txt` and installed `gym-aigis`, `maddpg`, `multiagent-particle-envs` in editable mode.

### B. Critical Runtime Fixes
1.  **MongoDB Disabled:** `lc.py` originally tried to log every step to a local MongoDB. This was disabled (commented out) to allow running without a DB.
2.  **Robust Metrics Collection:**
    *   **Prometheus:** Modified `utils/metrics.py` to cast values to float and handle errors, ensuring only numeric data enters the RL model.
    *   **Caliper/Docker:** Modified `collector.py` to handle missing/NaN Docker stats (CPU/Mem). If Caliper fails to monitor Docker, it defaults to 0.0 instead of crashing. This ensures the reward function (based on TPS/Latency) still works.
3.  **Pandas Compatibility:** Modified `env.py` to split chained `fillna().clip()` operations and force float casting, preventing `AssertionError` in newer Pandas versions.

---

## 5. Running the Automated Training Loop

The training consists of two components:
1.  **Parameter Server (`main.py`):** Handles deployment requests and executes Ansible playbooks.
2.  **DRL Agent (`lc.py`):** Generates configurations, requests deployment, observes metrics, and learns.

### Startup Command
Run both in background (using `nohup` recommended):

```bash
# 1. Start the Parameter Server (Terminal 1)
cd /root/Athena
nohup python3 main.py > main.log 2>&1 &

# 2. Start the DRL Agent (Terminal 2)
cd /root/Athena/maddpg/maddpg/experiments
nohup python3 lc.py > train.log 2>&1 &
```

### Monitoring
*   **Deployment Status:** `tail -f /root/Athena/main.log`
*   **Training Progress:** `tail -f /root/Athena/maddpg/maddpg/experiments/train.log`
*   **Reports:** Check `/root/Athena/caliper-deploy-tool/history/` for HTML reports generated after each iteration.

---

## 6. Troubleshooting DRL Training

**Q: `NameError: name 'mycol' is not defined`**
A: You commented out the MongoDB initialization but left the `insert_one` call active.
*   **Fix:** Ensure all MongoDB-related lines in `lc.py` are commented out.

**Q: `ValueError: All arrays must be of the same length`**
A: `main.py` attempted to aggregate metrics from different node types (Peer vs Orderer) into a single DataFrame.
*   **Fix:** In `main.py`, ensure `metrics_collector.collect_from_prometheus()` is called **without** a handler argument. Let `env.py` handle the raw dictionary structure.

**Q: `TypeError: Could not convert ... to numeric`**
A: Pandas encountered strings (e.g., container names) when calculating means.
*   **Fix:** Ensure `utils/metrics.py` casts values to float, and `collector.py` uses `mean(numeric_only=True)`.

**Q: GPU Warnings in `train.log`?**
A: `kernel driver does not appear to be running...`
*   **Status:** Harmless. TensorFlow falls back to CPU automatically.
