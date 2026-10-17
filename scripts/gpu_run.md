# GPU Precomputation & Benchmark Runbook

Step-by-step instructions for executing Decider 2B inference on a remote GPU environment (Google Colab T4/A100 or Kaggle Notebooks P100/T4) to precompute decision caches and measure warm GPU latency without CPU reference kernel bottlenecks.

---

## 1. Google Colab (Free T4 or Pro A100)

### Step 1: Initialize GPU Runtime
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime** → **Change runtime type** → select **T4 GPU** (or A100 if available) → **Save**.
3. Verify GPU availability in the first cell:
```bash
!nvidia-smi
```

### Step 2: Clone or Upload Repository
```bash
# Clone the repository
!git clone https://github.com/harshitsaxena214/patient-listener.git
%cd patient-listener
```

### Step 3: Install Dependencies
```bash
# In Colab, uninstall pre-installed torchao first to avoid version conflict (peft requires >= 0.16.0)
!pip uninstall -y torchao

# Install PyTorch with CUDA support and Strands Decider
!pip install --upgrade pip
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install strands-decider strands-agents boto3 psutil pyyaml numpy pydantic-settings
```

### Step 4: Configure GPU Environment Variables
```bash
import os

# Configure cache paths and execution mode
os.environ["HF_HOME"] = "/content/hf-cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["DECIDER_BACKEND"] = "local"
os.environ["MOCK_BEDROCK"] = "1"
os.environ["PYTHONPATH"] = "."

print("Configured GPU Environment:")
print("  HF_HOME:", os.environ["HF_HOME"])
print("  DECIDER_BACKEND:", os.environ["DECIDER_BACKEND"])
```

### Step 5: Run Decider Benchmark (Measure Real GPU Latency)
```bash
!python scripts/bench_decider.py
```
*Expected output: warmup + warm inference median and p95 latencies across 20, 60, and 150 token states on GPU (Tesla T4 median ~235.6 ms).*

### Step 6: Verify Precomputation Coverage & Runtime
```bash
# Verify all base states, pace profile variants, and Q2 mid_thought questions are enumerated
!python scripts/precompute_decisions.py --enumerate
```
*Expected enumeration: exactly 8,450 unique states across dev and test splits, estimated at ~33.2 minutes total on Tesla T4 at 235.6 ms/call (well under the 90-minute limit).*

### Step 7: Precompute Single Unified Decision Cache on GPU
```bash
# Runs single-process, fully resumable precomputation over all 8,450 states (skips cached keys)
!python scripts/precompute_decisions.py local
```
*The cache is automatically written and flushed to `backend/eval/data/decisions_cache.jsonl`.*

### Step 8: Download Precomputed Cache & Benchmark Output
```bash
from google.colab import files

# Download the populated GPU cache
files.download("backend/eval/data/decisions_cache.jsonl")
```

---

## 2. Kaggle Notebook (Free 2x T4 or P100)

### Step 1: Create Notebook with GPU Accelerator
1. In Kaggle, create a **New Notebook**.
2. In the right panel under **Settings**:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
   - **Internet**: Toggle **On**.

### Step 2: Clone Repository & Install Dependencies
```bash
!git clone https://github.com/harshitsaxena214/patient-listener.git /kaggle/working/patient-listener
%cd /kaggle/working/patient-listener

!pip install strands-decider strands-agents boto3 psutil pyyaml numpy pydantic-settings
```

### Step 3: Set Environment & Execute Precomputation
```bash
import os
os.environ["HF_HOME"] = "/kaggle/working/hf-cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["DECIDER_BACKEND"] = "local"
os.environ["MOCK_BEDROCK"] = "1"
os.environ["PYTHONPATH"] = "/kaggle/working/patient-listener"

# Run latency benchmark
!python scripts/bench_decider.py

# Run precomputation
!python scripts/precompute_decisions.py local
```

### Step 4: Export Cache File
The output file is located at:
`/kaggle/working/patient-listener/backend/eval/data/decisions_cache.jsonl`
Download this file from the Kaggle file explorer and place it into `backend/eval/data/decisions_cache.jsonl` on your local machine to evaluate with real GPU weights.

---

## 3. Merging GPU Results Back into Local Pipeline

Once `decisions_cache.jsonl` is transferred back to `backend/eval/data/decisions_cache.jsonl`:
```bash
# Re-run local evaluation harness (replays on VirtualClock using the GPU cache)
python backend/eval/run_eval.py
```
The harness will automatically read the updated cache and evaluate the policy using the measured GPU latencies.
