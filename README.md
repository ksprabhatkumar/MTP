# 🏃‍♂️ Mamba-WHAR: State-Space Graph Framework for Wearable HAR

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-blue.svg)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains **Phase 1** of an M.Tech research thesis focusing on Deep Learning for Time-Series IMU data. It is a next-generation evolution of the **GT-WHAR (Zou et al., IEEE TETCI 2024)** architecture.

By replacing the computationally intensive **Bidirectional GRU** with a **State-Space Model (Mamba)**, this framework enables efficient parallel sequence modeling, significantly reducing inference latency while maintaining strong recognition performance on long-duration activities.

---

## 🧠 Architecture Overview

1. **Spatial Extraction (Body-Node Attention Graph Network)**  
   Utilizes Graph Isomorphism Networks (GIN) combined with a Node Attention branch to learn the structural relationships between body nodes and sensor modalities.

2. **Temporal Extraction (Pure PyTorch Mamba Block)**  
   Replaces the recurrent BiGRU with a **State Space Model (SSM)** based Mamba implementation, enabling efficient long-range temporal modeling without requiring custom CUDA or C++ compilation.

3. **Robust Regularization**  
   Incorporates **Z-Score Normalization**, **Residual Connections**, **Layer Normalization**, and **Dropout (0.3)** to improve generalization and reduce overfitting during cross-subject evaluation.

---

## 📂 Project Structure

```text
Mamba_WHAR_Phase1/
├── download_all_data.py      # Automated downloader for UCI HAR datasets
├── process_all_data.py       # Z-Score normalization and tensor generation
├── main.py                   # Master training loop with checkpointing
├── requirements.txt          # Python dependencies
├── results_log.txt           # Final experimental results and metrics
├── checkpoints/              # Stateful save-states for crash recovery
├── har_data/                 # Raw extracted datasets
├── dataset/
│   ├── processed/            # Final .pt tensors (X, Y, P)
│   └── data_loader.py        # Dynamic multi-dataset dataloader
├── models/
│   └── mamba_whar.py         # GIN + Mamba architecture
└── utils/
    ├── metrics.py            # Accuracy & Macro-F1 calculators
    └── logger.py             # Experiment logging utilities
```

---

## 🛠️ Installation & Setup

It is highly recommended to use a virtual environment. For optimal performance,
training should be performed on a CUDA-enabled GPU.

### 1️⃣ Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
# OR
venv\Scripts\activate          # Windows CMD / PowerShell
```

### 2️⃣ Install PyTorch with CUDA Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3️⃣ Install Remaining Dependencies

```bash
pip install torch-geometric numpy scikit-learn tqdm
```

---

## 🚀 Execution Pipeline

The complete workflow can be reproduced in three simple steps.

### Step 1: Download the Datasets

You can manually place the datasets inside `har_data/`, or download them automatically:

```bash
python download_all_data.py
```

### Step 2: Process and Normalize the Data

This script parses the raw sensor files, performs **Z-Score Normalization**, groups samples by Subject ID, and generates the final `.pt` tensors.

```bash
python process_all_data.py
```

### Step 3: Train the Model

Run a specific dataset:

```bash
python main.py --dataset dsads
```

Run all datasets sequentially:

```bash
python main.py --dataset all
```

---

## ⚡ Key Features

### 🔹 Dynamic Multi-Dataset Support

The framework automatically adapts its:

- Input dimensions
- Graph topology
- Classification head

for different datasets including:

- DSADS
- OPPORTUNITY
- REALDISP

---

### 🔹 Strict LOSO Validation Protocol

To eliminate **data leakage** and evaluate true cross-subject generalization, the framework employs **Leave-One-Subject-Out (LOSO)** cross-validation.

- Training is performed on **N−1 subjects**.
- Testing is conducted on **one completely unseen subject**.
- The process is repeated for all subjects and the final metrics are averaged.

---

### 🔹 Automatic Checkpointing & Crash Recovery

Training graph-based state-space models can be computationally expensive.
Therefore, the framework provides **stateful global checkpointing**.

After every epoch, the following are saved:

- Model weights
- Optimizer state
- Dataset name
- Current fold and epoch

Checkpoint location:

```text
checkpoints/latest_mamba_checkpoint.pth
```

If training is interrupted, simply rerun:

```bash
python main.py --dataset all
```

The script automatically detects the checkpoint and resumes from the exact dataset, fold, and epoch where training stopped.

---

## 📊 Results

The framework reports performance using:

- **Classification Accuracy**
- **Macro F1-Score**

Detailed fold-wise metrics and the final averaged cross-subject performance are automatically appended to:

```text
results_log.txt
```

The proposed **GIN + Mamba** architecture serves as a scalable and efficient baseline for future research on graph-based state-space modeling for wearable human activity recognition.