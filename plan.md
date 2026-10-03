# MTP-1: Placement-Agnostic WHAR via Dynamic Graphs

## 1. The Core Problem
In real-world Wearable Human Activity Recognition (WHAR), users frequently misplace sensors (e.g., strapping a smartwatch to the right leg instead of the right arm). Standard Graph Neural Networks (GNNs) rely on a **static, hardcoded skeletal adjacency matrix**. When a sensor is displaced, the static graph feeds leg acceleration data into arm-specific spatial filters, causing a catastrophic collapse in classification accuracy. The network cannot mathematically adapt its topology to realize the sensor has moved.

---

## 2. Task Breakdown & Deliverables

### Task 1: Core Dynamic Graph Engineering (Architectural Overhaul)
**Objective:** Replace the rigid Boolean skeleton with a fluid, continuously learning probability matrix capable of re-routing sensor connections on the fly.
*   **Subtask 1.1 (Formula Implementation):** Code the 3-part Residual Adaptive Graph formula: $A_{final} = (\alpha \times A_{phys}) + (\beta \times A_{learn}) + (\gamma \times A_{dyn}(X))$.
*   **Subtask 1.2 (Dense Message Passing):** Bypass the PyTorch Geometric (PyG) sparse `edge_index` limitation by engineering a custom PyTorch module that uses batched matrix multiplication (`torch.bmm`) to pass features through dense, continuous probability decimals.
*   **Subtask 1.3 (Gradient Stabilization):** Implement Temperature Scaling ($\sqrt{d_k}$) and Activation Guarding (`torch.tanh`) before the attention Softmax to prevent gradient explosions (NaN loss) caused by high-frequency IMU noise.
*   **Deliverable:** 
    *   Updated `models/mamba_whar.py` containing the fully functional `ResidualDynamicGraph` module.

### Task 2: Mutual Displacement Stress-Testing
**Objective:** Empirically prove that the dynamic graph prevents model failure when sensors are physically strapped to the wrong limbs.
*   **Subtask 2.1 (Data Extraction):** Modify the data processing pipeline to parse and normalize the REALDISP "Mutual Displacement" logs alongside the "Ideal" logs.
*   **Subtask 2.2 (Stress-Test Script):** Write a standalone evaluation script that loads weights trained strictly on *Ideal* data, but evaluates them purely on *Displaced* data.
*   **Deliverable:** 
    *   Updated `process_all_data.py`.
    *   New script: `test_displacement.py`.
    *   Quantitative metric: A recorded F1-score drop (Ideal vs. Mutual) that establishes a new state-of-the-art compared to legacy baselines.

### Task 3: Interpretability & Attention Visualization
**Objective:** Visually prove to the evaluation committee that the mathematical network is actually changing its internal shape in response to displaced sensors.
*   **Subtask 3.1 (Forward Hooking):** Extract the real-time continuous attention matrix ($A_{dyn}$) from the network during the forward pass.
*   **Subtask 3.2 (Heatmap Generation):** Write a plotting script that maps the $N \times N$ matrix into a color-coded heatmap, showing which sensors are strongly attending to which body parts.
*   **Deliverable:** 
    *   New script: `plot_attention.py`.
    *   Visual artifact: Heatmap images (e.g., `.png`) demonstrating the connections shifting when displacement occurs.

### Task 4: Component Ablation Study
**Objective:** Mathematically justify why the complex 3-part residual equation ($\alpha, \beta, \gamma$) is necessary, rather than just using pure attention.
*   **Subtask 4.1 (Toggle Logic):** Add command-line arguments in the training script to selectively disable the Physical Anchor ($\alpha=0$), the Learnable Priors ($\beta=0$), or the Dynamic Attention ($\gamma=0$).
*   **Subtask 4.2 (Run Ablations):** Train the model under each disabled configuration and document the Epoch 1 stability and final convergence accuracy.
*   **Deliverable:** 
    *   Updated `main.py` with ablation flags.
    *   Quantitative metric: An Ablation Table proving that removing the physical anchor causes early-epoch collapse, and removing dynamic attention causes failure on displaced data.

---

## 3. Final MTP-1 Reporting (End-of-Semester)
*Note: This is to be compiled after all coding tasks are complete.*
*   **Deliverable:** The Mid-Term Thesis Report containing:
    *   The methodology of Dense Message Passing.
    *   The Robustness Degradation tables (Hardware Failures & Mutual Displacements).
    *   The IEEE-style Convergence graphs and Attention Heatmaps.