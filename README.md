# NeuroSearch 🧠🔍

NeuroSearch is a PyTorch-based **Neural Architecture Search (NAS)** framework using Reinforcement Learning. It automates the process of designing high-performing, lightweight deep learning networks by training an RNN controller to discover optimal cell topologies.

---

## 🏛️ Project Architecture

```
nas_rl/
├── __init__.py
├── controller/
│   ├── __init__.py
│   ├── baseline.py          # Exponential moving average baseline for RL
│   ├── lstm_controller.py   # LSTM RNN agent predicting edge-operations
│   └── test_controller.py   # Unit tests for controller and baseline
├── search_space/
│   ├── __init__.py
│   ├── cell.py              # Directed Acyclic Graph (DAG) cell structure
│   ├── ops.py               # Candidate neural network operations (convolutions, pooling, etc.)
│   ├── supernet.py          # Stacks cells to form the complete weight-sharing network
│   ├── test_cell_supernet.py# Unit tests for cells and supernet
│   └── test_ops.py          # Unit tests for individual operations
├── data.py                  # Data loader configurations
├── export/                  # Exporting architectures to clean models
├── reward/                  # FLOPs and latency cost estimations
├── search/                  # Orchestration of the NAS search loop
└── trainer/                 # Supernet and candidate training loops
```

---

## 🧩 Components Explained

### 1. Search Space (`nas_rl/search_space/`)
* **`ops.py`**: Defines candidate operations like standard `ConvBNReLU`, Depthwise Separable `SepConv`, `Identity` skip connections, and `Zero` (no-connection) paths.
* **`cell.py`**: Constructs a DAG cell structure. Cells can be **normal** (stride 1) or **reduction** (stride 2 to downsample resolution). A cell takes outputs of the previous two cells and routes them through edges based on choices from the controller.
* **`supernet.py`**: The complete weight-sharing neural network. Stacks normal and reduction cells. It can run forward passes using any arbitrary architecture spec.

### 2. Controller & RL (`nas_rl/controller/`)
* **`lstm_controller.py`**: A recurrent neural network that sequentially predicts which operation to put on each edge of each cell. It outputs the architecture blueprint, action log probabilities, and entropy.
* **`baseline.py`**: Implements an `ExponentialBaseline` tracking moving average rewards. Subtracting the baseline from actual rewards reduces RL policy gradient variance.

---

## 🚦 How to Run Sanity Checks

Before starting the search loop, run the unit test scripts to verify the correctness of the tensor dimensions, mapping indices, and policy gradient backwards pass.

```bash
# Test individual operations shapes
python3 nas_rl/search_space/test_ops.py

# Test normal and reduction cell routing and supernet dimension compatibility
python3 nas_rl/search_space/test_cell_supernet.py

# Test controller recurrent sampling, greedy choice, and policy gradient backward pass
python3 nas_rl/controller/test_controller.py
```

---

## 🎯 Roadmap
- [ ] Implement `cost_estimator.py` to calculate FLOPs and parameter sizes.
- [ ] Implement `reward_combiner.py` to integrate model accuracy and efficiency constraints.
- [ ] Implement `supernet_trainer.py` to pretrain the shared supernet weights.
- [ ] Implement `search_loop.py` to run the active controller search.
