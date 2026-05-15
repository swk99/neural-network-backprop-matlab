# Neural Network Backpropagation from Scratch in MATLAB

A hand-coded implementation of a feedforward neural network with backpropagation in MATLAB — no toolboxes, no `trainlm`, no abstractions. Weights are updated manually per pattern using the delta rule, making the learning dynamics fully transparent.

---

## Network Architecture

```
Input layer      Hidden layer       Output layer
   x1  ──w1──► h1 (sigmoid)
   x1  ──w3──► h2 (sigmoid) ──w9, w8, w7──► y (linear)
   x2  ──w4──► h2
   x2  ──w6──► h3 (sigmoid)
   x1  ──w2──────────────────────────────► y (direct)
   x2  ──w5──────────────────────────────► y (direct)
```

- **Inputs:** 2 (`x1`, `x2`)
- **Hidden units:** 3 (sigmoid activation)
- **Output:** 1 (linear — no activation function)
- **Connections:** mixed architecture with both direct input→output weights and input→hidden→output paths

---

## Algorithm

Online (pattern-by-pattern) backpropagation over 100 epochs:

**Forward pass:**
```
h1 = σ(w1·x1)
h2 = σ(w3·x1 + w4·x2)
h3 = σ(w6·x2)
y  = w2·x1 + w5·x2 + w9·h1 + w8·h2 + w7·h3
```

**Error signal:**
```
β_out = t − y
```

**Output layer weight updates:**
```
w ← w + η · β_out · (input to that weight)
```

**Hidden layer deltas (chain rule):**
```
β_k = h_k(1 − h_k) · β_out · w_k_out
```

**Hidden layer weight updates:**
```
w ← w + η · β_k · (input to that weight)
```

All inputs, targets, learning rate, and initial weights are entered interactively at runtime.

---

## Usage

**Requirements:** MATLAB (any recent version; no additional toolboxes needed)

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/neural-network-backprop-matlab.git
   ```

2. Open `Coursework1_Seonwoo_Kim.m` in MATLAB and run.

3. Follow the interactive prompts:
   ```
   Numbers of Input : 4          % number of training patterns
   Enter input:
   Pattern 1
    x1 = 0
    x2 = 0
   ...
   Enter Learning rate (eta): 0.1
   Enter initial weights:
   w1 = 0.1
   ...
   ```

4. The script prints hidden outputs, network output `y`, error `β_out`, and all updated weights after every pattern in every epoch.

---

## Example (XOR-like problem)

| Pattern | x1 | x2 | Target |
|---------|----|----|--------|
| 1 | 0 | 0 | 0 |
| 2 | 0 | 1 | 1 |
| 3 | 1 | 0 | 1 |
| 4 | 1 | 1 | 0 |

Run with `eta = 0.1` and small random initial weights to observe weight convergence over 100 epochs.

---

## Repository Structure

```
neural-network-backprop-matlab/
├── Coursework1_Seonwoo_Kim.m   # Full implementation
└── README.md                   # This file
```
