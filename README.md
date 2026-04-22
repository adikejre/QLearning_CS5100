# Empirical Validation of Q-Learning Convergence

**CS5100 — Foundations of Artificial Intelligence, Spring 2026**

## Aditya Kejrewal

## Overview

This project empirically validates the convergence theorem from Watkins & Dayan's 1992 paper "Q-Learning." The paper proves mathematically that Q-learning converges to optimal action-values with probability 1 under specified conditions, but contains no experiments. This project fills that gap by implementing Q-learning from scratch, computing ground truth Q* via value iteration, and designing controlled experiments to test each condition of the theorem.

## Paper

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.

## Project Structure

```
├── capstone.py              # Main implementation (all algorithms + experiments)
├── plots/                   # Generated experiment plots
│   ├── exp1_convergence.png
│   ├── exp2_learning_rates.png
│   ├── exp3_exploration.png
│   ├── exp4_discount.png
│   ├── exp5_ql_vs_sarsa.png
│   ├── exp6_frozenlake.png
│   ├── exp7_taxi.png
│   ├── q_value_heatmap.png
│   └── learned_policy.png
├── report.pdf               # Final report
└── README.md
```

## Algorithms Implemented (from scratch)

- **Value Iteration** — Dynamic programming, computes exact Q* as ground truth
- **Tabular Q-Learning** — The paper's algorithm with configurable learning rate and exploration
- **SARSA** — On-policy TD control for comparison
- All implementations use only NumPy. No RL libraries.

## Environments

- **GridWorld 4x4** (custom) — Deterministic, core convergence testing
- **FrozenLake 4x4** (Gymnasium) — Stochastic transitions
- **CliffWalking** (Gymnasium) — Q-learning vs SARSA comparison
- **Taxi-v3** (Gymnasium) — Scalability test (500 states)

## Experiments

| # | Experiment | Purpose |
|---|---|---|
| 1 | Core Convergence | Validate Q → Q* under proper conditions |
| 2 | Learning Rate | Test effect of decaying vs fixed α |
| 3 | Exploration | Test effect of ε-greedy vs pure greedy |
| 4 | Discount Factor | Compare convergence across γ values |
| 5 | Q-Learning vs SARSA | Off-policy vs on-policy on CliffWalking |
| 6 | FrozenLake | Convergence in stochastic environment |
| 7 | Taxi-v3 | Scalability to 500 states |

## How to Run

```bash
pip install numpy matplotlib gymnasium
python capstone.py
```

All experiments run sequentially. Total runtime: ~10 minutes. Plots are saved as PNG files.

## Key Results

- **Convergence confirmed:** Mean Q-value error drops to 0.003, learned policy is 100% optimal
- **Conditions matter:** Violating learning rate decay or exploration causes measurably worse outcomes
- **SARSA vs Q-learning:** SARSA learns safer path on CliffWalking due to on-policy updates
- **Scalability:** Q-learning successfully learns on Taxi-v3 (500 states)
