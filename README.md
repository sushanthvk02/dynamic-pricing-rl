# Dynamic Pricing using Reinforcement Learning

This repository implements a reinforcement learning (RL) framework for **Dynamic Pricing**, where an agent learns to set product prices dynamically to maximize long-term revenue under demand uncertainty and limited inventory.

Developed as part of **CSC 496 – Principles of Machine Learning (Fall 2025)** at the **University of Arizona**.

---

## Overview

In real-world retail and e-commerce, pricing decisions must balance **profit per item** with **sales volume**. Setting a high price increases profit per sale but reduces demand, while setting a low price increases sales but risks stockouts.  

This project frames that trade-off as a **sequential decision-making problem** using reinforcement learning.

The environment simulates:
- A product with finite inventory.
- Price-sensitive stochastic demand.
- A fixed selling horizon (daily decisions).
- Revenue minus holding cost as the reward.

Agents interact with this environment to learn pricing strategies that maximize total profit.

---

## Project Structure

dynamic-pricing-rl/
│
├── envs/
│ ├── pricing_env.py # Custom Gymnasium environment
│ └── smoke_test.py # Quick environment test
│
├── agents/
│ ├── train_random_fixed.py # Baseline random and fixed-price agents
│ ├── train_qlearning.py # Tabular Q-Learning implementation
│ └── tune_qlearning.py # Hyperparameter tuning and evaluation
│
├── analysis/
│ ├── plot_learning_curves.py # Training curve visualization
│ └── plot_compare_best_q_vs_baselines.py # Compare Q-Learning vs baselines
│
├── runs/
│ ├── csv/ # Summary metrics (small CSVs kept)
│ ├── plots/ # Plots and visualizations
│ └── models/ # Trained models (ignored in Git)
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## Setup and Usage

### 1. Clone the repository
```bash
git clone https://github.com/sushanthvk02/dynamic-pricing-rl.git
cd dynamic-pricing-rl
2. Create and activate a virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate    # On Windows
# source venv/bin/activate   # On Mac/Linux
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Test the environment
bash
Copy code
python -m envs.smoke_test
5. Run baseline agents
bash
Copy code
python -m agents.train_random_fixed
6. Train Q-Learning agent
bash
Copy code
python -m agents.train_qlearning --episodes 20000
7. Run hyperparameter tuning
bash
Copy code
python -m agents.tune_qlearning
8. Generate plots
bash
Copy code
python -m analysis.plot_learning_curves
python -m analysis.plot_compare_best_q_vs_baselines
Mid-Project Progress
Implemented
DynamicPricingEnv Gym environment

Baseline agents: Random and Fixed-price strategies

Tabular Q-Learning with epsilon and learning-rate decay

Greedy evaluation for trained Q-table

Hyperparameter tuning and experiment tracking

Learning curve visualization and Q-Learning vs Baseline comparison

Next Steps
Implement SARSA (on-policy comparison)

Add PPO from Stable-Baselines3 (deep RL extension)

Experiment with demand elasticity and cost structure variations

Generate final comparison plots and analysis for report

References
Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.

Chen, L., & Simchi-Levi, D. (2022). Dynamic Pricing and Learning with Finite Inventories.

Gymnasium Documentation – https://gymnasium.farama.org/

Author
Viswa Sushanth Karuturi
University of Arizona
B.S. Computer Science & B.S. Statistics and Data Science