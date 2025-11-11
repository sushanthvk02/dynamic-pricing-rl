# Dynamic Pricing using Reinforcement Learning

This repository implements a reinforcement learning (RL) framework for **Dynamic Pricing** — modeling how an agent learns to set product prices dynamically to maximize long-term revenue under demand uncertainty and limited inventory.

Developed as part of **CSC 496 – Principles of Machine Learning (Fall 2025)** at the **University of Arizona**.

---

## Overview
In real-world retail and e-commerce, pricing decisions must balance profit per item with sales volume. Setting a high price reduces demand but increases per-unit profit; a low price increases sales but risks stockouts.  
This project frames that trade-off as a **sequential decision-making problem** using reinforcement learning.

The environment simulates:
- A product with finite inventory.
- Price-sensitive stochastic demand.
- A finite selling horizon (daily decisions).
- Revenue minus holding cost as the reward signal.

Agents learn optimal pricing strategies by interacting with this environment.

---

## Project Structure

dynamic-pricing-rl/
│
├── envs/
│ ├── pricing_env.py # Custom Gymnasium environment
│ └── smoke_test.py # Quick test to verify environment behavior
│
├── agents/
│ ├── train_random_fixed.py # Baseline random/fixed price agents
│ ├── train_qlearning.py # Tabular Q-Learning agent
│ └── tune_qlearning.py # Hyperparameter sweep and evaluation
│
├── analysis/
│ ├── plot_learning_curves.py # Learning curve visualization
│ └── plot_compare_best_q_vs_baselines.py # Compare Q-Learning vs baselines
│
├── runs/
│ ├── csv/ # Summary metrics (small CSVs kept in Git)
│ ├── plots/ # Training and comparison plots 
│ ├── models/ # Trained models (ignored in Git)
│
├── requirements.txt
└── README.me

---

## Setup and Usage

1. Clone the repository
git clone https://github.com/sushanthvk02/dynamic-pricing-rl.git
cd dynamic-pricing-rl

2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Test the environment
python -m envs.smoke_test

5. Run baseline agents
python -m agents.train_random_fixed

6. Train Q-Learning agent
python -m agents.train_qlearning --episodes 20000

7. Hyperparameter sweep
python -m agents.tune_qlearning

8. Generate plots
python -m analysis.plot_learning_curves
python -m analysis.plot_compare_best_q_vs_baselines


Mid-Project Progress:
Implemented:

DynamicPricingEnv Gym environment
Baseline agents (Random, Fixed)
Tabular Q-Learning with ε- and α-decay schedules
Evaluation mode for greedy policy
Hyperparameter tuning (grid search)
Learning curve visualization and Q-learning vs baseline comparison\

Next Steps:
Implement SARSA for on-policy learning comparison
Add PPO (Stable-Baselines3) as a deep RL extension
Experiment with demand elasticity and cost structures
Generate final performance plots for the report

References
Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press
Chen, L., & Simchi-Levi, D. (2022). Dynamic Pricing and Learning with Finite Inventories Gymnasium Documentation

👨‍💻 Author
Viswa Sushanth Karuturi
University of Arizona
B.S. Computer Science & B.S. Statistics and Data Science
