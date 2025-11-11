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

References
Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
Chen, L., & Simchi-Levi, D. (2022). Dynamic Pricing and Learning with Finite Inventories.

Gymnasium Documentation – https://gymnasium.farama.org/

Author
Viswa Sushanth Karuturi
University of Arizona
B.S. Computer Science & B.S. Statistics and Data Science