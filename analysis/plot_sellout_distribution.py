import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from envs.pricing_env import DynamicPricingEnv
from agents.train_qlearning import build_edges, obs_to_state, idx_to_price


def compute_sellout_days_tabular(Q, env, inv_edges, price_edges, episodes=200):
    """Runs a greedy rollout using Q-table and records sell-out days."""
    sell_days = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            inv_bin, day_idx = obs_to_state(obs, env.horizon_days, inv_edges)
            a_idx = int(np.argmax(Q[inv_bin, day_idx]))
            action = idx_to_price(a_idx, price_edges)
            obs, r, done, _, info = env.step(action)
        sell_days.append(info["episode"]["l"])
    return sell_days


def compute_sellout_days_random(env, episodes=200):
    days = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, done, _, info = env.step(action)
        days.append(info["episode"]["l"])
    return days


def compute_sellout_days_fixed(env, fixed_price, episodes=200):
    days = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, r, done, _, info = env.step([fixed_price])
        days.append(info["episode"]["l"])
    return days


def main():
    os.makedirs("runs/plots", exist_ok=True)

    # Load baseline to extract BEST fixed price 
    df_base = pd.read_csv("runs/csv/baselines_rewards.csv")
    fixed_price = df_base[df_base["algo"] == "Fixed"]["reward"].mean() 

    env = DynamicPricingEnv(seed=0)
    fixed_price = (env.price_min + env.price_max) / 2.0

    # Load Q-learning + SARSA
    Q_q = np.load("runs/models/qlearning_Q.npy")
    Q_s = np.load("runs/models/sarsa_Q.npy")

    INV_EDGES, PRICE_EDGES = build_edges(env)

    # Compute sell-out day histograms
    random_days = compute_sellout_days_random(env)
    fixed_days = compute_sellout_days_fixed(env, fixed_price)
    q_days = compute_sellout_days_tabular(Q_q, env, INV_EDGES, PRICE_EDGES)
    sarsa_days = compute_sellout_days_tabular(Q_s, env, INV_EDGES, PRICE_EDGES)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.hist(random_days, bins=20, alpha=0.5, label="Random")
    plt.hist(fixed_days, bins=20, alpha=0.5, label="Fixed")
    plt.hist(q_days, bins=20, alpha=0.5, label="Q-Learning")
    plt.hist(sarsa_days, bins=20, alpha=0.5, label="SARSA")

    plt.xlabel("Sell-Out Day")
    plt.ylabel("Frequency")
    plt.title("Sell-Out Day Distribution Across Policies")
    plt.legend()

    plt.savefig("runs/plots/sellout_days.png", dpi=200)
    print("Saved sellsout_days.png")


if __name__ == "__main__":
    main()
