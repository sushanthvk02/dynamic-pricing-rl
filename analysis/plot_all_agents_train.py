# plot_all_agents_train.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(x, w=300):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w), "valid") / w

def main():
    os.makedirs("runs/plots", exist_ok=True)

    df_base = pd.read_csv("runs/csv/baselines_rewards.csv")
    df_q = pd.read_csv("runs/csv/qlearning_rewards.csv")
    df_s = pd.read_csv("runs/csv/sarsa_rewards.csv")

    df_rand = df_base[df_base["algo"] == "Random"]
    df_fixed = df_base[df_base["algo"] == "Fixed"]
    best_fixed = df_fixed["reward"].mean()

    plt.figure(figsize=(16, 8))

    plt.plot(moving_average(df_rand["reward"]), label="Random", alpha=0.7)
    plt.plot(moving_average(df_fixed["reward"]), label="Fixed Baseline", alpha=0.7)
    plt.plot(moving_average(df_q[df_q["algo"] == "Q-Learning"]["reward"]), label="Q-Learning", alpha=0.9)
    plt.plot(moving_average(df_s[df_s["algo"] == "SARSA"]["reward"]), label="SARSA", alpha=0.9)

    plt.axhline(best_fixed, color="black", linestyle="--", label=f"Best Fixed ({best_fixed:.0f})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curves — Random vs Fixed vs Q-Learning vs SARSA")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.savefig("runs/plots/all_agents_train.png", dpi=200)
    print("Saved runs/plots/all_agents_train.png")

if __name__ == "__main__":
    main()
