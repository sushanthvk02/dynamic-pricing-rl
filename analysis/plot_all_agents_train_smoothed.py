import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(x, k=300):
    return np.convolve(x, np.ones(k)/k, mode="same")

def main():
    os.makedirs("runs/plots", exist_ok=True)

    df_base = pd.read_csv("runs/csv/baselines_rewards.csv")
    df_q = pd.read_csv("runs/csv/qlearning_rewards.csv")
    df_s = pd.read_csv("runs/csv/sarsa_rewards.csv")

    random = df_base[df_base["algo"] == "Random"]["reward"].values
    fixed = df_base[df_base["algo"] == "Fixed"]["reward"].values
    q = df_q[df_q["algo"] == "Q-Learning"]["reward"].values
    sarsa = df_s[df_s["algo"] == "SARSA"]["reward"].values

    best_fixed = fixed.mean()

    plt.figure(figsize=(18, 8))
    plt.plot(smooth(random), label="Random (Smoothed)", alpha=0.8)
    plt.plot(smooth(fixed), label="Fixed Baseline (Smoothed)", alpha=0.8)
    plt.plot(smooth(q), label="Q-Learning (Smoothed)", alpha=0.9)
    plt.plot(smooth(sarsa), label="SARSA (Smoothed)", alpha=0.9)

    plt.axhline(best_fixed, linestyle="--", color="black",
                label=f"Best Fixed ({best_fixed:.0f})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Smoothed Learning Curves — Random vs Fixed vs Q-Learning vs SARSA")
    plt.legend()
    plt.grid(True)

    plt.savefig("runs/plots/all_agents_train_smoothed.png", dpi=200)
    print("Saved all_agents_train_smoothed.png")

if __name__ == "__main__":
    main()
