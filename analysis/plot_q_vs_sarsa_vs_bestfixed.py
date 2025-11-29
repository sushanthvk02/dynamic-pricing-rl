import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def rolling_mean(x, w=300):
    return np.convolve(x, np.ones(w) / w, mode="same")

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"Warning: missing file {path}")
    return None

def main():
    df_q = load_csv("runs/csv/qlearning_rewards.csv")
    df_sarsa = load_csv("runs/csv/sarsa_rewards.csv")
    df_fixed = load_csv("runs/csv/best_fixed_price_sweep.csv")

    plt.figure(figsize=(12, 7))

    # Best fixed baseline
    if df_fixed is not None:
        best_row = df_fixed.loc[df_fixed["mean_reward"].idxmax()]
        best_reward = best_row["mean_reward"]
        best_price = best_row["price"]

        plt.axhline(best_reward, color="red", linestyle="--",
                    label=f"Best Fixed Price = {best_price:.2f}")

    # Q-Learning
    if df_q is not None:
        ql = df_q[df_q["algo"] == "Q-Learning"]["reward"].values
        plt.plot(rolling_mean(ql), label="Q-Learning", color="orange")

    # SARSA
    if df_sarsa is not None:
        sa = df_sarsa[df_sarsa["algo"] == "SARSA"]["reward"].values
        plt.plot(rolling_mean(sa), label="SARSA", color="green")

    plt.title("Q-Learning vs SARSA vs Best Fixed Price")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("runs/plots", exist_ok=True)
    plt.savefig("runs/plots/q_vs_sarsa_vs_bestfixed.png")
    print("Saved plot to runs/plots/q_vs_sarsa_vs_bestfixed.png")


if __name__ == "__main__":
    main()
