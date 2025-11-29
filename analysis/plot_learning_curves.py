import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def rolling_mean(x, w=300):
    return np.convolve(x, np.ones(w)/w, mode="same")

def load_csv(path):
    if not os.path.exists(path):
        print(f"[Warning] Missing file: {path}")
        return None
    return pd.read_csv(path)

def plot_learning_curves():
    plt.figure(figsize=(12, 7))

    # Load all logs
    df_baselines = load_csv("runs/csv/baselines_rewards.csv")
    df_q = load_csv("runs/csv/qlearning_rewards.csv")
    df_sarsa = load_csv("runs/csv/sarsa_rewards.csv")

    # Fixed
    if df_baselines is not None:
        fixed = df_baselines[df_baselines["algo"] == "Fixed"]["reward"].values
        plt.plot(rolling_mean(fixed), label="Fixed", color="blue")

    # Random
    if df_baselines is not None:
        rnd = df_baselines[df_baselines["algo"] == "Random"]["reward"].values
        plt.plot(rolling_mean(rnd), label="Random", color="red")

    # Q-Learning
    if df_q is not None:
        ql = df_q[df_q["algo"] == "Q-Learning"]["reward"].values
        plt.plot(rolling_mean(ql), label="Q-Learning", color="orange")

        ql_eval = df_q[df_q["algo"] == "Q-Learning (eval, greedy)"]["reward"].values
        if len(ql_eval) > 0:
            plt.plot(rolling_mean(ql_eval), "--", label="Q-Learning (eval)", color="orange", alpha=0.7)

    # SARSA
    if df_sarsa is not None:
        sa = df_sarsa[df_sarsa["algo"] == "SARSA"]["reward"].values
        plt.plot(rolling_mean(sa), label="SARSA", color="green")

        sa_eval = df_sarsa[df_sarsa["algo"] == "SARSA (eval, greedy)"]["reward"].values
        if len(sa_eval) > 0:
            plt.plot(rolling_mean(sa_eval), "--", label="SARSA (eval)", color="green", alpha=0.7)

    plt.title("Learning Curves (Rolling Mean, window = 300)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("runs/plots", exist_ok=True)
    plt.savefig("runs/plots/learning_curves_all.png")
    print("Saved plot to runs/plots/learning_curves_all.png")


if __name__ == "__main__":
    plot_learning_curves()
