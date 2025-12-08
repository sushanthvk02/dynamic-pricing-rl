import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(x, k=500):
    return np.convolve(x, np.ones(k)/k, mode="same")

def main():
    os.makedirs("runs/plots", exist_ok=True)

    df_train = pd.read_csv("runs/csv/sarsa_rewards.csv")
    df_tune = pd.read_csv("runs/csv/sarsa_tuning_full.csv")

    train_curve = df_train[df_train["algo"] == "SARSA"]["reward"].values

    best_id = df_tune.groupby("config_id")["reward"].mean().idxmax()
    df_best = df_tune[df_tune["config_id"] == best_id]

    tune_curve = df_best["reward"].values

    plt.figure(figsize=(18, 8))
    plt.plot(smooth(train_curve), label="SARSA Train (Smoothed)")
    plt.plot(smooth(tune_curve), label=f"SARSA Tune Best ({best_id}) (Smoothed)")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("SARSA — Train vs Tune (Smoothed Comparison)")
    plt.legend()
    plt.grid(True)

    plt.savefig("runs/plots/sarsa_train_vs_tune_smoothed.png", dpi=200)
    print("Saved sarsa_train_vs_tune_smoothed.png")

if __name__ == "__main__":
    main()
