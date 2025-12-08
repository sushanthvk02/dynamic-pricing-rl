import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    os.makedirs("runs/plots", exist_ok=True)

    df_q = pd.read_csv("runs/csv/qlearning_rewards.csv")
    df_s = pd.read_csv("runs/csv/sarsa_rewards.csv")

    Q_q = np.load("runs/models/qlearning_Q.npy")
    Q_s = np.load("runs/models/sarsa_Q.npy")

    q_actions = np.argmax(Q_q, axis=2).flatten()
    s_actions = np.argmax(Q_s, axis=2).flatten()

    plt.figure(figsize=(12, 6))
    bins = np.arange(q_actions.min(), q_actions.max() + 2)

    plt.hist(q_actions, bins=bins, rwidth=0.45, label="Q-Learning", alpha=0.7)
    plt.hist(s_actions, bins=bins, rwidth=0.45, label="SARSA", alpha=0.7)

    plt.xlabel("Action Index")
    plt.ylabel("Frequency")
    plt.title("Action Distribution — Q-Learning vs SARSA")
    plt.legend()

    plt.savefig("runs/plots/action_histograms.png", dpi=200)
    print("Saved action_histograms.png")

if __name__ == "__main__":
    main()
