import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    os.makedirs("runs/plots", exist_ok=True)

    df_base = pd.read_csv("runs/csv/baselines_rewards.csv")
    df_q = pd.read_csv("runs/csv/qlearning_rewards.csv")
    df_s = pd.read_csv("runs/csv/sarsa_rewards.csv")
    df_tune_q = pd.read_csv("runs/csv/qlearning_tuning_full.csv")
    df_tune_s = pd.read_csv("runs/csv/sarsa_tuning_full.csv")


    best_q_id = df_tune_q.groupby("config_id")["reward"].mean().idxmax()
    best_s_id = df_tune_s.groupby("config_id")["reward"].mean().idxmax()

    best_q_rewards = df_tune_q[df_tune_q["config_id"] == best_q_id]["reward"]
    best_s_rewards = df_tune_s[df_tune_s["config_id"] == best_s_id]["reward"]

    labels = [
        "Random",
        "Fixed",
        "Q-Learning Train",
        "SARSA Train",
        f"Q-Learning Tune ({best_q_id})",
        f"SARSA Tune ({best_s_id})",
    ]

    data = [
        df_base[df_base["algo"] == "Random"]["reward"],
        df_base[df_base["algo"] == "Fixed"]["reward"],
        df_q[df_q["algo"] == "Q-Learning"]["reward"],
        df_s[df_s["algo"] == "SARSA"]["reward"],
        best_q_rewards,
        best_s_rewards,
    ]

    plt.figure(figsize=(16, 8))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.ylabel("Reward")
    plt.title("Reward Distribution Across Agents")
    plt.grid(True, alpha=0.3)

    plt.savefig("runs/plots/reward_distributions.png", dpi=200)
    print("Saved reward_distributions.png")

if __name__ == "__main__":
    main()
