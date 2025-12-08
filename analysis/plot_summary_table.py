import pandas as pd
import numpy as np
import os

def summarize(name, rewards):
    return {
        "Agent": name,
        "Mean Reward": rewards.mean(),
        "Std Reward": rewards.std(),
        "Median": rewards.median(),
        "Best Episode": rewards.max(),
        "Worst Episode": rewards.min()
    }

def main():
    os.makedirs("runs/tables", exist_ok=True)

    df_base = pd.read_csv("runs/csv/baselines_rewards.csv")
    df_q = pd.read_csv("runs/csv/qlearning_rewards.csv")
    df_s = pd.read_csv("runs/csv/sarsa_rewards.csv")
    df_tune_q = pd.read_csv("runs/csv/qlearning_tuning_full.csv")
    df_tune_s = pd.read_csv("runs/csv/sarsa_tuning_full.csv")

    # Compute best tuning config performance
    best_q = df_tune_q.groupby("config_id")["reward"].mean().max()
    best_s = df_tune_s.groupby("config_id")["reward"].mean().max()

    rows = [
        summarize("Random", df_base[df_base["algo"]=="Random"]["reward"]),
        summarize("Fixed", df_base[df_base["algo"]=="Fixed"]["reward"]),
        summarize("Q-Learning Train", df_q[df_q["algo"]=="Q-Learning"]["reward"]),
        summarize("SARSA Train", df_s[df_s["algo"]=="SARSA"]["reward"]),
        summarize("Q-Learning Tune (best)", pd.Series([best_q])),
        summarize("SARSA Tune (best)", pd.Series([best_s])),
    ]

    table = pd.DataFrame(rows)
    table.to_csv("runs/tables/summary_table.csv", index=False)

    print("\n==== SUMMARY TABLE ====\n")
    print(table.to_string(index=False))

    with open("runs/tables/summary_table.txt", "w") as f:
        f.write(table.to_string(index=False))

    print("\nSaved summary_table.csv and summary_table.txt")

if __name__ == "__main__":
    main()
