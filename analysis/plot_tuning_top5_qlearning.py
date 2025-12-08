import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    os.makedirs("runs/plots", exist_ok=True)
    df = pd.read_csv("runs/csv/qlearning_tuning_full.csv")

    grouped = df.groupby("config_id")["reward"].mean().sort_values(ascending=False)
    top5 = grouped.head(5)

    plt.figure(figsize=(12, 6))
    plt.bar(top5.index, top5.values, color="orange")
    plt.title("Top 5 Q-Learning Tuning Configurations")
    plt.ylabel("Mean Reward")
    plt.xlabel("Config ID")
    plt.grid(True, alpha=0.3)

    plt.savefig("runs/plots/qlearning_top5_configs.png", dpi=200)
    print("Saved qlearning_top5_configs.png")

if __name__ == "__main__":
    main()
