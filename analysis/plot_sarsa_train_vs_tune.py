import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(x, w=300):
    return np.convolve(x, np.ones(w), 'same') / w

def main():
    os.makedirs("runs/plots", exist_ok=True)

    df_train = pd.read_csv("runs/csv/sarsa_rewards.csv")
    df_tune = pd.read_csv("runs/csv/sarsa_tuning_full.csv")
    df_summary = pd.read_csv("runs/csv/sarsa_tuning_summary.csv")

    # Best tuning config
    best_id = df_summary.sort_values("mean_lastN", ascending=False).iloc[0]["config_id"]
    df_best = df_tune[df_tune["config_id"] == best_id]

    # Normalize tuning episodes
    df_best = df_best.copy()
    df_best["episode_norm"] = np.linspace(0, len(df_train), len(df_best))

    plt.figure(figsize=(16, 8))

    plt.plot(df_train["episode"],
             moving_average(df_train["reward"], 300),
             label="SARSA (Train)", color="green")

    plt.plot(df_best["episode_norm"],
             moving_average(df_best["reward"], 300),
             label=f"SARSA (Tune Best Config: {best_id})",
             color="orange", alpha=0.8)

    plt.title("SARSA — Train vs Tune (Normalized Comparison)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)

    out = "runs/plots/sarsa_train_vs_tune.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
