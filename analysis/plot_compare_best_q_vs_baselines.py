import os, json
import pandas as pd
import matplotlib.pyplot as plt

def main(window=300):
    os.makedirs("runs/plots", exist_ok=True)
    base = pd.read_csv("runs/csv/baselines_rewards.csv")
    sweep = pd.read_csv("runs/csv/qlearning_sweep.csv")
    summ  = pd.read_csv("runs/csv/qlearning_sweep_summary.csv").sort_values("mean_lastN", ascending=False)
    best_id = summ.iloc[0]["config_id"]
    qbest = sweep[sweep["config_id"] == best_id]

    plt.figure()

    for algo, sub in base.groupby("algo"):
        sub = sub.sort_values("episode")
        plt.plot(sub["episode"], sub["reward"].rolling(window, min_periods=1).mean(), label=algo)

    q_mean = qbest.groupby("episode")["reward"].mean().sort_index()
    plt.plot(q_mean.index, q_mean.rolling(window, min_periods=1).mean(), label=f"Q-Learning[{best_id}]")

    plt.xlabel("Episode"); plt.ylabel(f"Episodic Reward (rolling mean, w={window})")
    plt.legend(); plt.tight_layout()
    out = "runs/plots/best_q_vs_baselines.png"
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
