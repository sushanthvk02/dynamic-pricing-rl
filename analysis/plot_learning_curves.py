import os, glob
import pandas as pd
import matplotlib.pyplot as plt

def main(window=300):
    os.makedirs("runs/plots", exist_ok=True)
    csv_paths = glob.glob("runs/csv/*_rewards.csv")
    if not csv_paths:
        print("No CSVs in runs/csv/. Run a training script first.")
        return

    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

    plt.figure()
    for algo, sub in df.groupby("algo"):
        sub = sub.sort_values("episode")
        roll = sub["reward"].rolling(window, min_periods=1).mean()
        ls = "--" if "(eval" in algo else "-"
        plt.plot(sub["episode"], roll, linestyle=ls, label=algo)

    plt.xlabel("Episode")
    plt.ylabel(f"Episodic Reward (rolling mean, w={window})")
    plt.legend()
    plt.tight_layout()
    out = "runs/plots/learning_curves.png"
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
