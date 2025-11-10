import os, glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    os.makedirs("runs/plots", exist_ok=True)

    csv_paths = glob.glob("runs/csv/*_rewards.csv")
    if not csv_paths:
        print("No CSVs in runs/csv/. Run a training script first.")
        return

    df_list = []
    for p in csv_paths:
        try:
            df_list.append(pd.read_csv(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")

    df = pd.concat(df_list, ignore_index=True)

    # Plot rolling mean reward by algorithm (smooths noise)
    plt.figure()
    for algo, sub in df.groupby("algo"):
        sub = sub.sort_values("episode")
        roll = sub["reward"].rolling(20, min_periods=1).mean()
        plt.plot(sub["episode"], roll, label=algo)

    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward (rolling mean)")
    plt.legend()
    plt.tight_layout()
    out = "runs/plots/learning_curves.png"
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
