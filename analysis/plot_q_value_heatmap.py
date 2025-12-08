import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    os.makedirs("runs/plots", exist_ok=True)

    Q = np.load("runs/models/qlearning_Q.npy")
    Q_mean = Q.mean(axis=2)

    plt.figure(figsize=(12, 6))
    plt.imshow(Q_mean, cmap="plasma", aspect="auto", origin="lower")
    plt.colorbar(label="Mean Q-value")
    plt.xlabel("Day")
    plt.ylabel("Inventory Bin")
    plt.title("Q-Learning Value Function — Mean Q Heatmap")

    plt.savefig("runs/plots/q_value_heatmap.png", dpi=200)
    print("Saved q_value_heatmap.png")

if __name__ == "__main__":
    main()
