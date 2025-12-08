import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    os.makedirs("runs/plots", exist_ok=True)

    Q = np.load("runs/models/qlearning_Q.npy")
    policy = np.argmax(Q, axis=2) 

    plt.figure(figsize=(12, 6))
    plt.imshow(policy, cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(label="Chosen Action Index")
    plt.xlabel("Day")
    plt.ylabel("Inventory Bin")
    plt.title("Q-Learning Greedy Policy Heatmap")

    plt.savefig("runs/plots/q_policy_heatmap.png", dpi=200)
    print("Saved q_policy_heatmap.png")

if __name__ == "__main__":
    main()
