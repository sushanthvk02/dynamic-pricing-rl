import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from envs.pricing_env import DynamicPricingEnv
import os

def evaluate_fixed_price(env, price, episodes = 200):
    """Run multiple episodes with a constant price."""
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, G = False, 0.0
        while not done:
            action = np.array([price], dtype=np.float32)
            obs, r, done, _, _ = env.step(action)
            G += r
        rewards.append(G)
    return np.mean(rewards)

def sweep_fixed_prices(price_min = 5.0, price_max = 25.0,
                       price_steps = 40, episodes = 200):
    """Sweep through a range of prices and find the best constant price."""
    env = DynamicPricingEnv()

    prices = np.linspace(price_min, price_max, price_steps)
    results = []

    for price in prices:
        mean_reward = evaluate_fixed_price(env, price, episodes)
        results.append({"price": price, "mean_reward": mean_reward})
        print(f"Price = {price:.2f}, Mean Reward = {mean_reward:.2f}")

    df = pd.DataFrame(results)
    return df

def main():
    os.makedirs("runs/csv", exist_ok=True)
    os.makedirs("runs/plots", exist_ok=True)

    df = sweep_fixed_prices()
    df.to_csv("runs/csv/best_fixed_price_sweep.csv", index=False)

    # Find best price
    best_row = df.loc[df["mean_reward"].idxmax()]
    best_price = best_row["price"]
    best_reward = best_row["mean_reward"]

    print("\nBest Fixed Price Baseline")
    print(f"Best Price: {best_price:.2f}")
    print(f"Mean Reward: {best_reward:.2f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["price"], df["mean_reward"], marker="o")
    plt.axvline(best_price, color="red", linestyle="--", label=f"Best Price = {best_price:.2f}")
    plt.xlabel("Fixed Price")
    plt.ylabel("Mean Episode Reward")
    plt.title("Fixed Price Sweep - Best Constant Price Baseline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = "runs/plots/best_fixed_price.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
