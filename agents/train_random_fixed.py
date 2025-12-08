# train_random_fixed.py

import os
import numpy as np
import pandas as pd
from envs.pricing_env import DynamicPricingEnv


def evaluate_fixed_price(env, price, episodes=300):
    """
    Evaluate a fixed price over many independent episodes.
    Returns the mean episode reward.
    """
    rewards = []

    for _ in range(episodes):
        seed = np.random.randint(0, 1_000_000)
        obs, _ = env.reset(seed=seed)

        done = False
        total_r = 0.0

        while not done:
            obs, r, terminated, truncated, _ = env.step(
                np.array([price], dtype=np.float32)
            )
            done = terminated or truncated
            total_r += r

        rewards.append(total_r)

    return float(np.mean(rewards))


def compute_best_fixed_price(env, grid_points=50, eval_episodes=300):
    """
    Brute-force search for the best single fixed price.
    Uses many episodes per price for stable estimation.
    """
    prices = np.linspace(env.price_min, env.price_max, grid_points)

    best_price = None
    best_mean_reward = -1e12

    for p in prices:
        mean_r = evaluate_fixed_price(env, p, episodes=eval_episodes)

        if mean_r > best_mean_reward:
            best_mean_reward = mean_r
            best_price = p

    print(f"[Fixed Baseline] Best price = {best_price:.2f}, Mean reward = {best_mean_reward:.2f}")
    return float(best_price)


def run_baseline(policy: str, episodes: int, seed: int, fixed_price=None):
    """
    Runs the Random or Fixed (best-price) baselines.
    """
    env = DynamicPricingEnv(seed=seed)
    rows = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=np.random.randint(1_000_000))
        done = False
        total = 0.0

        while not done:
            if policy == "random":
                action = env.action_space.sample()
            elif policy == "fixed":
                action = np.array([fixed_price], dtype=np.float32)
            else:
                raise ValueError("Unknown policy type.")

            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += r

        rows.append({
            "episode": ep,
            "algo": policy.capitalize(),
            "reward": total
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs("runs/csv", exist_ok=True)

    env = DynamicPricingEnv(seed=999)

    best_price = compute_best_fixed_price(env, grid_points=50, eval_episodes=300)

    df_random = run_baseline("random", episodes=50_000, seed=42)
    df_fixed  = run_baseline("fixed",  episodes=50_000, seed=43, fixed_price=best_price)

    df = pd.concat([df_random, df_fixed], ignore_index=True)
    df.to_csv("runs/csv/baselines_rewards.csv", index=False)

    print("Saved baselines_rewards.csv")


if __name__ == "__main__":
    main()
