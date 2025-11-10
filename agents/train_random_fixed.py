# agents/train_random_fixed.py
import os
import numpy as np
import pandas as pd
from envs.pricing_env import DynamicPricingEnv

def run_baseline(policy: str, episodes: int = 200, seed: int = 0) -> pd.DataFrame:
    """Runs Random or Fixed-price baseline and returns a dataframe of episodic rewards."""
    env = DynamicPricingEnv(seed=seed)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            if policy == "random":
                action = env.action_space.sample()
            else:  
                action = np.array([(env.price_min + env.price_max) / 2.0], dtype=np.float32)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
        rows.append({"episode": ep, "algo": policy.capitalize(), "reward": ep_reward})
    return pd.DataFrame(rows)

def main():
    os.makedirs("runs/csv", exist_ok=True)
    df = pd.concat(
        [run_baseline("random", episodes=200, seed=42),
         run_baseline("fixed",  episodes=200, seed=43)],
        ignore_index=True
    )
    out = "runs/csv/baselines_rewards.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
