# tune_sarsa.py

import os, json, hashlib
import numpy as np
import pandas as pd
from agents.train_sarsa import train_sarsa


def cid(cfg):
    """Generate short unique hash for configuration."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def run():
    os.makedirs("runs/csv", exist_ok=True)
    os.makedirs("runs/models", exist_ok=True)

    # Hyperparameter sweep grid
    grid = []
    for n_price_bins in [10, 20, 30]:
        for n_inv_bins in [8, 10, 12]:
            for gamma in [0.97, 0.99]:
                for eps_start in [0.3, 0.4]:
                    for alpha_start in [5e-3, 3e-3]:
                        grid.append(dict(
                            episodes=50_000,
                            n_price_bins=n_price_bins,
                            n_inv_bins=n_inv_bins,
                            gamma=gamma,
                            eps_start=eps_start,
                            eps_end=0.05,
                            alpha_start=alpha_start,
                            alpha_end=1e-3,
                        ))

    all_rows = []
    summaries = []
    seeds = [0, 1, 2]

    for cfg in grid:
        config_id = cid(cfg)
        print(f"Running SARSA config {config_id}: {cfg}")

        per_seed_rewards = []

        for sd in seeds:
            df, Q = train_sarsa(seed=sd, **cfg)

            df["config_id"] = config_id
            df["seed"] = sd
            for k, v in cfg.items():
                df[k] = v

            all_rows.append(df)

            rewards = df[df["algo"] == "SARSA"]["reward"].values
            per_seed_rewards.append(rewards)

        # last 500 episode performance
        lastN = 500
        means = [np.mean(r[-lastN:]) for r in per_seed_rewards]
        stds = [np.std(r[-lastN:]) for r in per_seed_rewards]

        summaries.append(dict(
            config_id=config_id,
            seeds=len(seeds),
            mean_lastN=float(np.mean(means)),
            std_lastN=float(np.mean(stds)),
            **cfg
        ))

    pd.concat(all_rows).to_csv("runs/csv/sarsa_tuning_full.csv", index=False)

    leaderboard = pd.DataFrame(summaries).sort_values("mean_lastN", ascending=False)
    leaderboard.to_csv("runs/csv/sarsa_tuning_summary.csv", index=False)

    print("\nSARSA tuning complete.")
    print("Saved sarsa_tuning_full.csv and sarsa_tuning_summary.csv")


if __name__ == "__main__":
    run()
