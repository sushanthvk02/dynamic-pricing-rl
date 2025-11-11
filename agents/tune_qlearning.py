import os, json, hashlib
import numpy as np
import pandas as pd
from agents.train_qlearning import train_q_learning


def cid(cfg):
    """Create a short unique hash ID for each hyperparameter configuration."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def run():
    """Run a small hyperparameter sweep for Q-Learning and save results."""
    os.makedirs("runs/csv", exist_ok=True)
    os.makedirs("runs/models", exist_ok=True)

    # Defining the hyperparameter grid
    grid = []
    for n_price_bins in [10, 20, 30]:
        for n_inv_bins in [8, 10]:
            for gamma in [0.97, 0.99]:
                for eps_start in [0.3, 0.4]:
                    for alpha_start in [5e-3, 3e-3]:
                        grid.append(dict(
                            episodes=20_000,
                            n_price_bins=n_price_bins,
                            n_inv_bins=n_inv_bins,
                            gamma=gamma,
                            eps_start=eps_start,
                            eps_end=0.05,
                            alpha_start=alpha_start,
                            alpha_end=1e-3,
                        ))

    summaries, all_rows = [], []
    seeds = [0, 1, 2]

    # Run each config across multiple seeds
    for cfg in grid:
        id_ = cid(cfg)
        per_seed = []

        for sd in seeds:
            df, Q = train_q_learning(seed=sd, **cfg)
            df["config_id"] = id_
            df["seed"] = sd
            for k, v in cfg.items():
                df[k] = v
            all_rows.append(df)
            per_seed.append(df[df["algo"] == "Q-Learning"]["reward"].values)

        # average reward across last 500 episodes
        lastN = 500
        means = [np.mean(r[-lastN:]) for r in per_seed]
        stds = [np.std(r[-lastN:]) for r in per_seed]

        summaries.append(dict(
            config_id=id_,
            seeds=len(seeds),
            mean_lastN=float(np.mean(means)),
            std_lastN=float(np.mean(stds)),
            **cfg
        ))

    # Save all episode level data and summary table
    pd.concat(all_rows).to_csv("runs/csv/qlearning_sweep.csv", index=False)
    pd.DataFrame(summaries).sort_values("mean_lastN", ascending=False).to_csv(
        "runs/csv/qlearning_sweep_summary.csv", index=False
    )

    print("Saved sweep CSVs in runs/csv/")


if __name__ == "__main__":
    run()
