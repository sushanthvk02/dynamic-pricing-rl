# train_qlearning.py

import os, json, argparse
import numpy as np
import pandas as pd
from envs.pricing_env import DynamicPricingEnv


def build_edges(env, n_inv_bins=10, n_price_bins=20):
    """Build discretization edges for inventory and price."""
    inv_edges = np.linspace(0.0, float(env.max_inventory), n_inv_bins + 1)
    price_edges = np.linspace(float(env.price_min), float(env.price_max), n_price_bins + 1)
    return inv_edges, price_edges


def obs_to_state(obs, horizon_days, inv_edges):
    """Map continuous observation - discrete state indices."""
    inv, _, day = obs
    inv_bin = int(np.digitize(inv, inv_edges) - 1)
    inv_bin = max(0, min(inv_bin, len(inv_edges) - 2))
    day_idx = max(0, min(int(day) - 1, horizon_days - 1))
    return inv_bin, day_idx


def idx_to_price(a_idx, price_edges):
    """Select midpoint price of a discrete price bin."""
    lo, hi = price_edges[a_idx], price_edges[a_idx + 1]
    return np.array([(lo + hi) * 0.5], dtype=np.float32)


def linear_decay(step, start, end, total_steps):
    """Linear schedule from start - end over total_steps."""
    if total_steps <= 0:
        return end
    frac = min(1.0, step / float(total_steps))
    return start + (end - start) * frac


def evaluate_greedy(Q, env, inv_edges, price_edges, episodes=20, seed=123):
    """Evaluate greedy policy (no exploration)."""
    rng = np.random.default_rng(seed)
    rewards = []

    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        G, done = 0.0, False

        while not done:
            inv_bin, day_idx = obs_to_state(obs, env.horizon_days, inv_edges)
            a_idx = int(np.argmax(Q[inv_bin, day_idx]))
            action = idx_to_price(a_idx, price_edges)

            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            G += r

        rewards.append(G)

    return float(np.mean(rewards))


def train_q_learning(
    episodes = 50_000,
    gamma = 0.99,
    eps_start = 0.30,
    eps_end = 0.05,
    alpha_start = 5e-3,
    alpha_end = 1e-3,
    n_inv_bins = 10,
    n_price_bins = 20,
    seed = 0,
    eval_every = 500,
    eval_episodes = 20,
):
    """Train a tabular Q-Learning agent."""
    env = DynamicPricingEnv(seed=seed)
    INV_EDGES, PRICE_EDGES = build_edges(env, n_inv_bins, n_price_bins)

    n_inv = len(INV_EDGES) - 1
    n_day = env.horizon_days
    n_act = len(PRICE_EDGES) - 1

    Q = np.zeros((n_inv, n_day, n_act), dtype=np.float32)

    total_steps_target = episodes * env.horizon_days
    rows = []
    global_step = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        G, done = 0.0, False

        # capturing starting eps/alpha 
        eps_log = linear_decay(global_step, eps_start, eps_end, total_steps_target)
        alpha_log = linear_decay(global_step, alpha_start, alpha_end, total_steps_target)

        while not done:
            eps = linear_decay(global_step, eps_start, eps_end, total_steps_target)
            alpha = linear_decay(global_step, alpha_start, alpha_end, total_steps_target)

            inv_bin, day_idx = obs_to_state(obs, env.horizon_days, INV_EDGES)

            # epsilon greedy
            if np.random.rand() < eps:
                a_idx = np.random.randint(n_act)
            else:
                a_idx = int(np.argmax(Q[inv_bin, day_idx]))

            action = idx_to_price(a_idx, PRICE_EDGES)

            obs2, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            G += r
            global_step += 1

            # TD update
            if done:
                target = r
            else:
                inv2, day2 = obs_to_state(obs2, env.horizon_days, INV_EDGES)
                target = r + gamma * np.max(Q[inv2, day2])

            Q[inv_bin, day_idx, a_idx] += alpha * (target - Q[inv_bin, day_idx, a_idx])

            obs = obs2

        #  logging episode results
        rows.append({
            "episode": ep,
            "algo": "Q-Learning",
            "reward": G,
            "eps": eps_log,
            "alpha": alpha_log,
            "n_price_bins": n_price_bins,
            "n_inv_bins": n_inv_bins,
            "gamma": gamma,
            "seed": seed,
        })

        # Periodic evaluation of GREEDY policy 
        if eval_every and (ep + 1) % eval_every == 0:
            eval_mean = evaluate_greedy(
                Q, env, INV_EDGES, PRICE_EDGES,
                episodes=eval_episodes,
                seed=seed + 500 + ep
            )
            rows.append({
                "episode": ep,
                "algo": "Q-Learning (eval, greedy)",
                "reward": eval_mean,
                "eps": 0.0,
                "alpha": alpha_log,
                "n_price_bins": n_price_bins,
                "n_inv_bins": n_inv_bins,
                "gamma": gamma,
                "seed": seed,
            })

    return pd.DataFrame(rows), Q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps_start", type=float, default=0.30)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--alpha_start", type=float, default=5e-3)
    p.add_argument("--alpha_end", type=float, default=1e-3)
    p.add_argument("--n_inv_bins", type=int, default=10)
    p.add_argument("--n_price_bins", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_episodes", type=int, default=20)
    args = p.parse_args()

    os.makedirs("runs/csv", exist_ok=True)
    os.makedirs("runs/models", exist_ok=True)

    df, Q = train_q_learning(**vars(args))
    df.to_csv("runs/csv/qlearning_rewards.csv", index=False)
    np.save("runs/models/qlearning_Q.npy", Q)

    with open("runs/models/qlearning_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Saved rewards/model/config for Q-Learning.")


if __name__ == "__main__":
    main()
