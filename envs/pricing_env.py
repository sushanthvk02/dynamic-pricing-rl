# pricing_env.py
#
# Reference notice (design inspiration only; no direct code reuse):
#   Alibandehloo, H. "Dynamic Pricing Optimization using Deep Q-Network (DQN)."
#   GitHub repository: https://github.com/Hossein-Alibandehloo/Dynamic-Pricing-with-Reinforcement-Learning
#   (accessed Nov 2025)
#
#   Zhao, M. "Calyber_QLearning – Q-learning for dynamic pricing/matching."
#   GitHub repository: https://github.com/MalcolmZhao/Calyber_QLearning
#   (accessed Nov 2025)
#
# I implemented this environment from scratch (Gymnasium API). The above projects
# informed our high-level setup (pricing-as-MDP, reward framing, experiment workflow).


import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict

class DynamicPricingEnv(gym.Env):
    """
    Dynamic pricing environment for a single product with finite inventory.
    State = [inventory, last_realized_demand, day]
    Action = continuous price in [price_min, price_max]
    Reward = price * sales − holding_cost * remaining_inventory
    Episode ends when inventory is 0 or horizon is exceeded.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 0,
        horizon_days: int = 30,
        max_inventory: int = 300,
        demand_cap: float = 50.0,
        price_min: float = 5.0,
        price_max: float = 25.0,
        base_demand: float = 50.0,
        price_sensitivity: float = 0.2,
        demand_noise_std: float = 2.0,
        holding_cost_per_unit: float = 0.1
    ):
        super().__init__()

        self.rng = np.random.default_rng(seed)

        # Parameters
        self.horizon_days = int(horizon_days)
        self.max_inventory = float(max_inventory)
        self.demand_cap = float(demand_cap)
        self.price_min = float(price_min)
        self.price_max = float(price_max)
        self.base_demand = float(base_demand)
        self.price_sensitivity = float(price_sensitivity)
        self.demand_noise_std = float(demand_noise_std)
        self.holding_cost_per_unit = float(holding_cost_per_unit)

        self.episode_reward = 0.0

        # Actions = continuous price directly in [min, max]
        self.action_space = spaces.Box(
            low=np.array([self.price_min], dtype=np.float32),
            high=np.array([self.price_max], dtype=np.float32),
            dtype=np.float32
        )

        # Observation = inventory, last_demand, day
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            high=np.array([self.max_inventory, self.demand_cap, float(self.horizon_days)], dtype=np.float32),
            dtype=np.float32
        )

        self._reset_internal()

    def _reset_internal(self):
        self.day = 1
        self.inventory = float(self.max_inventory)
        self.last_realized_demand = 0.0

    def _sample_demand(self, price: float) -> float:
        """Exponential demand curve with mild seasonality + noise."""
        mean_price_effect = self.base_demand * np.exp(-self.price_sensitivity * price)
        seasonal = 1.0 + 0.30 * np.sin(2 * np.pi * self.day / self.horizon_days)
        noise = self.rng.normal(0.0, self.demand_noise_std)
        demand = mean_price_effect * seasonal + noise
        return float(np.clip(demand, 0.0, self.demand_cap))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:

        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_internal()
        self.episode_reward = 0.0

        obs = np.array(
            [self.inventory, self.last_realized_demand, self.day],
            dtype=np.float32
        )

        return obs, {}

    def step(self, action):
        """Takes a price action and advances the environment one day."""
        if np.isscalar(action):
            price = float(action)
        else:
            price = float(action[0])

        price = float(np.clip(price, self.price_min, self.price_max))

        demand = self._sample_demand(price)
        sales = float(min(self.inventory, demand))
        self.inventory = float(max(0.0, self.inventory - sales))

        reward = price * sales - self.holding_cost_per_unit * self.inventory
        self.episode_reward += reward

        self.day += 1
        self.last_realized_demand = demand

        terminated = (self.inventory <= 0.0) or (self.day > self.horizon_days)

        obs = np.array(
            [self.inventory, self.last_realized_demand, min(self.day, self.horizon_days)],
            dtype=np.float32
        )

        info = {"price": price, "sales": sales, "demand": demand}

        if terminated:
            info["episode"] = {
                "r": float(self.episode_reward),
                "l": int(self.day - 1)
            }

        return obs, float(reward), terminated, False, info
