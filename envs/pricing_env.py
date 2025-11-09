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
    Single-product dynamic pricing with finite inventory and stochastic demand.

    State  = [inventory (0..max_inventory), last_realized_demand (0..demand_cap), day (1..horizon_days)]
    Action = price in [price_min, price_max]  (continuous)
    Reward = price * sales - holding_cost_per_unit * remaining_inventory
    Episode ends when inventory == 0 or day > horizon_days.
    """

    metadata = {"render_modes": []}

    def __init__(
            
        # Initialization parameters
        self,
        seed: int = 0,
        horizon_days: int = 30,               
        max_inventory: int = 100,             
        demand_cap: float = 50.0,               
        price_min: float = 5.0,                
        price_max: float = 25.0,               
        base_demand: float = 50.0,              
        price_sensitivity: float = 2.0,         
        demand_noise_std: float = 5.0,          
        holding_cost_per_unit: float = 0.1    
    ):
        super().__init__()

        self.rng = np.random.default_rng(seed)

        # Configuration
        self.horizon_days = int(horizon_days)
        self.max_inventory = int(max_inventory)
        self.demand_cap = float(demand_cap)
        self.price_min, self.price_max = float(price_min), float(price_max)
        self.base_demand = float(base_demand)
        self.price_sensitivity = float(price_sensitivity)
        self.demand_noise_std = float(demand_noise_std)
        self.holding_cost_per_unit = float(holding_cost_per_unit)

        # Action space: one continuous action (price)
        self.action_space = spaces.Box(
            low=np.array([self.price_min], dtype=np.float32),
            high=np.array([self.price_max], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: [inventory, last_realized_demand, day]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            high=np.array([float(self.max_inventory), self.demand_cap, float(self.horizon_days)], dtype=np.float32),
            dtype=np.float32
        )

        # Internal state
        self._reset_internal()

    def _reset_internal(self) -> None:
        """Reset internal episode variables."""
        self.day = 1
        self.inventory = float(self.max_inventory)
        self.last_realized_demand = 0.0

    def _sample_demand(self, price: float) -> float:
        """
        Linear demand with Gaussian noise:
        demand = base_demand - price_sensitivity * price + Normal(0, demand_noise_std)
        Clipped to [0, demand_cap].
        """
        noise = self.rng.normal(0.0, self.demand_noise_std)
        demand = self.base_demand - self.price_sensitivity * price + noise
        demand = float(np.clip(demand, 0.0, self.demand_cap))
        return demand

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Returns initial observation and info dict."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internal()
        obs = np.array([self.inventory, self.last_realized_demand, self.day], dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        action: shape (1,) or scalar -> interpreted as price
        returns: (obs, reward, terminated, truncated, info)
        """

        if np.isscalar(action):
            price = float(action)
        else:
            price = float(action[0])

        price = float(np.clip(price, self.price_min, self.price_max))

        demand = self._sample_demand(price)
        sales = float(min(self.inventory, demand))
        self.inventory = float(max(0.0, self.inventory - sales))

        reward = price * sales - self.holding_cost_per_unit * self.inventory

        self.day += 1
        self.last_realized_demand = demand

        terminated = (self.inventory <= 0.0) or (self.day > self.horizon_days)

        obs = np.array(
            [self.inventory, self.last_realized_demand, min(self.day, self.horizon_days)],
            dtype=np.float32
        )

        info = {"price": price, "sales": sales, "demand": demand}
        return obs, float(reward), bool(terminated), False, info
