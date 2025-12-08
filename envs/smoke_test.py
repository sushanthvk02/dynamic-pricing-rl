from envs.pricing_env import DynamicPricingEnv
import numpy as np

def run_smoke_test(seed: int = 0, steps: int = 10):
    """
    Runs a lightweight rollout using random price actions.
    Verifies that reset() and step() behave as expected.
    """
    env = DynamicPricingEnv(seed=seed)

    obs, info = env.reset()
    print(f"Initial observation: {obs}")

    done = False
    total_reward = 0.0
    step_count = 0

    while not done and step_count < steps:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step_count += 1

        print(
            f"Step {step_count:02d} | "
            f"Price={info['price']:.2f}, "
            f"Sales={info['sales']:.2f}, "
            f"Demand={info['demand']:.2f}, "
            f"Reward={reward:.2f}, "
            f"Inventory={obs[0]:.2f}"
        )

    print("\nSmoke test completed successfully.")
    print(f"Total reward after {step_count} steps: {total_reward:.2f}")


if __name__ == "__main__":
    run_smoke_test(seed=42, steps=15)
