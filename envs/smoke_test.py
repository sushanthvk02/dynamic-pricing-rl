from envs.pricing_env import DynamicPricingEnv
import numpy as np

def run_smoke_test(seed: int = 0, steps: int = 10):
    """
    Runs a quick environment rollout with random pricing actions.
    Confirms that the Gymnasium API (reset/step) works correctly.
    """

    # Create environment instance
    env = DynamicPricingEnv(seed=seed)

    # Reset to initial state
    observation, _ = env.reset()
    print(f"Initial observation: {observation}")

    done = False
    total_reward = 0.0
    step_count = 0

    while not done and step_count < steps:
        # Sample a random price from the action space
        action = env.action_space.sample()

        # Advance the environment one step
        observation, reward, done, _, info = env.step(action)
        total_reward += reward
        step_count += 1

        print(
            f"Step {step_count:02d} | Price={info['price']:.2f}, "
            f"Sales={info['sales']:.2f}, Demand={info['demand']:.2f}, "
            f"Reward={reward:.2f}, Inventory={observation[0]:.2f}"
        )

    print("\nSmoke test completed successfully.")
    print(f"Total reward after {step_count} steps: {total_reward:.2f}")

if __name__ == "__main__":
    # Run a short random-action rollout
    run_smoke_test(seed=42, steps=15)