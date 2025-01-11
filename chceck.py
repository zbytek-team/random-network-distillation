import torch

print(torch.cuda.is_available())
print(torch.__version__)

import gymnasium as gym
import ale_py

def main() -> None:
    gym.register_envs(ale_py)

    env = gym.make("ALE/MontezumaRevenge-v5", render_mode="human")

    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()