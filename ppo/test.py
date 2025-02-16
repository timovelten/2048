import gymnasium as gym
from .ppo import PPO, PPOParameters
from stable_baselines3.common.monitor import Monitor
import numpy as np

# Sanity check. Make sure that our PPO implementation can at least learn CartPole
# Run this as python3 -m ppo.test from the root

class AllActionsWrapper(gym.Wrapper):
    """Wrapper that adds "trivial" action masking to an environment, that is
    all actions are available all the time"""
    action_space: gym.spaces.Discrete

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(self.action_space, gym.spaces.Discrete)

    def action_masks(self):
        return np.ones((self.action_space.n,), dtype=bool)

def train(id):
    def make_env(i: int):
        env = gym.make("CartPole-v1")
        env = AllActionsWrapper(env = env)
        env = Monitor(env = env)
        return env
    
    agent = PPO(env = make_env, params=PPOParameters())
    agent.learn(max_timesteps=100_000)

if __name__ == "__main__":
    train("CartPole-v1")