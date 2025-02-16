from typing import Type
import numpy as np
import gymnasium as gym

class VectorizedEnv:
    def __init__(self, envs):
        self.envs = envs
        self.n_envs = len(envs)

        action_space = self.envs[0].action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        self.action_space = action_space

        obs_space = self.envs[0].observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        self.observation_space = obs_space
        
        self.actions_dim = self
        shape = self.envs[0].observation_space.shape
        assert shape is not None

        self.obs_shape = shape

        try:
            _ = self.envs[0].get_wrapper_attr("action_masks")
        except AttributeError:
            raise Exception("Environment should have an action_masks method that returns for every action whether it is possible or not. Note that" +
                            "you can implement this trivially, if you do not want to use action masking")

    def reset(self, seed=None, **kwargs):
        rng = np.random.default_rng(seed=seed)
        seeds = rng.integers(low=0, high=np.iinfo(np.int64).max, size=self.n_envs)

        self.current_obs = np.zeros((self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.needs_reset = np.ones((self.n_envs,), dtype=bool)
        for idx in range(self.n_envs):
            self.current_obs[idx], _ = self.envs[idx].reset(seed=int(seeds[idx], **kwargs))
        
    def prepare(self):        
        for idx in range(self.n_envs):
            if self.needs_reset[idx]:
                self.current_obs[idx], _ = self.envs[idx].reset()

        trajectory_starts = self.needs_reset.copy()
        self.needs_reset[...] = False
        action_masks = np.stack([env.get_wrapper_attr("action_masks")() for env in self.envs])

        return self.current_obs.copy(), trajectory_starts, action_masks
        
    def step(self, actions: np.ndarray):
        rewards = np.zeros((self.n_envs,))
        terminateds = np.zeros((self.n_envs,), dtype=bool)
        truncateds = np.zeros((self.n_envs,), dtype=bool)
        infos = []

        for i in range(self.n_envs):
            obs, reward, terminated, truncated, info = self.envs[i].step(actions[i])
            
            rewards[i] = reward
            self.current_obs[i] = obs
            terminateds[i] = terminated
            truncateds[i] = truncated
            self.needs_reset[i] = terminated or truncated
            infos.append(info)
        
        return self.current_obs.copy(), rewards, terminateds, truncateds, infos