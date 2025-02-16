import numpy as np
import torch as th

from typing import Union
from dataclasses import dataclass
from gymnasium import spaces

@dataclass
class RolloutSample:
    # Note that, for now, we store the log probs of *all* actions, but these are the log probs of the selected actions.
    # (as those are the only one possibly relevant for an on policy algorithm)
    log_probs: th.Tensor
    actions: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    observations: th.Tensor
    action_masks: th.Tensor

class RolloutBuffer:
    def __init__(self, n_envs: int, steps: int, action_space: spaces.Discrete, observation_space: spaces.Box, discount_factor: float, gae_lambda: float):
        self.steps = steps
        self.n_envs = n_envs
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = discount_factor
        self.gae_lambda = gae_lambda

        self.reset()

    def reset(self):
        # Scalar values we observe in each step. All of these are [n, n_envs]
        self.values = np.zeros((self.steps, self.n_envs), dtype=np.float32)
        self.rewards = np.zeros((self.steps, self.n_envs), dtype=np.float32)
        self.actions = np.zeros((self.steps, self.n_envs), dtype=np.int64)
        self.trajectory_continues = np.zeros((self.steps, self.n_envs), dtype=bool)

        # Non-Scalar values
        self.observations = np.zeros((self.steps, self.n_envs, *self.observation_space.shape), dtype=self.observation_space.dtype)
        self.log_probs = np.zeros((self.steps, self.n_envs, self.action_space.n), dtype=np.float32) 
        self.action_masks = np.zeros((self.steps, self.n_envs, self.action_space.n), dtype=bool)

        # For every environment, get a final value of the value function (technically we only need these values
        # for non-terminated environments right now, but we do not rely on that fact)
        self.last_values = np.zeros((self.n_envs,), dtype=np.float32)

        self.idx = 0

        # Note: We actually require the log_probs to be log_probs (as opposed to logits). That is, they lie in the range (-inf, 0].

    def add(
        self,
        observations: np.ndarray,
        log_probs: np.ndarray,
        action_masks: np.ndarray, 
        values: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray,
        trajectory_starts: np.ndarray,
        truncated_values: np.ndarray | float,
    ):
        """
        Note: truncated_values should contain value function evaluations for observations that are truncated *in the middle of a rollout*.
        This value should be zero for the last step. Value function evaluations for the last states are passed to finalize instead.
        """
        # Note that the observations here are the state *before* the action was taken. So really they are the
        # "observations" of the previous action.
        self.observations[self.idx] = observations
        self.values[self.idx] = values
        self.log_probs[self.idx] = log_probs
        self.actions[self.idx] = actions
        self.rewards[self.idx] = rewards + self.gamma * truncated_values
        self.action_masks[self.idx] = action_masks

        # We currently only observe if this is the start of a new trajectory, but in the calculations we main
        if self.idx > 0:
            self.trajectory_continues[self.idx - 1] = trajectory_starts == False  # noqa: E712

        self.finalized = False

        self.idx += 1

    def finalize(self, last_values, last_terminated, last_truncated):
        """Note: last_values do not have to be masked. However, all values of terminated environments are ignored. 
        The ones of truncated are very much not ignored though."""

        assert self.idx >= self.steps

        # Bootstrap in place
        self.rewards[-1] += self.gamma * last_values * (last_terminated == False) # noqa: E712
        self.trajectory_continues[-1] = np.logical_or(last_terminated, last_truncated) == False # noqa: E712
        self.finalized = True
        self.compute_returns_and_advantages()

    def compute_returns_and_advantages(self):
        assert self.finalized # Need to call .finalize() before calling this

        self.advantages = np.zeros((self.steps, self.n_envs), dtype=np.float32)
        self.finalized = True

        # Consider
        # * Sum of discounted rewards G(t) = \sum_{k \geq t} gamma^t R_t
        # * Advantage function A(t) = G(t) - V(t)
        # * d_t = r_t + gamma * V(t + 1) - V(t)
        # Then (telescoping sum)
        #   A(t) = \sum_{k \geq t} gamma^{k - t} d_t = d_t + gamma * A(t + 1)
        acc = np.zeros_like(self.last_values)

        # First, bootstrap from the value (if not terminated!)
        for step in reversed(range(self.steps)):
            # If this is the final step of a trajectory, there is no "next" (as in later in time) value. 
            # We also need to ignore the accumulated value then.
            if step < self.steps - 1:
                next_values = self.values[step + 1] * self.trajectory_continues[step]
            else:
                next_values = 0

            delta = self.rewards[step] + self.gamma * next_values - self.values[step]
            acc = delta + self.gamma * self.gae_lambda * self.trajectory_continues[step] * acc
            self.advantages[step] = acc

        self.returns = self.advantages + self.values
    
    def get(self, batch_size: Union[int, None], rng: np.random.Generator, device = None):
        # This is very heavily inspired by how it is done in stable baselines 3
        n = self.steps * self.n_envs
        if batch_size is None:
            batch_size = n

        indices = rng.permutation(n)

        # Flatten the observations from [rollout, n_envs, ...] to [n, ...]
        returns = th.from_numpy(self.returns).flatten().to(device)
        actions = th.from_numpy(self.actions).flatten().to(device)
        advantages = th.from_numpy(self.advantages).flatten().to(device)
        observations = th.from_numpy(self.observations).flatten(start_dim=0, end_dim=1).to(device)
        action_masks = th.from_numpy(self.action_masks).flatten(start_dim=0, end_dim=1).to(device)
        log_probs = th.from_numpy(self.log_probs).flatten(start_dim=0, end_dim=1).to(device)

        # In what follows, we only need log probs of selected actions. Discard the rest.
        log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze().to(device)

        # Yield the observations. Note that these are shuffled.
        idx = 0
        while idx < n:
            part = indices[idx:idx + batch_size]
            yield RolloutSample(
                log_probs=log_probs[part],
                actions=actions[part],
                advantages=advantages[part],
                returns=returns[part],
                observations=observations[part],
                action_masks=action_masks[part]
            )

            idx += batch_size
