from typing import Any, Callable
import torch as th
import torch.nn.functional as F
import numpy as np
from ppo.model import DefaultModel, BaseModel
from ppo.vec_env import VectorizedEnv
from torch.utils.tensorboard.writer import SummaryWriter
from ppo.buffer import RolloutBuffer
from collections import deque
import time
import gymnasium as gym
from dataclasses import dataclass

@dataclass
class PPOParameters:
    steps_per_rollout: int = 256
    """How many steps to take in every rollout. Note that this then results in steps_per_rollout * n_envs timesteps / observations per iteration."""

    gamma: float = 0.99
    """Discount factor"""

    gae_lambda: float = 0.95
    """Generalized advantage estitmation parameter"""

    entropy_coeff: float = 0.0
    value_coeff: float = 1.0
    learning_rate: float = 0.0003
    n_envs: int = 16

    num_epochs: int = 10
    """In each iteration, do num_epoch passes over the experience buffer."""
    
    minibatch_size: int = 64
    """In each pass over the experience buffer, proceed in steps of minibatch_size. Therefore, this number should divide steps_per_rollout * n_envs"""

    clip_range: float = 0.2
    seed: Any | None = None

    def ensure_seeded(self):
        if self.seed is not None:
            return
        
        rng = np.random.default_rng()
        self.seed = int(rng.integers(0, 0xff_ff_ff_ff))

class PPO:
    def __init__(
            self,
            env: Callable[[int], gym.Env],
            params: PPOParameters,
            ep_stats_window=100,
            log_interval=5,
            device=th.device("cpu"),
    ):
        params.ensure_seeded()
        
        # Seed BEFORE intializing the model
        self.params = params
        self.np_rng = np.random.default_rng(seed=params.seed)
        th.manual_seed(params.seed)
        self.device = device

        self.log_interval = log_interval
        
        self.env = VectorizedEnv(envs=[env(i) for i in range(self.params.n_envs)])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.buffer = RolloutBuffer(params.n_envs, params.steps_per_rollout, self.env.action_space, self.env.observation_space, params.gamma, params.gae_lambda)
        self.is_validation_run = False
        self.ep_info_buffer = deque(maxlen=ep_stats_window)

        self.make_model()
        self.optim = th.optim.AdamW(self.model.parameters(), lr=params.learning_rate)

    def make_model(self):
        self.model: BaseModel = DefaultModel(self.observation_space, self.action_space, device=self.device)

    def update_ep_info_buffer(self, infos):
        # Find episode summaries and store them in the buffer
        for info in infos:
            if "episode" not in info:
                continue

            self.on_new_episode_info(info["episode"])

    def on_new_episode_info(self, ep_info):
        self.ep_info_buffer.append(ep_info)

        # Also log key data to tensorboard
        self.writer.add_scalar("rollout/ep_length", ep_info["l"], self.total_timesteps)
        self.writer.add_scalar("rollout/ep_total_reward", ep_info["r"], self.total_timesteps)
        
    def collect_rollouts(self):
        self.buffer.reset()

        for step in range(self.params.steps_per_rollout):
            obs, trajectory_starts, action_mask = self.env.prepare()
            self.on_before_step(locals())

            with th.no_grad():
                pi, values = self.model.forward(self.model.obs_as_tensor(obs), action_mask=action_mask)
                actions = pi.sample().cpu().numpy()
                log_probs: np.ndarray = pi.logits.cpu().numpy() # type: ignore
                values: np.ndarray = values.cpu().numpy() # type: ignore
            
            obs_after_step, rewards, terminations, truncations, infos = self.env.step(actions)
            self.total_timesteps += self.params.n_envs

            # Check if we need to collect critic values for potentially truncated states. Note
            # that if this is the last step, then we do not have to do this, as it is done by the call below.
            if step < self.params.steps_per_rollout - 1:
                # If this is not the last step, we might need to provide critic evaluations for truncated states.
                # Note that observations that arise in the last step are handled below.
                truncated_obs = obs_after_step[truncations]
                truncated_values = 0.0
                if truncated_obs.size > 0:
                    truncated_values = np.zeros((self.params.n_envs,))
                    with th.no_grad():
                        truncated_values[truncations] = self.model.forward_critic(self.model.obs_as_tensor(truncated_obs)).cpu().numpy()

            self.on_after_step(locals())
            self.update_ep_info_buffer(infos)
            self.buffer.add(
                obs,
                log_probs,
                action_mask,
                values,
                actions,
                rewards,
                trajectory_starts,
                truncated_values
            )

        with th.no_grad():
            last_values = self.model.forward_critic(self.model.obs_as_tensor(obs_after_step)).cpu().numpy()
            self.buffer.finalize(last_values, terminations, truncations)

        if self.iteration > 0:
            self.dump_rollout_summary()

    def optimize(self):
        # PPO uses "mini batches". We do *multiple* optimization steps for one set of 
        # observations. That is also the reason why the "ratio" below is not just equal to one
        # (it is equal to one in the first iteration though)
        # In the "spinning up" version, they simply use multiple steps of gradient descent
        entropy_losses = []
        for _epoch in range(self.params.num_epochs):
            # In each epoch do a full pass over the rollout buffer. This is how it is done in stable baselines
            batches = self.buffer.get(self.params.minibatch_size, self.np_rng, device=self.device)

            for batch in batches:
                dist, values = self.model.forward(self.model.obs_as_tensor(batch.observations), action_mask=batch.action_masks)

                # Normalize advantages
                advs = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

                # Calculate losses
                ratio = th.exp(dist.log_prob(batch.actions) - batch.log_probs)
                actor_loss_1 = advs * ratio
                actor_loss_2 = advs * th.clamp(ratio, 1 - self.params.clip_range, 1 + self.params.clip_range)
                actor_loss = -th.min(actor_loss_1, actor_loss_2).mean()

                critic_loss = F.mse_loss(values, batch.returns)
                entropy_loss = -dist.entropy().mean()

                # Optimizer step
                loss = actor_loss + self.params.entropy_coeff * entropy_loss + self.params.value_coeff * critic_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                entropy_losses.append(float(entropy_loss))

        self.writer.add_scalar("optimize/entropy_loss", np.mean(entropy_losses), self.total_timesteps)

    def dump_rollout_summary(self):
        """Gets called every self.log_interval iterations."""
        rewards = [ep_info["r"] for ep_info in self.ep_info_buffer]
        if rewards:
            mean_reward = np.mean(rewards)
            self.writer.add_scalar("rollout/ep_reward_mean", mean_reward, self.total_timesteps)
        else:
            mean_reward = None

        lengths = [ep_info["l"] for ep_info in self.ep_info_buffer]
        if lengths:
            mean_length = np.mean(lengths)
            self.writer.add_scalar("rollout/ep_length_mean", mean_length, self.total_timesteps)
        else:
            mean_length = None

        steps_per_second = self.total_timesteps / (time.time() - self.start_time)

        print(f"Iteration {self.iteration} | Timesteps {self.total_timesteps} | Mean reward {mean_reward:.3f} | Mean length {mean_length:.1f} | Steps per second {steps_per_second:.1f}")
        

    def checkpoint_dict(self):
        res = {
            "state_dict": self.model.state_dict(),
            "optimizer_dict": self.optim.state_dict(),
        }

        return res
    
    def load_checkpoint_dict(self, saved):
        if isinstance(saved, str):
            saved = th.load(saved, weights_only=True)

        self.model.load_state_dict(saved["state_dict"])
        self.optim.load_state_dict(saved["optimizer_dict"])


    def learn(self, max_iterations: int | None = None, max_timesteps: int | None = None):
        # Initialize the tensorboard stuff only when we actually start learning
        self.writer = SummaryWriter(flush_secs=20)

        INF = 0xff_ff_ff_ff_ff_ff_ff_ff
        if max_iterations is None:
            max_iterations = INF
        if max_timesteps is None:
            max_timesteps = INF

        self.env.reset(seed=self.params.seed)
        self.iteration = 0
        self.total_timesteps = 0
        self.start_time = time.time()

        self.on_learn()
        for _ in range(max_iterations):
            self.on_before_iteration()

            self.collect_rollouts()
            self.optimize()

            self.on_after_iteration()

            self.iteration += 1
            if self.total_timesteps > max_timesteps:
                break

    def on_before_step(self, locals):
        pass

    def on_after_step(self, locals):
        pass

    def on_after_iteration(self):
        pass

    def on_before_iteration(self):
        pass

    def on_learn(self):
        pass