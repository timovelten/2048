from typing import Callable
from ppo.ppo import PPO, PPOParameters
import sys
import os
import time
import tempfile
import twentyfortyeight as twfe
import numpy as np
from ppo.model import BaseModel
from gymnasium import spaces
from torch import nn
import torch as th
from trajectory_writer import TrajectoryWriter
from ppo.model import build_fc_layers
from ppo.vec_env import VectorizedEnv
import dataclasses
import json
import datetime
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeatureExtractor(BaseFeaturesExtractor):
    """Given observations obs of shape (.., 4, 4), this returns a tensor A of shape (.., channels, 4, 4) such that

    Note that this also means that self.channels - 1 is the highest value that the observation can ever contain (or this will crash!)."""

    def __init__(self, observation_space: spaces.Box, channels: int, device=None):
        assert isinstance(
            observation_space, spaces.Box
        ) and observation_space.shape == (4, 4)
        conv = nn.Conv2d(channels, 12, 2, padding=1, device=device)
        with th.no_grad():
            obs = th.zeros((channels, 4, 4), dtype=th.float, device=device)
            self.out_shape = conv(obs).shape

        super().__init__(
            observation_space=observation_space, features_dim=np.prod(self.out_shape)
        )
        self.id_mat = th.eye(channels, device=device)
        self.layers = th.nn.Sequential(
            conv, nn.Flatten(start_dim=-len(self.out_shape)), nn.ReLU()
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # Use numpy magic to translate the obs tensor of shape (..., 4, 4) to a tensor A of shape (..., channels, 4, 4) such that
        #       A[.., v, x, y] == 1 <=> obs[.., x, y] == v.
        # Explanation of numpy magic: Assume that obs has shape (*K, *L), here L = (4, 4), but we keep it general.
        # Then consider A = self.id_mat[obs], then
        #      A[*k, *l, v] = I[obs[*k, *l], v]    i.e.   A[*k, *l, v] = 1 <=> obs[*k, *l] = v.
        # After swapping axis appropriately, we are done.
        x = self.id_mat[obs.int()].moveaxis(-1, -3)
        return self.layers(x)


class Model(BaseModel):
    def __init__(
        self, observation_space: spaces.Box, action_space: spaces.Discrete, device=None
    ):
        # 16 channels, i.e. the highest value we can represent is 2**15 = 32768. I doubt that
        # we will ever reach 2**16 = 65536 with the approach we take here (or possibly ever without
        # some SERIOUS effort), so we should be fine for a while.
        extractor = FeatureExtractor(
            observation_space=observation_space, channels=16, device=device
        )
        pi_layers, pi_out_features = build_fc_layers(extractor.features_dim, [128, 64])
        vf_layers, vf_out_features = build_fc_layers(extractor.features_dim, [128, 64])

        super().__init__(
            shared_net=extractor,
            action_space=action_space,
            vf_net=nn.Sequential(*vf_layers),
            vf_net_features=vf_out_features,
            pi_net=nn.Sequential(*pi_layers),
            pi_net_features=pi_out_features,
            device=device,
        )


def make_env(_: int):
    return twfe.Env()

def summarize_episodes(ep_infos):
    lines = []

    percentiles = [50, 80, 90, 95, 98]
    rewards = [ep_info["r"] for ep_info in ep_infos]
    scores = [ep_info["score"] for ep_info in ep_infos]
    highest_tiles = [ep_info["highest_tile"] for ep_info in ep_infos]

    row_format = "{:>12}" + "{:12.2f}" * len(percentiles)
    lines.append(row_format.format("", *percentiles))
    lines.append(row_format.format("Score", *np.percentile(scores, percentiles)))
    lines.append(
        row_format.format("Max Tile", *np.percentile(highest_tiles, percentiles))
    )
    lines.append(row_format.format("Reward", *np.percentile(rewards, percentiles)))
    highest_tiles_histogram: dict[int, float] = {
        tile: (count / len(highest_tiles))
        for tile, count in zip(*np.unique(highest_tiles, return_counts=True))
    }
    lines.append(
        "\t"
        + " | ".join(
            f"{tile}:{prob:.2f}" for tile, prob in highest_tiles_histogram.items()
        )
    )

    return "\n".join(lines), highest_tiles_histogram

class TwentyfortyEightAgent(PPO):
    def __init__(self, log_trajectories=False, **kwargs):
        # Openai call H = 1 / (1 - gamma) the horizon. It holds that gamma = 1 - 1 / H
        params = PPOParameters(
            gamma=1 - 1 / 1000,
            gae_lambda=0.95,
            learning_rate=5e-5,
            entropy_coeff=0.005,
            n_envs=512,
            value_coeff=1.0,
            steps_per_rollout=128,
            minibatch_size=128,
            num_epochs=3,
            clip_range=0.2,
        )
        super().__init__(
            env=make_env,
            params=params,
            log_interval=5,
            device=th.device("cpu"),
            **kwargs
        )
        self.validation_interval = 400
        self.log_trajectories = log_trajectories
        if self.log_trajectories:
            self.trajectory_writer = TrajectoryWriter("trajectories.db")

    def make_model(self):
        self.model = Model(
            self.observation_space, self.action_space, device=self.device
        )
        print(self.model)


    def dump_rollout_summary(self):
        super().dump_rollout_summary()

        if self.iteration % self.log_interval != 0:
            return

        summary, highest_tiles = summarize_episodes(self.ep_info_buffer)
        print(summary)

        cum = 0
        for i in reversed(range(1, 14)):
            tile = 2 << i
            cum += highest_tiles.get(tile, 0.0)

            if i < 7:
                continue

            self.writer.add_scalar(
                f"rollout/cum_highest_tile_{tile}", cum, self.total_timesteps
            )

        self.writer.add_text("rollout/summary", summary, self.total_timesteps)

    def on_after_step(self, locals):
        if not self.log_trajectories:
            return

        self.trajectory_writer.record(
            locals["obs"][0],
            int(locals["actions"][0]),
            float(locals["rewards"][0]),
            float(locals["values"][0]),
            locals["log_probs"][0],
            locals["terminations"][0],
            locals["truncations"][0],
            locals["infos"][0],
        )

    def on_learn(self):
        self.writer.add_text("model", str(self.model))
        self.writer.add_text("params", json.dumps(dataclasses.asdict(self.params)))

    def on_after_iteration(self):
        if not (self.iteration % self.validation_interval == 0 and self.iteration > 0):
            return

        seed = self.np_rng.integers(0, 2**32)        
        infos = run_validation(100, seed, self.model.get_actions)
        summary, _ = summarize_episodes(infos)
        summary = (
            summary
            + "\nSaved model to {}".format(self.save_checkpoint("runs"))
            + f"\nSeed: {seed}"
        )
        print("Validation")
        print(summary)
        self.writer.add_text("validation", summary, self.total_timesteps)

    def save_checkpoint(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        path = os.path.join(
            dir, f"checkpoint-{datetime.datetime.now().isoformat()}.pth"
        )
        th.save(self.checkpoint_dict(), path)
        return path

def run_validation(n, seed, get_actions: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    ep_infos = []
    ignore = np.zeros((n,), dtype=bool)
    env = VectorizedEnv([make_env(i) for i in range(n)])
    env.reset(seed=seed)

    while not np.all(ignore):
        obs, _, action_masks = env.prepare()
        actions = get_actions(obs, action_masks)
        obs, _, terminations, truncations, infos = env.step(actions)
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            if not ignore[idx] and maybe_ep_info is not None:
                ep_infos.append(maybe_ep_info)

        ignore |= terminations
        ignore |= truncations

    return ep_infos

def get_log_dir():
    tmpdir = tempfile.gettempdir()
    outdir = os.path.join(tmpdir, f"learn-{int(time.time())}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def train_2048():
    agent = TwentyfortyEightAgent(log_trajectories=True)
    agent.learn()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore
    train_2048()
