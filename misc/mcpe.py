import twentyfortyeight as twfe
import train
import numpy as np
import gymnasium as gym
from ppo.vec_env import VectorizedEnv
import torch as th
import argparse
import itertools

class FixedInitialStateWrapper(gym.Wrapper):
    def __init__(self, env, state):
        super().__init__(env)
        self.initial_state = state

    def reset(self, **kwargs):
        return self.env.reset(options = {"state": self.initial_state}, **kwargs)

class MonteCarloPolicy:
    def __init__(self, agent: train.TwentyfortyEightAgent, np_random: np.random.Generator | None = None):
        self.agent = agent
        self.model = agent.model

        if np_random is None:
            self.np_random = np.random.default_rng()
        else:
            self.np_random = np_random

    def evaluate_action(self, state: twfe.State, first_action: int, n_envs = 64, n_steps = 32):
        envs = [FixedInitialStateWrapper(twfe.Env(), state) for i in range(n_envs)]
        env = VectorizedEnv(envs)
        env.reset()

        # We will collect at least 100 runs, possibly more if some of them finish early
        n_runs = 0
        returns = 0

        gamma = np.ones((n_envs,), dtype=float)
        
        for _i in range(n_steps):
            obs, trajectory_starts, action_mask = env.prepare()
            actions = self.model.get_actions(obs, action_mask)
            actions[trajectory_starts] = first_action

            obs_after_step, rewards, terminations, _truncations, _infos = env.step(actions)
            returns += np.sum(gamma * rewards)

            gamma *= self.agent.params.gamma

            # If an environment terminated, reset its discount factor
            gamma[terminations] = 1.0
            n_runs += np.count_nonzero(terminations)

        # Now, bootstrap all the remaining environment from the value function
        not_terminated_mask = np.invert(terminations)
        masked_obs = obs_after_step[not_terminated_mask]

        with th.no_grad():
            rets = self.model.forward_critic(self.model.obs_as_tensor(masked_obs)).cpu().numpy() * gamma[not_terminated_mask]
            n_runs += np.size(rets)
            returns += np.sum(rets)

        return returns / n_runs

    def get_action(self, state: twfe.State):
        with th.no_grad():
            pi, _ = self.model.forward(self.model.obs_as_tensor(state.tiles), state.action_mask)
            action_probs = pi.probs.cpu().numpy() # type: ignore

        # Mask out those actions that the model really does not like
        mask = np.copy(state.action_mask)
        mask &= action_probs >= 0.1

        # If we are only going to simulate one action, we might as well return that now
        if np.count_nonzero(mask) == 1:
            return np.argmax(mask).item()

        returns = np.zeros(4)
        for action in range(4):
            if not state.action_mask[action] or action_probs[action] < 0.2:
                returns[action] = -np.inf
                continue

            returns[action] = self.evaluate_action(state, action)

        return np.argmax(returns).item()

def validate_once(policy: MonteCarloPolicy):
    env = twfe.Env()
    obs, _ = env.reset()
    
    for i in itertools.count():
        action = policy.get_action(env.current_state)
        if i % 100 == 0:
            print(env.current_state)

        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return env.current_state, info

def validate(policy: MonteCarloPolicy, n = 10):
    infos = []
    for i in range(n):
        _, info = validate_once(policy)
        infos.append(info["episode"])
        summary, _ = train.summarize_episodes(infos)
        print(summary)

    print(train.summarize_episodes(infos))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_pth")
    args = parser.parse_args()

    agent = train.TwentyfortyEightAgent()
    agent.load_checkpoint_dict(args.model_pth)
    policy = MonteCarloPolicy(agent)

    validate(policy)