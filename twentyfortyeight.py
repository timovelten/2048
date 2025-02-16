import numpy as np
from numpy.typing import NDArray
from typing import Any
from dataclasses import dataclass
import gymnasium as gym
import time

from numba import jit

# This is encoded to mean "how many 90 degree rotation do I need to make this equivalent to the LEFT action"
UP = 1
DOWN = 3
LEFT = 0
RIGHT = 2
ACTIONS = ["left", "up", "right", "down"]


@jit
def fall(tiles: np.ndarray, action: int):
    new_tiles = np.copy(tiles)
    new_tiles_normalized = np.rot90(new_tiles, action)

    reward = 0
    for y in range(tiles.shape[0]):
        row = new_tiles_normalized[
            y, :
        ]  # This makes the code cleaner and has no performance impact (tested)

        for i in range(3):
            if row[i + 1] == 0:
                row[i + 1] = row[i]
                row[i] = 0
            elif row[i] == row[i + 1]:
                # Make the merged tile in the tile that is "further left" in order
                # to intentionally create a gap to the next tile (so that this tile cannot merge again)
                row[i] += 1
                row[i + 1] = 0
                reward += 1 << int(row[i])

        # Move zeros to the right. Idea: sort the two lower and upper tiles so that they are in {x 0, x x, 0 0}.
        if row[0] == 0:
            row[0] = row[1]
            row[1] = 0
        if row[2] == 0:
            row[2] = row[3]
            row[3] = 0

        # Now there are 9 cases remaining
        # 1) x 0 x 0
        # 2) x 0 x x
        # 3) x 0 0 0
        # 4) x x x 0
        # 5) x x x x
        # 6) x x 0 0
        # 7) 0 0 x 0
        # 8) 0 0 x x
        # 9) 0 0 0 0
        if row[0] == 0:
            # Cases 7, 8, 9
            row[0] = row[2]
            row[1] = row[3]
            row[2] = 0
            row[3] = 0
        elif row[1] == 0:
            # Cases 1, 2, 3
            row[1] = row[2]
            row[2] = row[3]
            row[3] = 0
        # ...otherwise we are in cases 4, 5 or 6 where we dont have to do anything.

    legal = not np.array_equal(tiles, new_tiles)
    return new_tiles, legal, reward, (1 << np.max(new_tiles)) * legal


@dataclass
class StateTransition:
    state: "State"
    legal: bool
    score: int

@dataclass
class State:
    tiles: NDArray[np.uint8]
    tile_sum: int
    highest_tile: int
    _next_states: None | list[StateTransition] = None

    @staticmethod
    def starting_state(rng: np.random.Generator):
        state = State.empty_state()
        state.add_random_tile(rng)
        state.add_random_tile(rng)
        return state

    @staticmethod
    def empty_state():
        return State(np.zeros((4, 4), dtype=np.uint8), tile_sum=0, highest_tile=0)

    @staticmethod
    def from_tiles(tiles: np.ndarray):
        """Note: Expects log tiles, i.e. 0 for empty, 1 for the 2 tile,...,10 for the 1024 tile"""
        tiles = np.reshape(tiles, (4, 4)).astype(dtype=np.uint8)
        big_tiles = 2 ** tiles.astype(dtype=np.uint64) * (tiles != 0)
        return State(
            tiles=tiles, tile_sum=np.sum(big_tiles), highest_tile=np.max(big_tiles)
        )

    def add_random_tile(self, rng: np.random.Generator):
        ys, xs = np.where(self.tiles == 0)
        idx = rng.integers(0, len(xs))
        if rng.random() <= 0.9:
            self.tiles[ys[idx], xs[idx]] = 1
            self.tile_sum += 2
            if self.highest_tile < 2:
                self.highest_tile = 2
        else:
            self.tiles[ys[idx], xs[idx]] = 2
            self.tile_sum += 4
            if self.highest_tile < 4:
                self.highest_tile = 4

    def _next_state(self, action: int) -> StateTransition:
        new_tiles, legal, reward, highest_tile = fall(self.tiles, action)
        return StateTransition(
            State(new_tiles, tile_sum=self.tile_sum, highest_tile=highest_tile),
            legal,
            reward,
        )

    def __str__(self):
        lines = []
        hr = "+-------+-------+-------+-------+"

        for y in range(4):
            lines.append(hr)

            formatted_numbers = (
                "{:^7}".format(1 << int(a) if a != 0 else "") for a in self.tiles[y, :]
            )
            row = "|" + "|".join(formatted_numbers) + "|"
            lines.append(row)

        lines.append(hr)
        return "\n".join(lines)

    @property
    def next_states(self):
        if self._next_states is None:
            self._next_states = [self._next_state(i) for i in range(4)]
        return self._next_states

    @property
    def is_terminated(self):
        return not any(delta.legal for delta in self.next_states)

    @property
    def action_mask(self):
        return np.array([dela.legal for dela in self.next_states])

    def copy(self):
        return State(
            tiles=np.copy(self.tiles),
            tile_sum=self.tile_sum,
            highest_tile=self.highest_tile,
        )

class Env(gym.Env):
    def __init__(self, truncate_at_tile: int = 32768):
        """If the highest tile reaches the value truncate_at_tile, the step function will return truncated true. Note that
        the value truncate_at_tile will therefore appear in one and only one of the returned states."""
        self.observation_space = gym.spaces.Box(
            low=0, high=16, shape=(4, 4), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(4)
        self.current_state = State.empty_state()
        self.truncate_at = truncate_at_tile

    def state(self):
        return np.copy(self.current_state.tiles)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_state = (
            options["state"]
            if options and "state" in options
            else State.starting_state(self.np_random)
        )

        self.score = 0
        self.total_reward = 0
        self.len = 0
        self.start_time = time.time()

        return self.state(), {}

    def step(self, action: int):
        delta = self.current_state.next_states[action]
        truncated = delta.state.highest_tile >= self.truncate_at
        win_bonus = 2048.0 * float(
            delta.state.highest_tile == 2048 and self.current_state.highest_tile != 2048
        )

        reward = (delta.score + win_bonus) / self.current_state.tile_sum

        # Book keeping
        self.score += delta.score
        self.len += 1
        self.current_state = delta.state.copy()
        self.current_state.add_random_tile(self.np_random)

        self.total_reward += reward

        terminated = self.current_state.is_terminated
        if terminated or truncated:
            # The l, r and t are named that way to be compatible with SB3's Monitor env wrapper
            ep_info = dict(
                l=self.len,
                r=self.total_reward,
                t=time.time() - self.start_time,
                highest_tile=self.current_state.highest_tile,
                score=self.score,
            )
            info: dict[str, Any] = dict(episode=ep_info)
        else:
            info = {}

        return self.state(), reward, terminated, truncated, info

    def action_masks(self):
        return self.current_state.action_mask
