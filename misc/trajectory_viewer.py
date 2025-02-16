import argparse

from blessed import Terminal
from dataclasses import dataclass

import sqlite3
import numpy as np
import util
import twentyfortyeight as twfe

# Utility to view the trajectories that were stored during training in the database. This visualizes the game state
# in a way that is more human readable than plain numpy arrays.
# Controls: left, right arrows and a and d
# Run as python3 -m misc.trajectory_viewer from the root directory


@dataclass
class Step:
    step: int
    obs: np.ndarray
    action: int
    reward: float
    value: float
    log_probs: np.ndarray


def load_trajectory(db: sqlite3.Connection, id: str) -> list[Step]:
    stmt = db.execute(
        "SELECT step, obs, action, reward, value, log_probs FROM observations WHERE trajectory_id = ? ORDER BY step",
        (id,),
    )
    steps = stmt.fetchall()

    result = []
    for row in steps:
        step, obs, action, reward, value, log_probs = row
        data = Step(
            step,
            util.load_ndarray(obs),
            action,
            float(reward),
            float(value),
            util.load_ndarray(log_probs),
        )
        result.insert(data.step, data)

    return result


def game_state_from_obs(obs: np.ndarray):
    return twfe.State.from_tiles(obs)


def view_trajectory(data: list[Step]):
    if len(data) == 0:
        print("No data")
        return

    term = Terminal()
    idx = 0
    while True:
        step = data[idx]
        game_state = game_state_from_obs(step.obs)

        picked_action = twfe.ACTIONS[step.action]
        probs = util.softmax(step.log_probs)
        items = [
            f"#{step.step}",
            f"U: {probs[twfe.UP]:.3}",
            f"D: {probs[twfe.DOWN]:.3}",
            f"L: {probs[twfe.LEFT]:.3}",
            f"R: {probs[twfe.RIGHT]:.3}",
            f"Action: {picked_action}",
            f"Reward: {step.reward:.3}",
            f"Value: {step.value:.3}",
        ]
        print(" | ".join(items))
        print(game_state)
        print()

        while True:
            with term.raw():
                inp = term.inkey()
                if inp == chr(3) or inp == "q":
                    exit()

            new_idx = idx
            if inp.code == term.ENTER or inp == " " or inp.code == term.KEY_RIGHT:
                new_idx = min(len(data) - 1, idx + 1)
            if inp.code == term.KEY_LEFT:
                new_idx = max(0, idx - 1)
            if inp == "d":
                new_idx = new_idx = min(len(data) - 1, idx + 20)
            if inp == "a":
                new_idx = max(0, idx - 20)

            if 0 <= new_idx < len(data) and new_idx != idx:
                idx = new_idx
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database_file")
    parser.add_argument("trajectory_id")
    args = parser.parse_args()

    db = sqlite3.connect(args.database_file)
    data = load_trajectory(db, args.trajectory_id)
    view_trajectory(data)

if __name__ == "__main__":
    main()
