import numpy as np
from ulid import ULID
import sqlite3
import util
import json
import time

def ensure_database_initialized(database_path):
    con = sqlite3.connect(database_path)
    con.executescript(r"""
        CREATE TABLE IF NOT EXISTS "trajectories" (
                "id"            BLOB NOT NULL,
                "length"        INTEGER,
                "total_reward"  NUMERIC,
                "other"         TEXT,
                "finished_at"   TEXT,
                PRIMARY KEY("id")
        );
        CREATE TABLE IF NOT EXISTS "observations" (
                "trajectory_id" INTEGER NOT NULL,
                "step"          INTEGER NOT NULL,
                "obs"           BLOB,
                "action"        INTEGER,
                "reward"        NUMERIC,
                "value"         NUMERIC,
                "log_probs"     BLOB,
                PRIMARY KEY("trajectory_id", "step"),
                FOREIGN KEY("trajectory_id") REFERENCES trajectories(id)
        );
    """)
    con.execute('pragma journal_mode=off')
    con.commit()
    return con
    
class TrajectoryWriter:
    def __init__(self, database_path):
        self.db = ensure_database_initialized(database_path)
        self.needs_reset = True
        self.last_commit = time.time()

    def begin_trajectory(self):
        self.trajectory_id = str(ULID())
        self.db.execute("INSERT INTO trajectories (id) VALUES (?)", (self.trajectory_id,))
        self.step = 0

    def end_trajectory(self, total_reward: float | None, length: int | None, other: str):
        self.db.execute("UPDATE trajectories SET total_reward = ?, length = ?, other = ?, finished_at = ? WHERE id = ?", (total_reward, length, other, time.time(), self.trajectory_id))
        self.db.commit()

    def insert_observation(self, obs: bytes, action: int, reward: float, value: float, log_probs: bytes):
        self.db.execute(
            r"""INSERT INTO observations (trajectory_id, step, obs, action, reward, value, log_probs) VALUES (?, ?, ?, ?, ?, ?, ?)""", 
            (self.trajectory_id, self.step, obs, action, reward, value, log_probs)        
        )

        now = time.time()
        if time.time() - self.last_commit > 4.0:
            self.db.commit()
            self.last_commit = now
        
        self.step += 1

    def record(self, obs: np.ndarray, action: int, reward: float, value: float, log_probs: np.ndarray, terminated: bool, truncated: bool, info):
        if self.needs_reset:
            self.begin_trajectory()
            self.needs_reset = False

        self.insert_observation(util.save_ndarray(obs), action, reward, value, util.save_ndarray(log_probs))

        if terminated or truncated:
            self.needs_reset = True
            
            maybe_ep_info = info.get("episode", {})
            total_len = None
            total_reward = None
            if maybe_ep_info is not None:
                total_len = maybe_ep_info["l"]
                total_reward = maybe_ep_info["r"]
            
            maybe_ep_info["truncated"] = bool(truncated)
            self.end_trajectory(total_reward, total_len, json.dumps(maybe_ep_info))