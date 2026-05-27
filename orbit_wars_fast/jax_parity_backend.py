from __future__ import annotations

import multiprocessing as mp
from typing import Any

import jax.numpy as jnp
import numpy as np

from orbit_wars_fast.fast_sim import FastOrbitWarsSimulator
from orbit_wars_fast.reference_adapter import ReferenceOrbitWarsSimulator


def _pad_planets(planets: list[list[Any]], max_planets: int) -> dict[str, np.ndarray]:
    n = min(len(planets), max_planets)
    out = {
        "id": np.full((max_planets,), -1, dtype=np.int32),
        "owner": np.full((max_planets,), -1, dtype=np.int32),
        "x": np.zeros((max_planets,), dtype=np.float32),
        "y": np.zeros((max_planets,), dtype=np.float32),
        "radius": np.zeros((max_planets,), dtype=np.float32),
        "ships": np.zeros((max_planets,), dtype=np.float32),
        "production": np.zeros((max_planets,), dtype=np.float32),
        "alive": np.zeros((max_planets,), dtype=np.bool_),
    }
    for i in range(n):
        p = planets[i]
        out["id"][i] = int(p[0])
        out["owner"][i] = int(p[1])
        out["x"][i] = float(p[2])
        out["y"][i] = float(p[3])
        out["radius"][i] = float(p[4])
        out["ships"][i] = float(p[5])
        out["production"][i] = float(p[6])
        out["alive"][i] = True
    return out


def _pad_fleets(fleets: list[list[Any]], max_fleets: int) -> dict[str, np.ndarray]:
    n = min(len(fleets), max_fleets)
    out = {
        "id": np.full((max_fleets,), -1, dtype=np.int32),
        "owner": np.full((max_fleets,), -1, dtype=np.int32),
        "x": np.zeros((max_fleets,), dtype=np.float32),
        "y": np.zeros((max_fleets,), dtype=np.float32),
        "angle": np.zeros((max_fleets,), dtype=np.float32),
        "from_planet_id": np.full((max_fleets,), -1, dtype=np.int32),
        "ships": np.zeros((max_fleets,), dtype=np.float32),
        "alive": np.zeros((max_fleets,), dtype=np.bool_),
    }
    for i in range(n):
        f = fleets[i]
        out["id"][i] = int(f[0])
        out["owner"][i] = int(f[1])
        out["x"][i] = float(f[2])
        out["y"][i] = float(f[3])
        out["angle"][i] = float(f[4])
        out["from_planet_id"][i] = int(f[5])
        out["ships"][i] = float(f[6])
        out["alive"][i] = True
    return out


def _worker_loop(conn: mp.connection.Connection, num_agents: int, seed: int | None):
    sim = FastOrbitWarsSimulator(num_agents=num_agents, seed=seed, use_numba=True)
    conn.send(sim.observations())
    while True:
        try:
            cmd, data = conn.recv()
        except EOFError:
            break
        if cmd == "step":
            obs, done, reward = sim.step(data)
            conn.send((obs, done, reward))
        elif cmd == "snapshot":
            conn.send(sim.snapshot())
        elif cmd == "reset":
            obs = sim.reset(data)
            conn.send(obs)
        elif cmd == "close":
            break


class _ParallelParityAdapter:
    def __init__(self, num_agents: int, seed: int | None):
        self.parent_conn, child_conn = mp.Pipe()
        self.process = mp.Process(target=_worker_loop, args=(child_conn, num_agents, seed))
        self.process.start()
        self._obs = self.parent_conn.recv()
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, seed: int | None = None) -> list[dict[str, Any]]:
        self.parent_conn.send(("reset", seed))
        self._obs = self.parent_conn.recv()
        self._done = False
        return self._obs

    def observations(self) -> list[dict[str, Any]]:
        return self._obs

    def step(self, actions: list[list[list[Any]]]) -> tuple[list[dict[str, Any]], bool, list[float]]:
        self.parent_conn.send(("step", actions))
        self._obs, self._done, reward = self.parent_conn.recv()
        return self._obs, self._done, reward

    def snapshot(self) -> dict[str, Any]:
        self.parent_conn.send(("snapshot", None))
        return self.parent_conn.recv()

    def close(self) -> None:
        if hasattr(self, "parent_conn"):
            try:
                self.parent_conn.send(("close", None))
                self.process.join()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()


class JAXParityOrbitWarsSimulator:
    """Parity-first backend with JAX-friendly tensor observations.

    Transition dynamics are delegated to the official interpreter via
    `ReferenceOrbitWarsSimulator`, preserving simulator parity first.
    """

    def __init__(
        self,
        num_agents: int = 2,
        seed: int | None = None,
        max_planets: int = 64,
        max_fleets: int = 1024,
        use_parallel_parity_numba: bool = False,
    ):
        self.num_agents = num_agents
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self._use_parallel = use_parallel_parity_numba

        if use_parallel_parity_numba:
            self._sim = _ParallelParityAdapter(num_agents=num_agents, seed=seed)
        else:
            self._sim = ReferenceOrbitWarsSimulator(num_agents=num_agents, seed=seed)

    @property
    def done(self) -> bool:
        return self._sim.done

    def reset(self, seed: int | None = None) -> list[dict[str, Any]]:
        return self._sim.reset(seed=seed)

    def observations(self) -> list[dict[str, Any]]:
        return self._sim.observations()

    def observations_jax(self) -> dict[str, Any]:
        obs = self._sim.observations()
        planets_np = []
        fleets_np = []
        scalars = {
            "player": [],
            "angular_velocity": [],
            "next_fleet_id": [],
            "step": [],
            "remainingOverageTime": [],
        }

        for o in obs:
            p = _pad_planets(o["planets"], self.max_planets)
            f = _pad_fleets(o["fleets"], self.max_fleets)
            planets_np.append(p)
            fleets_np.append(f)
            for k in scalars:
                scalars[k].append(o[k])

        planets = {
            k: jnp.asarray(np.stack([p[k] for p in planets_np])) for k in planets_np[0]
        }
        fleets = {
            k: jnp.asarray(np.stack([f[k] for f in fleets_np])) for k in fleets_np[0]
        }
        scalar_tensors = {
            "player": jnp.asarray(np.asarray(scalars["player"], dtype=np.int32)),
            "angular_velocity": jnp.asarray(
                np.asarray(scalars["angular_velocity"], dtype=np.float32)
            ),
            "next_fleet_id": jnp.asarray(
                np.asarray(scalars["next_fleet_id"], dtype=np.int32)
            ),
            "step": jnp.asarray(np.asarray(scalars["step"], dtype=np.int32)),
            "remainingOverageTime": jnp.asarray(
                np.asarray(scalars["remainingOverageTime"], dtype=np.float32)
            ),
        }

        return {
            "planets": planets,
            "fleets": fleets,
            **scalar_tensors,
        }

    def step(
        self, actions: list[list[list[Any]]]
    ) -> tuple[list[dict[str, Any]], bool, list[float]]:
        return self._sim.step(actions)

    def snapshot(self) -> dict[str, Any]:
        return self._sim.snapshot()

    def close(self) -> None:
        if getattr(self, "_use_parallel", False):
            self._sim.close()

    def __del__(self) -> None:
        if getattr(self, "_use_parallel", False):
            self._sim.close()
