from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import kaggle_environments.envs.orbit_wars.orbit_wars as orbit_wars


def _resolved_config() -> dict[str, Any]:
    cfg_raw = dict(orbit_wars.specification.get("configuration", {}))
    cfg: dict[str, Any] = {}
    for k, v in cfg_raw.items():
        cfg[k] = v.get("default") if isinstance(v, dict) and "default" in v else v
    return cfg


@dataclass
class AgentState:
    observation: SimpleNamespace
    action: list
    status: str = "ACTIVE"
    reward: float = 0.0


class ReferenceOrbitWarsSimulator:
    """Thin adapter around Kaggle's official `orbit_wars.interpreter`.

    This gives us a deterministic, step-by-step reference implementation
    without going through the full Kaggle environment runtime.
    """

    def __init__(self, num_agents: int = 2, seed: int | None = None):
        if num_agents not in (2, 4):
            raise ValueError("Orbit Wars supports only 2 or 4 agents")
        self.num_agents = num_agents
        cfg = _resolved_config()
        if seed is not None:
            cfg["seed"] = int(seed)
        self.env = SimpleNamespace(
            configuration=SimpleNamespace(**cfg), info={}, done=False
        )
        self.state: list[AgentState] = []
        self.step_count = 0
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> list[dict[str, Any]]:
        cfg = _resolved_config()
        if seed is not None:
            cfg["seed"] = int(seed)
        self.env = SimpleNamespace(
            configuration=SimpleNamespace(**cfg), info={}, done=False
        )
        self.step_count = 0
        self.state = [
            AgentState(
                observation=SimpleNamespace(step=0, remainingOverageTime=60), action=[]
            )
            for _ in range(self.num_agents)
        ]
        self.state = orbit_wars.interpreter(self.state, self.env)
        self._sync_step()
        return self.observations()

    def _sync_step(self) -> None:
        for s in self.state:
            s.observation.step = self.step_count
            if not hasattr(s.observation, "remainingOverageTime"):
                s.observation.remainingOverageTime = 60

    def observations(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, s in enumerate(self.state):
            obs = s.observation
            out.append(
                {
                    "planets": obs.planets,
                    "fleets": obs.fleets,
                    "player": i,
                    "angular_velocity": obs.angular_velocity,
                    "initial_planets": obs.initial_planets,
                    "next_fleet_id": obs.next_fleet_id,
                    "comets": obs.comets,
                    "comet_planet_ids": obs.comet_planet_ids,
                    "step": self.step_count,
                    "remainingOverageTime": 60,
                }
            )
        return out

    def step(
        self, actions: list[list[list[Any]]]
    ) -> tuple[list[dict[str, Any]], bool, list[float]]:
        if self.done:
            return self.observations(), True, [float(s.reward) for s in self.state]

        for i, action in enumerate(actions):
            self.state[i].action = action

        self.state = orbit_wars.interpreter(self.state, self.env)
        self.step_count += 1
        self._sync_step()
        self.env.done = all(s.status == "DONE" for s in self.state)
        return self.observations(), self.done, [float(s.reward) for s in self.state]

    @property
    def done(self) -> bool:
        return all(s.status == "DONE" for s in self.state)

    def snapshot(self) -> dict[str, Any]:
        obs = self.state[0].observation
        return {
            "step": self.step_count,
            "done": self.done,
            "planets": obs.planets,
            "fleets": obs.fleets,
            "initial_planets": obs.initial_planets,
            "next_fleet_id": obs.next_fleet_id,
            "comets": obs.comets,
            "comet_planet_ids": obs.comet_planet_ids,
            "angular_velocity": obs.angular_velocity,
            "status": [s.status for s in self.state],
            "reward": [float(s.reward) for s in self.state],
        }
