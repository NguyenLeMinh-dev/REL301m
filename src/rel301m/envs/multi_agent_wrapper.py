"""Two-agent observation/action adapter for the verified Phase 1 environment."""

import json
from pathlib import Path

import numpy as np

from .contract import inspect_actions, inspect_observations, validate_action, validate_observations
from .robosuite_factory import PROJECT_ROOT


class MultiAgentWrapper:
    """Return actor-safe dictionaries; expose critic state through a separate property.

    reset() -> agent observations
    step(agent_0_action, agent_1_action) -> (agent observations, reward, done, info)

    Phase 1 snapshots are checked against the live environment. Dimensions are
    derived from those verified shapes and robot/controller action metadata.
    """

    agent_ids = ("agent_0", "agent_1")
    shared_object_keys = ("pot_pos", "pot_quat", "handle0_xpos", "handle1_xpos")

    def __init__(self, env, contract_dir=PROJECT_ROOT / "artifacts"):
        self._env = env
        contract_dir = Path(contract_dir)
        with (contract_dir / "observation_spec.json").open(encoding="utf-8") as stream:
            self._observation_contract = json.load(stream)
        with (contract_dir / "action_spec.json").open(encoding="utf-8") as stream:
            self._action_contract = json.load(stream)
        if len(env.robots) != len(self.agent_ids):
            raise ValueError("The two-agent contract requires two independent robots")
        self._validate_live_action_contract()
        self._local_keys = {
            agent: (robot.robot_model.naming_prefix + "proprio-state", f"gripper{index}_to_handle{index}")
            for index, (agent, robot) in enumerate(zip(self.agent_ids, env.robots))
        }
        self._critic_keys = tuple(self._local_keys[agent][0] for agent in self.agent_ids) + ("object-state",)
        observations = self._observation_contract["observations"]
        required = set(self.shared_object_keys + self._critic_keys)
        required.update(key for keys in self._local_keys.values() for key in keys)
        for key in required:
            if key not in observations or len(observations[key]["shape"]) != 1:
                raise ValueError(f"Phase 1 contract requires a state vector for {key!r}")
        shared_dim = sum(observations[key]["shape"][0] for key in self.shared_object_keys)
        self._observation_dims = {}
        for agent, keys in self._local_keys.items():
            local_dim = sum(observations[key]["shape"][0] for key in keys)
            self._observation_dims[agent] = {
                "local": local_dim, "shared_object": shared_dim, "actor_obs": local_dim + shared_dim,
            }
        self._critic_state_dim = sum(observations[key]["shape"][0] for key in self._critic_keys)
        self._critic_state = None

    def _validate_live_action_contract(self):
        live = inspect_actions(self._env)
        if live != self._action_contract:
            raise ValueError("Live action metadata does not match Phase 1 action_spec.json")
        return live

    def _validate_raw_observations(self, raw):
        validate_observations(raw)
        if list(raw) != self._observation_contract["returned_keys"]:
            raise ValueError("Observation keys/order do not match the Phase 1 contract")
        for key, spec in self._observation_contract["observations"].items():
            array = np.asarray(raw[key])
            if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
                raise ValueError(f"Observation shape/dtype for {key!r} does not match Phase 1")

    def _actor_observations(self, raw):
        shared = np.concatenate([raw[key] for key in self.shared_object_keys])
        agents = {}
        for agent in self.agent_ids:
            local = np.concatenate([raw[key] for key in self._local_keys[agent]])
            # Every returned array owns independent data; actor mutation cannot alter
            # raw observations, another agent's inputs, or the critic's cached state.
            agents[agent] = {
                "local": local,
                "shared_object": shared.copy(),
                "actor_obs": np.concatenate((local, shared)),
            }
        return agents

    def split_observations(self, raw):
        """Pure actor transformation: no raw state mutation or critic state exposure."""
        self._validate_raw_observations(raw)
        return self._actor_observations(raw)

    def build_critic_state(self, raw):
        """Pure training-only transformation of a supplied raw observation dictionary."""
        self._validate_raw_observations(raw)
        return np.concatenate([raw[key] for key in self._critic_keys])

    def _process_observations(self, raw):
        self._validate_raw_observations(raw)
        self._critic_state = np.concatenate([raw[key] for key in self._critic_keys])
        return self._actor_observations(raw)

    @property
    def observation_dims(self):
        return {agent: dict(dims) for agent, dims in self._observation_dims.items()}

    @property
    def critic_state_dim(self):
        return self._critic_state_dim

    @property
    def critic_state(self):
        """Training-only state from the latest reset/step; never part of actor outputs."""
        if self._critic_state is None:
            raise RuntimeError("Call reset() before requesting critic_state")
        return self._critic_state.copy()

    @property
    def action_dims(self):
        live = self._validate_live_action_contract()
        return {agent: robot["action_dim"] for agent, robot in zip(self.agent_ids, live["robots"])}

    @property
    def action_specs(self):
        """Per-agent live lower/upper bounds, returned as independent arrays."""
        live = self._validate_live_action_contract()
        low, high = np.asarray(live["low"]), np.asarray(live["high"])
        specs = {}
        for agent, robot in zip(self.agent_ids, live["robots"]):
            start, stop = robot["joint_slice"]
            specs[agent] = (low[start:stop].copy(), high[start:stop].copy())
        return specs

    def join_actions(self, agent_0_action, agent_1_action):
        """Validate then concatenate in env.robots order, without clipping inputs."""
        bounds = self.action_specs
        actions = []
        for agent, value in zip(self.agent_ids, (agent_0_action, agent_1_action)):
            action = np.asarray(value)
            if action.dtype.kind not in "fiu":
                raise ValueError(f"{agent} action must contain real numeric values")
            try:
                validate_action(action, *bounds[agent])
            except ValueError as error:
                raise ValueError(f"{agent}: {error}") from error
            actions.append(action)
        joint = np.concatenate(actions)
        validate_action(joint, np.asarray(self._action_contract["low"]), np.asarray(self._action_contract["high"]))
        return joint

    def reset(self):
        raw = self._env.reset()
        if inspect_observations(self._env, raw) != self._observation_contract:
            raise ValueError("Live observations/observables do not match Phase 1 observation_spec.json")
        return self._process_observations(raw)

    def step(self, agent_0_action, agent_1_action):
        joint_action = self.join_actions(agent_0_action, agent_1_action)
        raw, reward, done, info = self._env.step(joint_action)
        return self._process_observations(raw), reward, done, info

    def check_success(self):
        """Use the upstream task predicate, independently of reward and done."""
        return bool(self._env._check_success())

    @property
    def horizon(self):
        return self._env.horizon

    @property
    def control_freq(self):
        return self._env.control_freq

    def close(self):
        self._env.close()
