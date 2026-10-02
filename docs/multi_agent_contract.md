# Phase 2 — Multi-Agent Observation / Action Contract

`MultiAgentWrapper` adapts the verified state-only `TwoArmLift` environment into
an explicit two-agent interface. It reads the Phase 1 observation/action snapshots
in `artifacts/`, compares live controller metadata before accepting actions, and
validates raw observation keys, shapes, dtypes, and finite values on reset/step.
The Phase 1 contract remains unchanged.

## Actor observations

`reset()` and `step()` return an agent dictionary with exactly `agent_0` and
`agent_1`. Each agent has exactly `local`, `shared_object`, and `actor_obs`.
Source order is part of the contract:

| Agent | Field | Sources, in order | Measured dimension |
| --- | --- | --- | ---: |
| agent_0 | local | robot0_proprio-state, gripper0_to_handle0 | 50 + 3 = 53 |
| agent_1 | local | robot1_proprio-state, gripper1_to_handle1 | 50 + 3 = 53 |
| both | shared_object | pot_pos, pot_quat, handle0_xpos, handle1_xpos | 3 + 4 + 3 + 3 = 13 |
| each | actor_obs | own local, shared_object | 53 + 13 = 66 |

The dimensions above are validated by the live tests. Production code derives
sizes from the measured Phase 1 shapes and rejects live schema mismatches.
It does not embed 53/66/119 or robot action dimensions as slicing constants.

Actor construction selects these source keys explicitly. It never reads raw
`object-state`, the other robot's proprioception, or the other robot's relative
handle vector. Changing any of those private sources leaves the other agent's
entire output unchanged; changing only `object-state` leaves both actor outputs
unchanged. These source-isolation tests use counterfactual sensor values to check
data access independently of physical correlations.

`shared_object` remains common simulator ground truth. Its physical dependence on
both robots is expected by this contract. "No leakage" here means no additional
direct private/relative-state fields or array aliases enter actor outputs; it is
not a security boundary against Python code that deliberately accesses internals.
Do not pass the coordinator's wrapper object or critic state to actor policies.

Every output array is independently allocated. Mutating one agent's arrays cannot
change raw observations, the other agent's arrays, the same agent's other arrays,
or the cached critic state. The wrapper does not alter, remove, replace, or
flatten the original raw observation dictionary.

## Training-only centralized state

`wrapper.critic_state` is separate from the returned agent dictionary and returns
an independent copy of the most recent reset/step state:

```text
robot0_proprio-state + robot1_proprio-state + object-state
50 + 50 + 19 = 119
```

The critic includes both relative-handle vectors through the upstream object
aggregate. Reading `critic_state` before `reset()` raises `RuntimeError`.
`build_critic_state(raw)` provides the same pure transformation for offline
contract inspection; `split_observations(raw)` provides the actor-only pure
transformation. Neither modifies `raw` or the cached critic state.

## Actions

`step(agent_0_action, agent_1_action)` takes one real numeric vector per robot.
`join_actions(...)` performs the same validation/concatenation without stepping.
Robot order follows `env.robots` and the live public composite-controller metadata.

| Agent | Measured dimension | Joint slice, stop exclusive |
| --- | ---: | --- |
| agent_0 | 7 | 0:7 |
| agent_1 | 7 | 7:14 |
| joint | 14 | 0:14 |

Each Panda currently has six OSC_POSE inputs and one GRIP input. These sizes and
bounds are derived from `inspect_actions(env)` and checked against Phase 1;
they are not inferred from physical joint/finger counts.

`action_dims` reports each live robot dimension. `action_specs` returns each
agent's `(low, high)` vectors as copies. Current bounds are [-1, 1] per component.
Each join checks the current environment/controller metadata again, then checks
shape, finite real numeric values, and each robot's actual bounds. Wrong vector
sizes, matrices, NaN/Inf, nonnumeric values, and out-of-range actions are rejected
before `env.step()` runs. Actions are concatenated without clipping or padding.

## Reset/step and task semantics

```text
reset() -> agents
step(action0, action1) -> agents, reward, done, info
```

Reward remains the original shared scalar. The original `done` and `info` are
returned directly, without per-agent duplication, injected success flags, or
additional termination conditions. `check_success()` delegates to upstream
`env._check_success()` and never examines reward or episode return.

Height-based success, tilt-gated reward, and horizon termination retain the Phase 1
semantics. A pot high enough but tilted 90 degrees can have success=true and
reward=0; the wrapper continues the episode. Future logging must still separate
`episode_return`, `ever_success`, `first_success_step`, `final_success`, and
`episode_length`. The wrapper adds no algorithm or communication perturbation.

## Usage

Run from the research root with the existing editable package and Python venv:

```python
import numpy as np
from rel301m.envs.robosuite_factory import make_two_arm_lift
from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper

wrapper = MultiAgentWrapper(make_two_arm_lift())
rng = np.random.default_rng(0)
try:
    agents = wrapper.reset()
    critic_state = wrapper.critic_state  # Coordinator/training only.
    bounds = wrapper.action_specs
    for _ in range(wrapper.horizon):
        action0 = rng.uniform(*bounds["agent_0"])
        action1 = rng.uniform(*bounds["agent_1"])
        agents, reward, done, info = wrapper.step(action0, action1)
        critic_state = wrapper.critic_state
        task_success = wrapper.check_success()
        if done:
            break
finally:
    wrapper.close()
```

## Verification gate

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-phase0/bin/python -I -m pytest -q tests
.venv-phase0/bin/python -I scripts/check_install.py \
  --report experiments/phase2/phase0_regression.md \
  --output-dir experiments/phase2/phase0_regression
```

Phase 2 passes only if observation split, action split, both leakage directions,
nonmutation/storage isolation, a full finite random episode, all previous Phase 1
tests, and the Phase 0 system/simulator check pass. Regression outputs are written
to ignored experiment/log folders so the earlier contract and freeze artifacts
are preserved. No dependencies are added or changed in this phase.
