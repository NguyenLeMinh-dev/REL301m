# Phase 1 — Environment Contract

Generated from a live `TwoArmLift.reset()` by `scripts/inspect_env.py`.
Observation/action dimensions and status flags are measurements; no dictionary flattening is applied.
Measured at: 2026-10-02T03:21:52.618947+00:00.

## Configuration and versions

```yaml
env_name: TwoArmLift
robots:
- Panda
- Panda
env_configuration: opposed
controller: BASIC
has_renderer: false
has_offscreen_renderer: false
use_camera_obs: false
use_object_obs: true
reward_shaping: true
reward_scale: 1.0
control_freq: 20
horizon: 200
ignore_done: false
seed: 0
```

```json
{
  "python": "3.10.12",
  "robosuite": "1.5.2",
  "mujoco": "3.9.0",
  "numpy": "1.26.4"
}
```

## Observations

| Key | Shape | Dtype | Elements | Information group | Semantics |
| --- | --- | --- | ---: | --- | --- |
| `robot0_joint_pos` | `(7,)` | `float64` | 7 | robot_0_local | Arm joint positions (rad). |
| `robot0_joint_pos_cos` | `(7,)` | `float64` | 7 | robot_0_local | Cosine of arm joint positions (unitless). |
| `robot0_joint_pos_sin` | `(7,)` | `float64` | 7 | robot_0_local | Sine of arm joint positions (unitless). |
| `robot0_joint_vel` | `(7,)` | `float64` | 7 | robot_0_local | Arm joint velocities (rad/s). |
| `robot0_joint_acc` | `(7,)` | `float64` | 7 | robot_0_local | Arm joint accelerations (rad/s^2), simulator-derived. |
| `robot0_eef_pos` | `(3,)` | `float64` | 3 | robot_0_local | EEF site position in world coordinates (m). |
| `robot0_eef_quat` | `(4,)` | `float64` | 4 | robot_0_local | Legacy EEF BODY quaternion in world coordinates, xyzw; not the EEF site orientation. |
| `robot0_eef_quat_site` | `(4,)` | `float32` | 4 | robot_0_local | EEF SITE quaternion in world coordinates, xyzw; consistent with eef_pos. |
| `robot0_gripper_qpos` | `(2,)` | `float64` | 2 | robot_0_local | Gripper finger joint positions (m for Panda's prismatic finger joints). |
| `robot0_gripper_qvel` | `(2,)` | `float64` | 2 | robot_0_local | Gripper finger joint velocities (m/s). |
| `robot1_joint_pos` | `(7,)` | `float64` | 7 | robot_1_local | Arm joint positions (rad). |
| `robot1_joint_pos_cos` | `(7,)` | `float64` | 7 | robot_1_local | Cosine of arm joint positions (unitless). |
| `robot1_joint_pos_sin` | `(7,)` | `float64` | 7 | robot_1_local | Sine of arm joint positions (unitless). |
| `robot1_joint_vel` | `(7,)` | `float64` | 7 | robot_1_local | Arm joint velocities (rad/s). |
| `robot1_joint_acc` | `(7,)` | `float64` | 7 | robot_1_local | Arm joint accelerations (rad/s^2), simulator-derived. |
| `robot1_eef_pos` | `(3,)` | `float64` | 3 | robot_1_local | EEF site position in world coordinates (m). |
| `robot1_eef_quat` | `(4,)` | `float64` | 4 | robot_1_local | Legacy EEF BODY quaternion in world coordinates, xyzw; not the EEF site orientation. |
| `robot1_eef_quat_site` | `(4,)` | `float32` | 4 | robot_1_local | EEF SITE quaternion in world coordinates, xyzw; consistent with eef_pos. |
| `robot1_gripper_qpos` | `(2,)` | `float64` | 2 | robot_1_local | Gripper finger joint positions (m for Panda's prismatic finger joints). |
| `robot1_gripper_qvel` | `(2,)` | `float64` | 2 | robot_1_local | Gripper finger joint velocities (m/s). |
| `pot_pos` | `(3,)` | `float64` | 3 | shared_object | Pot body position in MuJoCo world coordinates (m). |
| `pot_quat` | `(4,)` | `float64` | 4 | shared_object | Pot body orientation in world coordinates, quaternion xyzw. |
| `handle0_xpos` | `(3,)` | `float64` | 3 | shared_object | Handle 0 site position in world coordinates (m). |
| `handle1_xpos` | `(3,)` | `float64` | 3 | shared_object | Handle 1 site position in world coordinates (m). |
| `gripper0_to_handle0` | `(3,)` | `float64` | 3 | robot_0_object_relation | Handle 0 minus robot 0 EEF site position, world-frame vector (m). |
| `gripper1_to_handle1` | `(3,)` | `float64` | 3 | robot_1_object_relation | Handle 1 minus robot 1 EEF site position, world-frame vector (m). |
| `robot0_proprio-state` | `(50,)` | `float64` | 50 | robot_0_local | Upstream concatenation of active/enabled proprioceptive sensors, including redundant encodings. |
| `robot1_proprio-state` | `(50,)` | `float64` | 50 | robot_1_local | Upstream concatenation of active/enabled proprioceptive sensors, including redundant encodings. |
| `object-state` | `(19,)` | `float64` | 19 | joint_object_aggregate | Concatenation of active/enabled object sensors; includes both robots' relative-handle vectors. |

The raw dictionary is the contract. Each `*-state` vector is generated upstream
by concatenating returned sensors of the corresponding modality, in sensor order.
Its components are recorded in `observation_spec.json` as `aggregation_of`.
Do not concatenate individual sensor keys together with their aggregate vectors:
this would duplicate features. Joint angles, cosine/sine encodings, and EEF
body/site orientations also contain related information even within aggregates.

### Observable status

Enabled means computed/updated; active means selected for return. A sensor
appears in observations only when both flags are true. Aggregates are not
independent registered Observable objects. These statuses are introspected
without enabling, disabling, or otherwise modifying observables.

| Registered observable | Modality | Enabled | Active | Returned |
| --- | --- | --- | --- | --- |
| `robot0_joint_pos` | `robot0_proprio` | True | True | True |
| `robot0_joint_pos_cos` | `robot0_proprio` | True | True | True |
| `robot0_joint_pos_sin` | `robot0_proprio` | True | True | True |
| `robot0_joint_vel` | `robot0_proprio` | True | True | True |
| `robot0_joint_acc` | `robot0_proprio` | True | True | True |
| `robot0_eef_pos` | `robot0_proprio` | True | True | True |
| `robot0_eef_quat` | `robot0_proprio` | True | True | True |
| `robot0_eef_quat_site` | `robot0_proprio` | True | True | True |
| `robot0_gripper_qpos` | `robot0_proprio` | True | True | True |
| `robot0_gripper_qvel` | `robot0_proprio` | True | True | True |
| `robot1_joint_pos` | `robot1_proprio` | True | True | True |
| `robot1_joint_pos_cos` | `robot1_proprio` | True | True | True |
| `robot1_joint_pos_sin` | `robot1_proprio` | True | True | True |
| `robot1_joint_vel` | `robot1_proprio` | True | True | True |
| `robot1_joint_acc` | `robot1_proprio` | True | True | True |
| `robot1_eef_pos` | `robot1_proprio` | True | True | True |
| `robot1_eef_quat` | `robot1_proprio` | True | True | True |
| `robot1_eef_quat_site` | `robot1_proprio` | True | True | True |
| `robot1_gripper_qpos` | `robot1_proprio` | True | True | True |
| `robot1_gripper_qvel` | `robot1_proprio` | True | True | True |
| `pot_pos` | `object` | True | True | True |
| `pot_quat` | `object` | True | True | True |
| `handle0_xpos` | `object` | True | True | True |
| `handle1_xpos` | `object` | True | True | True |
| `gripper0_to_handle0` | `object` | True | True | True |
| `gripper1_to_handle1` | `object` | True | True | True |

Complete enabled observable names:

```json
[
  "gripper0_to_handle0",
  "gripper1_to_handle1",
  "handle0_xpos",
  "handle1_xpos",
  "pot_pos",
  "pot_quat",
  "robot0_eef_pos",
  "robot0_eef_quat",
  "robot0_eef_quat_site",
  "robot0_gripper_qpos",
  "robot0_gripper_qvel",
  "robot0_joint_acc",
  "robot0_joint_pos",
  "robot0_joint_pos_cos",
  "robot0_joint_pos_sin",
  "robot0_joint_vel",
  "robot1_eef_pos",
  "robot1_eef_quat",
  "robot1_eef_quat_site",
  "robot1_gripper_qpos",
  "robot1_gripper_qvel",
  "robot1_joint_acc",
  "robot1_joint_pos",
  "robot1_joint_pos_cos",
  "robot1_joint_pos_sin",
  "robot1_joint_vel"
]
```

Complete active observable names:

```json
[
  "gripper0_to_handle0",
  "gripper1_to_handle1",
  "handle0_xpos",
  "handle1_xpos",
  "pot_pos",
  "pot_quat",
  "robot0_eef_pos",
  "robot0_eef_quat",
  "robot0_eef_quat_site",
  "robot0_gripper_qpos",
  "robot0_gripper_qvel",
  "robot0_joint_acc",
  "robot0_joint_pos",
  "robot0_joint_pos_cos",
  "robot0_joint_pos_sin",
  "robot0_joint_vel",
  "robot1_eef_pos",
  "robot1_eef_quat",
  "robot1_eef_quat_site",
  "robot1_gripper_qpos",
  "robot1_gripper_qvel",
  "robot1_joint_acc",
  "robot1_joint_pos",
  "robot1_joint_pos_cos",
  "robot1_joint_pos_sin",
  "robot1_joint_vel"
]
```

### Future information-access contract

The joint environment currently returns both robots' observations. The roles
below describe future actor/critic access, not an implemented multi-agent wrapper.

| Role | Candidate signals |
| --- | --- |
| Agent 0 local observations | Robot 0 proprioception and its own gripper-to-handle relation |
| Agent 1 local observations | Robot 1 proprioception and its own gripper-to-handle relation |
| Teammate information | The other robot's proprioception/relation; future explicit message channel |
| Shared/object information | Pot pose and both handle positions; currently simulator ground truth |
| Centralized critic global state | Both robot aggregate states plus object-state; training-only candidate |

Measured aggregate dimensions: 50 + 50 + 19 = **119**
for that centralized-critic candidate. This sum does not imply independent features.

`object-state` contains both gripper-to-handle vectors. Giving the full vector
to both actors can expose teammate state through those vectors, even when
teammate messages are dropped. Actor access must be partitioned explicitly
before future communication experiments. No actor input or privileged `qpos/qvel`
critic state is constructed in this phase.

## Action contract

`env.action_spec` is the official `(low, high)` vector pair; `env.action_dim`
is the official total dimension. Robot order follows `env.robots`. Body-part
slices below come from each composite controller's public `get_action_info_dict()`
and `get_action_info()` methods; all indices have an exclusive stop.

Total action dimension: **14**.

Complete `action_spec.low`: `[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]`

Complete `action_spec.high`: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

| Robot | Composite controller | Dimension | Joint action slice |
| --- | --- | ---: | --- |
| 0 (Panda0) | `BASIC` | 7 | `0:7` |
| 1 (Panda1) | `BASIC` | 7 | `7:14` |

| Robot | Part | Controller | Dimension | Robot-local slice | Joint slice | Input type/frame |
| --- | --- | --- | ---: | --- | --- | --- |
| 0 | `right` | `OSC_POSE` | 6 | `0:6` | `0:6` | delta / base |
| 0 | `right_gripper` | `GRIP` | 1 | `6:7` | `6:7` | n/a / n/a |
| 1 | `right` | `OSC_POSE` | 6 | `0:6` | `7:13` | delta / base |
| 1 | `right_gripper` | `GRIP` | 1 | `6:7` | `13:14` | n/a / n/a |

The measured Panda layout uses OSC_POSE delta inputs (translation and
axis-angle rotation) followed by the GRIP command. These are normalized
controller commands; the action dimension does not count arm joints plus
physical finger joints. Derive future wrapper partitions from the metadata.

## Reward, success, and termination

The following formulas were audited against the installed robosuite source.
The complete methods, source paths, line numbers, and SHA-256 hashes are stored
in `artifacts/env_spec.json` under `source_evidence`.

Let `h = site_z[pot_center] - pot.top_offset[2] - site_z[table_top]`.
The upstream success predicate is strictly **`h > 0.10 m`**.
Success has no tilt or grasp requirement. Never define success from episode return.

Let `c = 1` when the pot's local z-axis has world-z dot product
`>= cos(pi / 6)` (tilt at most 30 degrees), otherwise `c = 0`.
`d0/d1` are gripper-to-corresponding-handle distances; `g0/g1` are the upstream
handle-grasp checks (0 or 1). The installed reward implementation is:

```text
if success:
    raw_reward = 3 * c
elif reward_shaping:
    raw_reward = 10 * c * clip(h - 0.05, 0, 0.15)
                 + 0.25 * (g0 + g1)
                 + 0.5 * (1 - tanh(10 * d0))
                 + 0.5 * (1 - tanh(10 * d1))
else:
    raw_reward = 0
reward = raw_reward * reward_scale / 3  # when reward_scale is not None
```

Measured `reward_scale`: `1.0`.
The success branch replaces the shaped terms. A high-enough pot tilted
more than 30 degrees can therefore satisfy task success while yielding zero
reward. This distinction is covered by a real-simulator contract test.

`done = (timestep >= horizon) and not ignore_done` is independent of success.
The current configuration continues to horizon even if success happens earlier.
At 20 Hz, 200 control steps represent
10 seconds of simulated time.
Step uses robosuite's four-value API: `(observations, reward, done, info)`.

### Episode metrics

| Metric | Definition |
| --- | --- |
| `episode_return` | Sum of step rewards; independent of success. |
| `ever_success` | True if upstream task success is seen at reset or after any step. |
| `first_success_step` | 0 for success at reset, otherwise first successful 1-based step; null if never successful. |
| `final_success` | Upstream task success after the final step. |
| `episode_length` | Number of control steps actually executed. |

The rollout checks success at reset and after every step. `first_success_step`
uses 0 for reset, 1-based control-step indices otherwise, and JSON null when
success never occurs. Losing success later changes `final_success` without
erasing `ever_success`. Rewards/returns never infer these flags.

## Reproduce and verify

Run from the REL301m research root after editable installation:

```bash
python -I scripts/inspect_env.py
python -I scripts/random_rollout.py --episodes 3 --seed 0
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -m pytest -q tests
```

The inspector writes the three tracked JSON contracts under `artifacts/` and
regenerates this Markdown. Tests compare live dimensions/dtypes/layouts with
the saved contracts, verify sensor semantics and reward/success separation,
and check full-horizon termination. No camera, rendering context, PyTorch,
communication corruption, policy learning, or teammate prediction is required.

Official references: [environment API](https://robosuite.ai/docs/source/robosuite.environments.html),
[TwoArmLift API](https://robosuite.ai/docs/source/robosuite.environments.manipulation.html),
[composite controllers](https://robosuite.ai/docs/modules/controllers.html).
