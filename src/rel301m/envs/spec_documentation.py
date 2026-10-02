"""Render the measured environment contract as Markdown."""

import json
from pathlib import Path

import yaml


def write_environment_spec(path, env_spec, observation_spec, action_spec):
    config = env_spec["config"]
    observations = observation_spec["observations"]
    lines = [
        "# Phase 1 — Environment Contract", "",
        "Generated from a live `TwoArmLift.reset()` by `scripts/inspect_env.py`.",
        "Observation/action dimensions and status flags are measurements; no dictionary flattening is applied.",
        f"Measured at: {env_spec['measured_at']}.", "",
        "## Configuration and versions", "", "```yaml",
        yaml.safe_dump(config, sort_keys=False).rstrip(), "```", "", "```json",
        json.dumps(env_spec["versions"], indent=2), "```", "",
        "## Observations", "",
        "| Key | Shape | Dtype | Elements | Information group | Semantics |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for name, item in observations.items():
        lines.append(f"| `{name}` | `{tuple(item['shape'])}` | `{item['dtype']}` | {item['size']} | {item['information_group']} | {item['description']} |")
    lines += [
        "", "The raw dictionary is the contract. Each `*-state` vector is generated upstream",
        "by concatenating returned sensors of the corresponding modality, in sensor order.",
        "Its components are recorded in `observation_spec.json` as `aggregation_of`.",
        "Do not concatenate individual sensor keys together with their aggregate vectors:",
        "this would duplicate features. Joint angles, cosine/sine encodings, and EEF",
        "body/site orientations also contain related information even within aggregates.", "",
        "### Observable status", "",
        "Enabled means computed/updated; active means selected for return. A sensor",
        "appears in observations only when both flags are true. Aggregates are not",
        "independent registered Observable objects. These statuses are introspected",
        "without enabling, disabling, or otherwise modifying observables.", "",
        "| Registered observable | Modality | Enabled | Active | Returned |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, item in observation_spec["observables"].items():
        lines.append(f"| `{name}` | `{item['modality']}` | {item['enabled']} | {item['active']} | {item['returned']} |")
    lines += ["", "Complete enabled observable names:", "", "```json",
              json.dumps(observation_spec["enabled_observables"], indent=2), "```", "",
              "Complete active observable names:", "", "```json",
              json.dumps(observation_spec["active_observables"], indent=2), "```", "",
              "### Future information-access contract", "",
              "The joint environment currently returns both robots' observations. The roles",
              "below describe future actor/critic access, not an implemented multi-agent wrapper.", "",
              "| Role | Candidate signals |",
              "| --- | --- |",
              "| Agent 0 local observations | Robot 0 proprioception and its own gripper-to-handle relation |",
              "| Agent 1 local observations | Robot 1 proprioception and its own gripper-to-handle relation |",
              "| Teammate information | The other robot's proprioception/relation; future explicit message channel |",
              "| Shared/object information | Pot pose and both handle positions; currently simulator ground truth |",
              "| Centralized critic global state | Both robot aggregate states plus object-state; training-only candidate |", ""]
    aggregate_names = ["robot0_proprio-state", "robot1_proprio-state", "object-state"]
    if all(name in observations for name in aggregate_names):
        sizes = [observations[name]["size"] for name in aggregate_names]
        lines += [f"Measured aggregate dimensions: {sizes[0]} + {sizes[1]} + {sizes[2]} = **{sum(sizes)}**",
                  "for that centralized-critic candidate. This sum does not imply independent features.", ""]
    lines += [
        "`object-state` contains both gripper-to-handle vectors. Giving the full vector",
        "to both actors can expose teammate state through those vectors, even when",
        "teammate messages are dropped. Actor access must be partitioned explicitly",
        "before future communication experiments. No actor input or privileged `qpos/qvel`",
        "critic state is constructed in this phase.", "",
        "## Action contract", "",
        "`env.action_spec` is the official `(low, high)` vector pair; `env.action_dim`",
        "is the official total dimension. Robot order follows `env.robots`. Body-part",
        "slices below come from each composite controller's public `get_action_info_dict()`",
        "and `get_action_info()` methods; all indices have an exclusive stop.", "",
        f"Total action dimension: **{action_spec['action_dim']}**.", "",
        f"Complete `action_spec.low`: `{action_spec['low']}`", "",
        f"Complete `action_spec.high`: `{action_spec['high']}`", "",
        "| Robot | Composite controller | Dimension | Joint action slice |",
        "| --- | --- | ---: | --- |",
    ]
    for robot in action_spec["robots"]:
        start, stop = robot["joint_slice"]
        lines.append(f"| {robot['robot_id']} ({robot['robot_name']}) | `{robot['composite_controller']}` | {robot['action_dim']} | `{start}:{stop}` |")
    lines += ["", "| Robot | Part | Controller | Dimension | Robot-local slice | Joint slice | Input type/frame |",
              "| --- | --- | --- | ---: | --- | --- | --- |"]
    for robot in action_spec["robots"]:
        for part in robot["parts"]:
            start, stop = part["robot_slice"]
            joint_start, joint_stop = part["joint_slice"]
            input_mode = f"{part.get('input_type', 'n/a')} / {part.get('input_ref_frame', 'n/a')}"
            lines.append(f"| {robot['robot_id']} | `{part['name']}` | `{part['controller_type']}` | {part['action_dim']} | `{start}:{stop}` | `{joint_start}:{joint_stop}` | {input_mode} |")
    lines += [
        "", "The measured Panda layout uses OSC_POSE delta inputs (translation and",
        "axis-angle rotation) followed by the GRIP command. These are normalized",
        "controller commands; the action dimension does not count arm joints plus",
        "physical finger joints. Derive future wrapper partitions from the metadata.", "",
        "## Reward, success, and termination", "",
        "The following formulas were audited against the installed robosuite source.",
        "The complete methods, source paths, line numbers, and SHA-256 hashes are stored",
        "in `artifacts/env_spec.json` under `source_evidence`.", "",
        "Let `h = site_z[pot_center] - pot.top_offset[2] - site_z[table_top]`.",
        "The upstream success predicate is strictly **`h > 0.10 m`**.",
        "Success has no tilt or grasp requirement. Never define success from episode return.", "",
        "Let `c = 1` when the pot's local z-axis has world-z dot product",
        "`>= cos(pi / 6)` (tilt at most 30 degrees), otherwise `c = 0`.",
        "`d0/d1` are gripper-to-corresponding-handle distances; `g0/g1` are the upstream",
        "handle-grasp checks (0 or 1). The installed reward implementation is:", "",
        "```text",
        "if success:",
        "    raw_reward = 3 * c",
        "elif reward_shaping:",
        "    raw_reward = 10 * c * clip(h - 0.05, 0, 0.15)",
        "                 + 0.25 * (g0 + g1)",
        "                 + 0.5 * (1 - tanh(10 * d0))",
        "                 + 0.5 * (1 - tanh(10 * d1))",
        "else:",
        "    raw_reward = 0",
        "reward = raw_reward * reward_scale / 3  # when reward_scale is not None",
        "```", "",
        f"Measured `reward_scale`: `{env_spec['reward_contract']['reward_scale']}`.",
        "The success branch replaces the shaped terms. A high-enough pot tilted",
        "more than 30 degrees can therefore satisfy task success while yielding zero",
        "reward. This distinction is covered by a real-simulator contract test.", "",
        "`done = (timestep >= horizon) and not ignore_done` is independent of success.",
        "The current configuration continues to horizon even if success happens earlier.",
        f"At {env_spec['control_freq']} Hz, {env_spec['horizon']} control steps represent",
        f"{env_spec['horizon'] / env_spec['control_freq']:g} seconds of simulated time.",
        "Step uses robosuite's four-value API: `(observations, reward, done, info)`.", "",
        "### Episode metrics", "",
        "| Metric | Definition |", "| --- | --- |",
    ]
    for name, meaning in env_spec["episode_metrics"].items():
        lines.append(f"| `{name}` | {meaning} |")
    lines += [
        "", "The rollout checks success at reset and after every step. `first_success_step`",
        "uses 0 for reset, 1-based control-step indices otherwise, and JSON null when",
        "success never occurs. Losing success later changes `final_success` without",
        "erasing `ever_success`. Rewards/returns never infer these flags.", "",
        "## Reproduce and verify", "",
        "Run from the REL301m research root after editable installation:", "",
        "```bash", "python -I scripts/inspect_env.py",
        "python -I scripts/random_rollout.py --episodes 3 --seed 0",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -m pytest -q tests", "```", "",
        "The inspector writes the three tracked JSON contracts under `artifacts/` and",
        "regenerates this Markdown. Tests compare live dimensions/dtypes/layouts with",
        "the saved contracts, verify sensor semantics and reward/success separation,",
        "and check full-horizon termination. No camera, rendering context, PyTorch,",
        "communication corruption, policy learning, or teammate prediction is required.", "",
        "Official references: [environment API](https://robosuite.ai/docs/source/robosuite.environments.html),",
        "[TwoArmLift API](https://robosuite.ai/docs/source/robosuite.environments.manipulation.html),",
        "[composite controllers](https://robosuite.ai/docs/modules/controllers.html).", "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
