# Short BC-assisted MASAC fine-tune

This is a separate experimental variant; `masac.yaml`, `bc_pilot.yaml`, the simulator pins and Phase 1/2 contracts remain unchanged. It combines demonstrations with SAC updates to test whether BC behavior survives online learning. Improved success is a hypothesis, not an established result for this task.

## Correct interpretation of the earlier evidence

BC deterministic evaluation: 7/10 ever-success and 2/10 final-success. A separate five-episode noise probe with `log_std=-3` achieved 4/5 ever-success and 2/5 final-success; it did not establish 80% final-success. The saved 20k/100k checkpoints show drift, but no measurement establishes degradation after exactly one SAC update. Forty accepted demonstrations already exist; requiring 100 is not supported by the present evidence.

The [previous diagnosis](bc_pilot_seed0_diagnosis.md) contains the measured BC/stochastic and held-out MSE results. Alpha decreasing under automatic tuning alone is not evidence of insufficient exploration or a bug.

## Architecture and data

Live contract: actors 66 → 7 independently; critic state 119; joint action 14. No actor receives the teammate's private observation. Shared reward, native success and horizon/time-limit bootstrap semantics are retained.

`DemonstrationReplay` loads the completed NPZ manifest, verifies hashes, dimensions, live bounds, environment config and the exact episode split stored in the BC checkpoint. Only the 30 training episodes (6,000 transitions) enter prefill, retained replay or BC auxiliary loss. Five validation episodes measure drift; five test episodes are excluded from all losses and reporting here. Normalization is already folded into the exported BC actor, so raw 66-dimensional observations remain valid.

The mutable replay is prefilled with train demos and then receives online transitions. A separate immutable demo replay survives mutable-ring overwrites. `demo_ratio` is the fraction drawn explicitly from retained demos; the mutable replay may still contain prefilled demos, so this is **not an exact expert/online ratio** early in training. The BC loss always uses a separate train-demo batch.

For each actor:

```text
L_actor_i = mean(alpha_i * log_pi_i - min(Q1, Q2))
            + lambda_bc * mean((bounded_tanh_mean_i - expert_action_i)^2)
```

Each BC term sees only its own actor observation/action. Teammate samples have no gradient path; critics are frozen during actor updates. MSE is in physical action units (current bounds [-1, 1]).

## Default configuration

Use `configs/experiment/bc_finetune.yaml` and `configs/algo/masac_bc_finetune.yaml`:

| Setting | Value |
| --- | ---: |
| Environment steps | 10,000 |
| Random warmup | 0 |
| Initial log_std | -3 |
| Initial alpha / mode | 0.02 / auto |
| BC coefficient | 0.5 |
| Critic pretraining | 1,000 gradient updates, lr 1e-4 |
| Actor freeze after pretraining | 1,000 online gradient updates |
| Fixed log_std window | First 3,000 total critic updates, including pretraining |
| Retained-demo sample fraction | 0.75 → 0.5 over 5,000 environment steps |
| Periodic evaluation/checkpoint | Every 1,000 environment steps |
| Evaluation episodes / seed | 10 / 20,000 |

Initialization zeroes only the log_std head rows and sets their bias. Deterministic actor means remain identical to BC. During the fixed-std window, `forward()` returns constant log_std; after the window it uses the learnable head. The schedule is derived from checkpoint `updates`, including offline critic updates.

Critic-only updates step Q/target Q; they do not step actor or alpha optimizers, including their Adam momentum. Critic pretraining uses SAC Bellman targets from the frozen policy; it does not guarantee accurate Q or require an arbitrary TD-MSE threshold. The pretraining critic learning rate is restored before online training. Alpha remains fixed throughout actor freeze, then resumes automatic tuning. `alpha_mode: fixed` is available as a separate control.

`updates` includes offline pretraining; `actor_updates` counts online updates that actually step the actors. `freeze_log_std_updates` and `actor_freeze_updates` use these different counters deliberately. Changing `--steps` does not rescale either schedule.

## Run commands

Run from the repository root after installing `.venv-phase3` and generating the demos/BC checkpoint via [BC workflow](bc_warm_start.md). Default paths use the existing dataset and `experiments/phase3/bc_seed42/best.pt`.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-phase3/bin/python -I -m pytest -q tests

# First diagnostic experiment: 10k, one seed; new output folder required.
.venv-phase3/bin/python -I -u -m rel301m.training.train \
  --config configs/experiment/bc_finetune.yaml --seed 0 --steps 10000 \
  --run-dir experiments/phase3/bc_finetune_10k_seed0
```

The CLI shows progress, ETA, evaluation and critic-pretrain stages. Known optional Panda/BASIC notices are filtered; unknown warnings/errors remain visible. Add `--show-all-warnings` to restore verbosity. Logs survive piping through `tee`; use `set -o pipefail` if checking process exit through a pipe.

Only after reviewing the short run and deciding another diagnostic is justified:

```bash
.venv-phase3/bin/python -I -u -m rel301m.training.train \
  --config configs/experiment/bc_finetune.yaml --seed 0 --steps 30000 \
  --run-dir experiments/phase3/bc_finetune_30k_seed0
```

This starts a fresh experiment from BC; it is not exact continuation of the 10k run. Run independent seed 1/2 with new output folders once the one-seed diagnostics justify repetition. No 300k pilot is authorized by this protocol; fine-tune config enforces a maximum 30k environment-step budget.

## Outputs and acceptance

- `initial_bc_evaluation.json`, `initial_bc_episodes.csv`: paired deterministic evaluation before any optimizer updates.
- `initial_stochastic_bc_evaluation.json`, episode CSV: the same initial states with stochastic actions; training Torch RNG is restored after the probe.
- `initial_bc.pt`, `best.pt`: step-0 BC is included in best-checkpoint selection, so worse RL checkpoints do not overwrite it.
- `critic_pretrain.csv`, `post_critic_pretrain.pt`: offline Q updates and checkpoint.
- `losses.csv`: SAC/Q/alpha losses, BC MSE, actor-updated flag, entropy/log_pi, action saturation/std, TD targets/errors and sampling fraction.
- `validation.csv`: raw bounded-action MSE per actor at step 0 and each evaluation, with critic/actor update counts.
- `evaluation.csv`, episode CSVs: ever/final success, return, first success and paired initialization hashes.
- Existing config/environment snapshots, Git SHA/status, source hashes, pip freeze, TensorBoard, checkpoints, summary and diagnostics.

Code PASS requires regression tests, finite/bounded transitions and exercised Q/actor/alpha/BC updates. Learning PASS requires paired SR and drift evidence across independent seeds; a finite 1k run does not establish it. Inspect `initial_bc_evaluation.json` against later evaluations before increasing compute. Keep ever-success and final-success separate. `reward_components.csv` and TensorBoard record reaching/grasping/lifting/success terms at logging steps and episode boundaries. The diagnostic decomposition follows pinned robosuite 1.5.2, applies reward scaling and checks its sum against the actual upstream reward. Native success bypasses reaching/grasping/lift terms; tilt gates lift/success. The reward stored in replay remains unchanged.

Ablation controls use copied configs: log_std only (`lambda_bc: 0`, no prefill, retained ratios 0, critic pretraining/freeze 0; keep initial alpha and other parameters matched to its control), prefill-only (`lambda_bc: 0`), BC-only (`prefill_replay: false`, retained ratios 0), combined, with/without critic pretraining/freeze, and fixed alpha. Keep other settings, checkpoint, train split, eval seed and learning budget matched. Changing several parameters together cannot identify which mechanism caused an improvement.

## Collision blocker

The earlier failure returned 10 contacts for a box-box pair where MuJoCo 3.9.0 expects at most 8. In the [versioned collision source](https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_collision_driver.c), `mj_maxContact` returns 8 for box-box and narrowphase checks that count. This limit is not a user XML `mj_maxContact=16` setting; increasing generic contact memory does not repair this invariant failure.

The saved failure pose reproduces this exact error with `mj_forward` on the unchanged pinned model. Reproduce locally:

```bash
.venv-phase3/bin/python -I scripts/reproduce_collision.py \
  --run-dir experiments/phase3/bc_pilot_20261002T094730Z/seed0 \
  --output experiments/phase3/collision_reproduction.json
```

The diagnostic returns a JSON status (`reproduced` or `not_reproduced`); it does not train or claim exact resume.

No dependency, collision mask or task physics is changed here. `collision_fix_verified: false` remains in run metadata/diagnostics. A short collision-free run does not resolve the historical failure; do not launch a long pilot until a reproducible repair is verified against the simulator contract. Errors are never filtered or silently resumed.

## Literature context

[Overcoming Exploration in Reinforcement Learning with Demonstrations](https://arxiv.org/abs/1709.10089) and [Cycle-of-Learning](https://arxiv.org/abs/1910.04281) motivate combining retained demos with policy supervision. Their results do not prove that these coefficients or this multi-agent SAC implementation improve TwoArmLift.
