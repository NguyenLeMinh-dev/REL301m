# BC bootstrap for assisted Panda Lift

An independent `successful assisted-SAC rollouts → dataset → BC → real rollout` baseline. It imports no `rel301m` modules and changes no existing SAC baseline, native reward, success or physics. It does **not** run SAC fine-tuning.

Live contract: Lift/Panda, BASIC + OSC_POSE, `object-state` then `robot0_proprio-state` (**60 float32**), **7 float32 actions** bounded by [-1,1], 20 Hz, horizon 500, shaped reward scale 1, no camera/rendering during collection or training. Gripper assistance forces OPEN (-1) only when pre-step gripper-to-cube distance > 5 cm. Training/collection retain this assistance. Evaluation supports `--mode assisted` (default), `unassisted`, or `both`; unassisted removes only the gripper gate. Modes are reported separately, without pooled success or an unassisted claim based on assisted results.

`assisted_env.py` is a verbatim snapshot of the current assisted source; its hash/path are in [environment_source.json](environment_source.json). In this checkout that source/model lives in `robosuite_official_sac_lift/`, although the requested name was `robosuite_sac_lift_stage1_assisted/`. No source baseline was edited. A fresh clone needs the locally trained source `best_model.zip` **and its sibling config.json**; neither is committed here.

## Run from this folder

```bash
cd /home/minh/Documents/REL301m/REL301m-research/baselines/bc_lift_bootstrap
export PYTHONNOUSERSITE=1
unset PYTHONPATH  # exclude ROS when calling the dataset inspector directly
./install.sh
./smoke.sh
```

All shell wrappers locate `.venv-phase3/bin/python` from their own directory and unset PYTHONPATH. `install.sh` only verifies existing dependencies, including Torch 2.7.1, robosuite 1.5.2 and MuJoCo 3.9.0; it installs/upgrades nothing. Optional robosuite notices are quiet; exceptions/errors propagate.

For a future separately reviewed experiment, collect distinct sustained-success demos from the existing teacher. The completed acceptance uses only **2 demos / 2 BC epochs**; no 50-demo experiment is started by this patch. New output paths are required; files are never overwritten. Progress shows successes, attempts, success rate and accepted transitions.

```bash
./collect.sh \
  --model ../robosuite_official_sac_lift/runs/stage1_assisted_seed0_100k/best_model.zip \
  --successful-episodes 50 --success-hold-steps 10 --max-attempts 1000 --seed 10000 \
  --output data/lift_sustained_50_hold10.npz

../../.venv-phase3/bin/python inspect_dataset.py data/lift_sustained_50_hold10.npz

./train_bc.sh --dataset data/lift_sustained_50_hold10.npz \
  --epochs 100 --batch-size 256 --seed 0 \
  --run-dir runs/bc_lift_50demo_seed0

./eval_bc.sh --model runs/bc_lift_50demo_seed0/checkpoints/best_bc.zip \
  --episodes 20 --seed 20000 --mode both

./watch.sh runs/bc_lift_50demo_seed0/checkpoints/best_bc.zip --episodes 3
```

The collector also accepts `--model runs/stage1_assisted_seed0_100k/best_model.zip` and resolves that exact run in either assisted source folder. If the attempt budget is exhausted it saves accepted episodes, prints `COLLECTION_INCOMPLETE`, and exits **2**. Zero sustained successes produce an empty NPZ explicitly rejected by inspection/training. Episodes without the required consecutive success are rejected, including transient-only successes. Hand-coded actions, relabelled failures and duplicated episodes are never substituted. Review incomplete collection before choosing to train on its smaller dataset; training needs at least two successes.

## Labels, split and checkpoints

The recorder sits **below** the copied assistance wrapper. `actions == executed_actions` is the full argument actually sent to robosuite; `policy_actions` retains the ungated SAC output. Raw observations and labels correspond to the same timestep, and next observations are recorded after step. The simulator rollout still runs to horizon 500 for verification. Extraction finds the **first** completed native-success interval of `--success-hold-steps` consecutive post-step states (default **10**, 0.5 simulated seconds at 20 Hz). It saves the prefix through the end of that interval and excludes all later dropping/wandering. Earlier transients may precede the first sustained interval; metadata distinguishes `first_success_step` from `sustained_success_start`. Step numbers in those metadata fields are **1-based**; transition `steps` are **0-based**. Reset success, if any, is separately recorded as first success 0 and does not count as an executed hold step.

Schema **2** supports variable lengths. `demo_end` is true only on the last saved transition. `terminated`, `truncated` and `dones` are copied unchanged: an early data cut has **all false**, whereas a prefix ending at the actual horizon retains a true timeout. BC ignores these flags in its MSE loss. No MDP terminal is invented. Legacy schema-1 full-episode files must be recollected rather than silently reused.

`ever_success` repeats final demo eligibility; `current_success` records post-step native success. Distance/assist flags are pre-step; grasp/cube height are post-step. Distance is float64 to preserve threshold decisions; observations/actions/rewards are float32. Metadata records each demo's first success, first eligible sustained start, hold length, saved transition count, full verification length, truthful end flags and final next-observation hash. `rejected_transient_success` counts rejected episodes that ever succeeded but had no eligible hold interval. The source-policy collection success rate is explicitly **sustained-success acceptance**, with an additional ever-success rate.

NPZ and companion JSON contain versions, checkpoint hash/config, seed, attempt results, accepted episode seeds and counts. Inspection checks finite values, dtypes/shapes/bounds, executed-versus-policy gating, unique episode identities, variable-length contiguous prefixes, no next-observation leakage (including final-state hash), the first eligible sustained-success ending and truthful environment/demo boundaries.

The deterministic seed-based split is **80% episodes / 20% episodes**, never transitions. It retains at least one episode in each partition (the two-demo smoke therefore uses 1/1). BC trains a newly instantiated native SB3 SAC `MlpPolicy`, `[256,256]`, with Adam 3e-4 and plain `MSE(actor(obs, deterministic=True), executed_action)`. No extra loss/normalization is introduced. CSV reports total/arm/gripper MSE. Defaults: 100 epochs, batch 256, validation every epoch, patience 20; `--patience 0` disables early stopping. Validation minimum selects best; no evaluation success is used in selection.

Outputs under the new run directory:

- `checkpoints/best_bc.zip`, `last_bc.zip`, `bc_sac_warmstart.zip` (byte copy of best): complete native **SB3 SAC** checkpoints, loadable via `SAC.load(path, env=env)` without layer mapping.
- `checkpoints/bc_actor_state_dict.pt`: best actor weights for audit.
- `history.csv`, `summary.json`: source/dataset hashes, exact episode split, hyperparameters, versions, Git SHA, losses and checkpoint hash.
- `evaluation.json` (assisted), `evaluation_unassisted.json`, or `evaluation_both.json`: fixed new seeds, per-episode return/ever-success/final-success/first-success/length, aggregate return and success, grasp and raw-policy premature-close rate. Failed first-success values are null; its mean uses successful episodes only. Eval rejects demonstration seed overlap and updates summary with metrics keyed by mode. `both` verifies paired initial-state hashes and never averages modes. Unassisted forwards every policy action unchanged and reports zero gripper overrides.

Critics and temperature remain random/untrained. The std head has no direct deterministic BC supervision; shared-feature changes can still change stochastic outputs. **Deterministic BC does not calibrate stochastic SAC exploration** or match the teacher's stochastic policy. No log_std schedule/initialization is changed here. Save/reload regression records deterministic-action preservation on the same device; it does not assert stochastic equivalence to the teacher. A compatible `bc_sac_warmstart.zip` **does not establish safe BC→SAC fine-tuning**.

`watch.sh` opens a native MuJoCo passive viewer and prints distance, policy/executed gripper, grasp, cube height and success. Run it from a graphical desktop. `--headless` exercises diagnostics without opening a window; headless mode was tested in the earlier baseline acceptance and was not re-run in this follow-up. The interactive GUI has not been validated.

## Completed acceptance

See [VERIFICATION.md](VERIFICATION.md) and [acceptance_report.json](acceptance_report.json) for the **post-commit** acceptance on `baseline/bc-lift-bootstrap-clean`. Code is committed before smoke/regression/2 sustained demos/inspection/2-epoch BC/dual evaluation run. The report records that exact tested SHA; any later evidence-only commit is distinguished from it. No large training or SAC fine-tuning is included.

Regression command from repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1 \
  .venv-phase3/bin/python -I -m pytest -q baselines/bc_lift_bootstrap/test_contract.py
```

Datasets, runs, checkpoints and videos are ignored. Keep them locally; Git tracks baseline code and small acceptance evidence only.


## Clean branch provenance

`baseline/bc-lift-bootstrap-clean` starts directly at current main, cherry-picks only the original BC commit, and applies these BC-only fixes. Verify with `git diff --name-only main...baseline/bc-lift-bootstrap-clean`: every path must be under `baselines/bc_lift_bootstrap/`. No audit/MASAC/core/config changes or merge are carried into this branch. Existing local SAC baseline folders remain untracked and untouched.
