# Post-commit clean-branch acceptance

**BC_BRANCH_CLEAN=YES; BC_PIPELINE_READY=YES.** Measured 2026-10-03T00:45:16.524882+07:00.

Every command below executed at committed implementation SHA **`48b63afc97502c50e19a38441c877ffd9e4d1b4a`** on `baseline/bc-lift-bootstrap-clean`, with no uncommitted BC changes. Smoke printed that SHA; dataset, training summary and dual evaluation independently record it. This report is published in a later **evidence-only commit**; it does not claim its own publishing commit was the run SHA.

## Clean provenance and preservation

Base main: `6bc872c13830a7ee1e105f6fd717a63837c66eee`. Clean branch cherry-picks only original BC commit then applies BC-only review fixes. `git diff --name-only main...HEAD` contains **26 paths**, all under `baselines/bc_lift_bootstrap/`. No audit branch ancestry, core/MASAC/training/config/evidence changes outside this folder, or merge. All **33 existing SAC code/input files** were SHA-256 checked unchanged. The native assisted environment snapshot remains identical to its source. Existing local SAC folders remain untracked.

## Sustained demonstrations

Teacher deterministic rollouts verified all 500 environment steps; training examples end at the **first completed 10-step native-success interval**. Accepted **2/11 attempts**; saved **68 transitions**, excluding all later dropping/wandering. Real rejected-transient count in these attempts: **0**; regressions independently exercise transient rejection and reset the consecutive-success counter.

| Seed | First success (1-based) | Sustained start (1-based) | Hold steps | Saved transitions |
| ---: | ---: | ---: | ---: | ---: |
| 10005 | 27 | 27 | 10 | 36 |
| 10010 | 23 | 23 | 10 | 32 |

Dataset inspection PASS. `demo_end` count **2**, while saved `dones`, `terminated`, `truncated` counts are all **0**. Each final saved next observation is hash-linked to the state captured when the native hold interval completed. Actions equal recorded executed actions, captured below assistance; policy actions are retained separately. Observations **60 float32**, actions **7 float32** in [-1,1]. Variable lengths/contiguous steps/no episode leakage/first eligible hold/real timeout preservation are covered by tests. Legacy schema-1 datasets are not silently reused.

## BC and separate evaluation

Unchanged deterministic MSE and Adam: 2 epochs, 1 train episode / 36 transitions, 1 validation episode / 32 transitions. Best epoch **2**, validation MSE **0.328362737**. Native SAC best/last/warm-start zips and actor state_dict saved locally. Deterministic same-device reload maximum action error **0**; critics unchanged. No log_std initialization/schedule or additional loss was introduced. **Deterministic BC does not calibrate stochastic SAC exploration**, and load compatibility does not establish safe BC→SAC fine-tuning or stochastic equivalence to the teacher.

Both modes use seeds **20000/20001**, with identical initial-state hashes, 500-step native episodes, same task/controller/reward/observations/actions. Unassisted removes only the assistance wrapper and forwards policy actions unchanged.

| Mode | Mean return | Ever-success | Final-success | Gripper overrides |
| --- | ---: | --- | --- | ---: |
| Assisted | 1.057863 | 0/2 | 0/2 | 1000 |
| Unassisted | 1.044202 | 0/2 | 0/2 | 0 |

**Smoke PASS; 29 regression tests PASS in 3.25s.** Two demos/two epochs/two paired eval states establish pipeline operation, not improved learning/performance. No 50-demo experiment or SAC fine-tuning ran. GUI watch was not re-tested in this follow-up.

## Exact commands executed

CWD: `baselines/bc_lift_bootstrap/`. Environment: `PYTHONNOUSERSITE=1`, PYTHONPATH unset, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Wrappers locate the existing `.venv-phase3/bin/python`; no dependencies were installed/upgraded.

```bash
./install.sh
./smoke.sh
/home/minh/Documents/REL301m/REL301m-research/.venv-phase3/bin/python -I -m pytest -q test_contract.py --junitxml /home/minh/Documents/REL301m/REL301m-research/baselines/bc_lift_bootstrap/runs/clean_acceptance_48b63afc/logs/regression.xml
./collect.sh --model ../robosuite_official_sac_lift/runs/stage1_assisted_seed0_100k/best_model.zip --successful-episodes 2 --success-hold-steps 10 --max-attempts 50 --seed 10000 --device cpu --output /home/minh/Documents/REL301m/REL301m-research/baselines/bc_lift_bootstrap/data/clean_sustained_48b63afc.npz
/home/minh/Documents/REL301m/REL301m-research/.venv-phase3/bin/python inspect_dataset.py /home/minh/Documents/REL301m/REL301m-research/baselines/bc_lift_bootstrap/data/clean_sustained_48b63afc.npz
./train_bc.sh --dataset /home/minh/Documents/REL301m/REL301m-research/baselines/bc_lift_bootstrap/data/clean_sustained_48b63afc.npz --epochs 2 --batch-size 256 --seed 0 --device cpu --run-dir /home/minh/Documents/REL301m/REL301m-research/baselines/bc_lift_bootstrap/runs/clean_acceptance_48b63afc/bc
./eval_bc.sh --model /home/minh/Documents/REL301m/REL301m-research/baselines/bc_lift_bootstrap/runs/clean_acceptance_48b63afc/bc/checkpoints/best_bc.zip --episodes 2 --seed 20000 --mode both --device cpu
```

Local ignored run/logs: `baselines/bc_lift_bootstrap/runs/clean_acceptance_48b63afc`. Dataset/model/hash/versions/per-episode native success/metadata/command exits/source hashes are in [acceptance_report.json](acceptance_report.json). Datasets/checkpoints/runs stay out of Git. Re-run commands with new output paths to avoid overwriting evidence.
