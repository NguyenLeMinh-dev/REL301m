# BC Lift bootstrap acceptance

Final acceptance completed 2026-10-03 (Asia/Ho_Chi_Minh); initial checks began 2026-10-02. **BC_PIPELINE_READY=YES; SMOKE=PASS.** This confirms implementation and serialization, not a learning benefit.

- Live observations: **60 float32**; actions: **7 float32**, [-1,1]; BASIC/OSC_POSE; native reward/success and horizon-500 truncation.
- Existing stack verified, no installation/upgrades: Python Phase-3 env, robosuite 1.5.2, MuJoCo 3.9.0, Torch 2.7.1+cu128, SB3 2.7.1, NumPy 1.26.4. Torch CUDA availability: true.
- Real assisted-SAC teacher: **2 successful episodes / 11 attempts**, **1,000 transitions**, seeds **10005 and 10010**. No failed trajectory was relabelled; accepted episodes run the entire 500 steps. Source model/config hashes and all attempt outcomes remain in local metadata.
- Dataset checks PASS: finite shapes/dtypes/bounds, action labels after assistance, same-timestep observation/action storage, unique full episode boundaries, and next-observation continuity.
- BC: **2 epochs**, 1 train + 1 validation episode, 500 transitions each; best epoch 1, validation MSE **0.391715541295**. Best/last/native SAC warm-start zips and best actor state_dict saved locally; deterministic checkpoint reload/predict PASS. Critics unchanged.
- Fresh deterministic evaluation: seeds **20000/20001**, mean return **0.646338**, ever-success **0/2**, final-success **0/2**, episode length **500**. A two-episode, two-epoch smoke does not demonstrate improved success.
- Incomplete path: requested 2 successes, max 2 attempts at seed 10005; obtained 1, saved its 500 transitions, printed COLLECTION_INCOMPLETE and returned **exit 2**.
- **14 regression tests passed in 2.66s**: executed-label gating, corruption/NaN/bounds/dtype/boundary rejection, episode split, source-vs-recorder live behavior equality, untouched critics/std head and exact same-device actor save/load, JSON contract roundtrip.
- Watch diagnostics: one full headless episode PASS. Interactive native MuJoCo viewer code is provided; the graphical window was not tested here.
- **235 pre-existing files** were SHA-256 checked unchanged, including core, existing SAC baseline code and source checkpoints/configs. No SAC fine-tuning ran.

Exact executed commands, checkpoint/dataset hashes, tested source-file hashes, original environment path and full evaluation are in [acceptance_report.json](acceptance_report.json). Git HEAD during validation was the parent audit commit; code was untracked until the new baseline commit. The code hashes identify the tested worktree rather than mislabelling it as that parent revision.

The original stage-1 folder name differs from the requested name: it is currently `robosuite_official_sac_lift`. Its environment was copied verbatim; baseline source files remain untouched. Local data/models/runs are ignored and are not part of the commit. Re-execute the commands with **new** output paths; the pipeline refuses overwrites.
