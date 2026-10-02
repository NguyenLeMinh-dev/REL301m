# BC-init seed 0 — learning diagnosis

Measured 2026-10-02. Run: `experiments/phase3/bc_pilot_20261002T094730Z/seed0`.
The run completed 138,757 env steps / 128,758 optimizer updates, then failed on
attempted step 138,758. This report diagnoses saved behavior and short probes;
it does not report a completed 300k pilot or change baseline hyperparameters.

## What the saved trajectory shows

| RL steps | Ever SR / 10 episodes | Final SR / 10 | Validation expert-action MSE, agent 0 / 1 |
| --- | ---: | ---: | --- |
| BC only, before RL | 70% | 20% | 0.000583 / 0.000953 |
| 10k (only 1 SAC update) | 10% | 10% | 0.001248 / 0.001909 |
| 20k | 0% | 0% | 0.225378 / 0.159492 |
| 50k | 0% | 0% | 0.781343 / 0.788892 |
| 100k | 20% | 0% | 0.695228 / 0.568969 |
| 130k | 0% | 0% | 0.615642 / 0.676185 |

SR rows use the same ten monitoring initializations, seed 20000/CUDA. Validation
MSE was recomputed on the same five BC validation episodes, on CPU; final test
demos were not used. The initial MSE numbers are validation, not the previously
reported held-out test MSE. MSE is an expert-state diagnostic rather than the SAC
objective. The large drift together with poor closed-loop success is evidence
consistent with losing the demonstrated behavior; it does not identify a single
causal mechanism or mean every action differing from expert is necessarily wrong.

Mean shaped return grows from 16.27 at 100k to 32.17 at 130k while success falls
from 20% to 0%. Partial reward terms may improve without a successful lift; reward
components were not logged, so no particular term is claimed to be responsible.
Increasing training steps alone is not justified by this trajectory.

Figure: `experiments/phase3/bc_pilot_20261002T094730Z/diagnosis/learning_diagnosis.{png,pdf}`.
Raw measurements: `diagnosis/validation_drift.json`; original eval/loss CSVs unchanged.

## Direct probe of BC stochastic behavior

Two policies used identical BC mean weights, five full 200-step episodes, common
seed 20000/CUDA, and identical initialization-sequence SHA-256:
`3205c20f906548777c9ab6141cc69f2e1e045bdea0a61167155b0d86f2706c19`.
Stochastic probes reset Torch seed 42. These are five monitoring initializations,
not a new independent population estimate or a training experiment.

| Inference mode / initial distribution | Ever success | Final success | Mean return |
| --- | ---: | ---: | ---: |
| Original BC deterministic `tanh(mean)` | 5/5 | 2/5 | 37.1392 |
| Original BC Gaussian sampling | 0/5 | 0/5 | 5.1827 |
| Same mean, constant log_std=-2 (std≈0.1353) | 1/5 | 0/5 | 12.2547 |
| Same mean, constant log_std=-3 (std≈0.0498) | 4/5 | 2/5 | 30.6497 |

Only the std-output rows of the last Linear layer were changed in copied models
in memory; no BC or RL checkpoint was overwritten. This shows the initial
exploration distribution matters in this probe. The current BC loss supervises
`tanh(mean)` only. The log-std head has no supervised target, although BC training
changes the shared encoder. Therefore deterministic BC success does not imply
that the stochastic policy used by SAC has learned successful behavior.

Small std initialization is a candidate ablation, not a proven long-run fix.
SAC still updates log-std/alpha later. BC itself only achieved 20% final success
on the previous ten-episode reference, so retaining its behavior is not the same
as fully solving stable lift.

## Training transition likely needs attention

The frozen comparison uses an empty replay, 10k random-action warmup and fresh Q
weights. The first replay update sees entropy -91.36 / -105.08, actor-0 action
saturation 62.3% and mean Bellman target -39.30. On demonstration validation states,
BC has std about 1.23 / 1.12, but substantially different entropy. These metrics
are measured on different state distributions and are not interchangeable.

By 20k, actor behavior on expert states has drifted strongly. There is no demo
replay sampling or BC auxiliary loss to retain expert behavior in the current
actor-only initialization method. Distribution shift, uncalibrated stochastic
outputs and early actor updates against fresh critics are reasonable candidate
mechanisms; their individual causal contributions require controlled ablations.

At 100k, alpha is about 0.00231 / 0.00207 and replay entropy -6.91 / -7.32. At
138.7k, alpha is 0.00160 / 0.00147 and entropy -7.38 / -7.12. The late entropy
is near the target -7, so these observations alone do not establish a broken
automatic-alpha implementation. Do not arbitrarily fix alpha or change reward
to explain away failure. Normalization folding preserves initial policy
inference, but changes optimization parameterization; it is a separate concern,
not a demonstrated inference bug in this audit.

## Recommended next experiment

Keep this interrupted run as the original BC-init baseline. Do not silently
rename it or change the frozen scratch/BC-only-initialization protocol.

1. Test explicit initial log_std=-3 in a separately named configuration and
   verify pre-update deterministic/stochastic SR plus early validation MSE.
   This is an engineering choice supported by the small inference probe.
2. Test a **demonstration-assisted MASAC** variant: retain train demos for
   demonstration replay and add a BC auxiliary actor loss,
   `L_actor = L_SAC + lambda_BC * MSE(tanh(mean), expert_action)` on own-agent
   train-demo observations. No teammate-private inputs and no validation/test
   demos in gradients. Compare against the pure initialization baseline; it is
   a new algorithmic treatment, not the same MASAC experiment.
3. Examine the warmup/critic transition separately: initial behavior data near
   successful BC trajectories and critic fitting before releasing actor updates
   are candidate ablations. Do not bundle actor-lr, alpha, reward and replay
   changes together and then claim one of them caused an improvement.
4. First use a short diagnostic budget with evaluations before updates and soon
   after actor updates. Check whether SR/MSE remains near BC, not only finite
   losses. Only move to longer learning runs once the early degradation is addressed.

Combining demonstration supervision with RL has precedent; for example
[Cycle-of-Learning](https://arxiv.org/abs/1910.04281) studies BC+RL losses and the
transition from demonstrations to actor-critic learning. That paper does not
prove this particular two-agent SAC variant improves TwoArmLift. No lambda,
sampling ratio or learning-rate change has been selected by a long RL result here.

## Simulator failure is a separate blocker

FatalError: `mj_narrowphase: collision function returned 10 contacts for geom pair
(150, 153), expected at most 8 from mj_maxContact`. A fresh contract-matching model
maps this pair to `gripper1_right_finger1_pad_collision` and
`gripper1_right_finger2_pad_collision`, both box geoms. This is a MuJoCo collision
error, not a warning to suppress. Saved numeric counters show no NaN/Inf/action-bound
violations, but they do not establish the exact cause of the collision failure.
Increasing a generic contact-memory limit or changing packages has not been
validated as a repair. Reproduce/resolve this before another long pilot; the
existing qpos/qvel snapshot is not a complete exact-resume simulator state.

## Audit scope

Read-only checkpoint/CSV analysis and 20 short inference episodes (4,000 env steps),
no RL optimizer updates and no new training run. Source code, pinned packages,
original configs and checkpoints remain unchanged. Measurement/probe scripts
and JSON/CSV/figure artifacts are retained under `diagnosis/` for review.
