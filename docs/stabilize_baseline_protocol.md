# Reproduce and instrument Phase 3 before increasing compute

This branch preserves the failing local implementation first, then adds instruments and isolated treatments. Original `masac.yaml`, `bc_pilot.yaml`, `bc_finetune.yaml`, environment configs, contract JSONs and historical run directories are unchanged. The existing BC-demo fine-tune is a separate method, not the original MASAC baseline.

## Reproduction

```bash
# Run from repository root with the existing demos and BC checkpoint.
.venv-phase3/bin/python -I -u -m rel301m.training.train \
  --config configs/experiment/stabilize_repro.yaml --steps 1000 \
  --run-dir experiments/phase3/stabilize_repro_1k

.venv-phase3/bin/python -I -u -m rel301m.training.train \
  --config configs/experiment/stabilize_repro.yaml \
  --run-dir experiments/phase3/stabilize_repro_3k

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-phase3/bin/python -I -m pytest -q tests
```

Output directories must be new. This reproduction uses the original failing fine-tune learner parameters, including 1,000 offline critic updates, 1,000 online critic-only updates, demo prefill, lambda_BC=0.5 and the original std schedule. Extra probes restore Torch RNG and reset only the dedicated evaluation environment's RNG. They do not sample the training replay to choose diagnostic subsets.

The unmodified loaded BC checkpoint is evaluated deterministically and stochastically before any optimizer update or optional std override. `loaded_bc_*_evaluation.json` records the raw checkpoint policy. `initial_*` files in the fine-tune or std treatment describe the configured policy after the std override; these are distinct distributions even when their deterministic means agree.

All full evaluations use 10 fixed seed-20,000 initial states with sequence SHA:
`19841a3401962f77201c881bd1c5865b19aa9a104e7c42c02926d6ccedfd7721`.
Early two-episode diagnostics use the first two states of the same sequence. Their prefix SHA is `a87d6cf9dc15e5472586ef8aa7fd16523cdd528bd9cc6295fb55cf557421fb10`. Do not pool two-episode success rates with ten-episode rates or select a best checkpoint using the early subset.

## Counters and diagnostics

`updates` is retained for compatibility: completed learner cycles, including offline pretraining. It is not a count of actor optimizer steps.

- `environment_steps`: completed training steps; evaluation steps are additional compute.
- `critic_optimizer_updates`: actual calls to the joint Q1/Q2 optimizer. One call updates both critics.
- `actor_optimizer_updates_0/1`: actual calls to each independent actor optimizer.
- `actor_optimizer_steps_total`: the sum of those two counts.
- `alpha_optimizer_updates`: actual calls to the joint two-temperature optimizer.

At environment step 1,000 in the reproduced fine-tune, Q count 2,000 means 1,000 offline plus 1,000 online Q updates. Both actor counters and alpha count remain zero. The counters are saved in checkpoints; legacy checkpoint counts are recovered from their Adam states. Exact replay/RNG resume is still unsupported.

`early_diagnostics.csv` records losses, Q/target/TD quantiles, gradients, alpha, entropy, action std/saturation and actor drift every 100 Q updates, including offline pretraining. Loss/TD metrics describe the pre-optimizer batch; drift describes the post-update policy. Drift and expert MSE use fixed 256-transition train/validation subsets recorded in `audit_subset.json`; test demos never enter these probes or gradients. Frozen actor loss fields are inherited zero placeholders, and optimizer counters distinguish that phase from actor optimization.

`policy_diagnostics.csv` separates deterministic and stochastic evaluations with episode counts, prefix hashes and optimizer counters. Stochastic probes run in `torch.random.fork_rng`; they do not change subsequent training randomness. `optimizer_events.jsonl` records actor hashes and alpha before/after freeze and immediately before the first actor update. A runtime assertion checks frozen parameters.

## SAC and BC audit

For the shared reward and independent stochastic actors:

```text
y = r + gamma * (1 - done * (1 - timeout)) *
    (min(target_Q1(s', joint_a'), target_Q2(s', joint_a'))
     - sum_i alpha_i * log_pi_i(a_i' | o_i'))

L_actor_i = mean(alpha_i * log_pi_i - min(Q1, Q2))
L_log_alpha_i = -mean(log_alpha_i * (log_pi_i.detach() + target_entropy_i))
```

With `bootstrap_time_limits=false`, use raw done instead. Target entropy is minus each live action dimension (currently -7). The temperature derivative is estimated entropy minus target entropy: gradient descent increases alpha below the target and decreases alpha above it. Differential entropy can be negative. These formulas follow [SAC](https://arxiv.org/abs/1812.05905); the joint entropy sum is the current cooperative learner's extension. The [SAC equations](https://spinningup.openai.com/en/latest/algorithms/sac.html) also motivate the minimum twin target and entropy subtraction. Tests cover signs, terminal/timeout masks, affine tanh scaling, centralized input, actor isolation and Polyak updates. No demonstrated formula bug was found, so these components remain unchanged.

BC supervises bounded tanh(mean), with no direct std target. Output-head std rows have zero BC gradient and remain unchanged by a fresh Adam step; shared hidden features can still change std outputs indirectly. Tests distinguish this from accidental std supervision.

The original actor loop is sequential: actor 1's loss sees the newly stepped actor 0. The earlier intent is undocumented. This behavior is preserved and locked by a regression test; it has not been established as the cause of collapse. A simultaneous snapshot update would be a separate learner experiment.

## Isolated treatments

The short-control copies nominal MASAC settings with one explicitly documented common deviation: random warmup=0, so a 2k run reaches actor updates. It has no demo prefill, no BC loss, no offline critic pretraining and no fixed-std window. Train/validation demos are loaded only for observation diagnostics. Replay begins empty; with batch=256, the first Q update occurs at environment step 256.

| Experiment | Change relative to short-control |
| --- | --- |
| `stabilize_control.yaml` | None |
| `stabilize_a_critic_only.yaml` | `critic_only_updates: 1000` |
| `stabilize_b_initial_std.yaml` | `bc_init_log_std: -3` only; then learnable |
| `stabilize_c_actor_lr.yaml` | Actor lr 3e-5, critic lr remains 3e-4 |

A updates Q/targets normally while actor/alpha optimizer calls remain disabled; actor hashes are checked before/after freeze. B changes only std head rows after the raw-BC probes, preserves deterministic actions and records `log_std_override` in actor initialization metadata. B is disabled by default. C changes one learning rate. All other configs, source, seed, checkpoint and evaluation states match their short-control. Their algorithm-family metadata remains BC-initialized MASAC; the experiment names and resolved configs identify the individual treatment.

```bash
for variant in control a_critic_only b_initial_std c_actor_lr; do
  .venv-phase3/bin/python -I -u -m rel301m.training.train \
    --config "configs/experiment/stabilize_${variant}.yaml" \
    --run-dir "experiments/phase3/stabilize_${variant}_2k" || break
done
```

These are independent 2k seed-0 diagnostics, not continuation of the historical run. Audit configs are capped at 5k. Known optional Panda notices are filtered; errors remain visible. Use new run directories when repeating commands.

The previously implemented demo-prefill/BC-loss method remains separate (see [fine-tune protocol](bc_finetune.md)); its bundles are not single-factor ablations of this control. Account for its offline Q updates and demonstration collection/BC compute separately.

## Limits

A local reproduction can localize collapse without proving a universal root cause or improvement. Pre-update stochastic differences do not alone explain later deterministic drift. No test demos are used for training/model selection, and no architecture or task success semantics change.

The native MuJoCo box-box collision failure remains reproducible and unresolved. Short collision-free runs do not make a long pilot safe. Do not run 300k × 3 seeds or merge this branch automatically; review the [audit report](stabilize_baseline_report.md) and simulator blocker first.
