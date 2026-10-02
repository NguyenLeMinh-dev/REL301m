# BC warm-start — measured report

Date: 2026-10-02. Scope đã thực hiện: thu 20–50 successful demos, chạy BC và kiểm
thử ngắn. **Không chạy RL pilot 300k × 3 seeds.** Pipeline engineering PASS không
đồng nghĩa nominal learning PASS. Lệnh tự chạy: [BC warm-start guide](bc_warm_start.md).

## Demonstrations

Dataset: `data/demonstrations/two_arm_lift_scripted/`.

- 40 accepted / 63 attempted full episodes; 200 steps mỗi episode.
- 8,000 retained transitions; 12,600 actual collection steps, gồm rejected attempts.
- Tất cả 40 accepted demos có final native success và cả hai native grasps tại
  frame cuối. Demo acceptance yêu cầu hai grasps trong success, tách biệt predicate
  native chỉ kiểm tra pot height; reward/done/horizon giữ nguyên.
- Measured actor observations 66 + 66; critic state 119, Q input 133;
  actions 7 + 7 = 14 với live bounds [-1, 1]. Dataset raw observations không chứa
  thêm privileged teacher features ở actor input.
- Manifest SHA-256:
  `593bd2dddad2b168eb4498263b95562b1ce3059d25c769f848ba234247e34b21`.
- Preview `preview_attempt_000.mp4`: H.264, 640×576, 20 FPS, 200 decoded frames,
  duration 10.000 s. Decode toàn bộ không lỗi; đã inspect frame cuối đoạn lift.

## Behavior cloning

Run: `experiments/phase3/bc_seed42/`, CUDA. Episode split seed 42 là 30 train /
5 validation / 5 test; train-only normalization. Mỗi test split gồm 1,000 correlated
transitions từ 5 independent episodes, không phải 1,000 independent task evaluations.
300 epochs, batch 256, lr 3e-4, 7,200 optimizer updates mỗi actor.

| Metric | Measured |
| --- | ---: |
| Selected epoch (validation only) | 134 |
| Mean validation action MSE | 0.0007680041 |
| Agent 0 held-out action MSE | 0.0009466055 |
| Agent 1 held-out action MSE | 0.0011269181 |
| Agent 0 max action difference after folding | 0.0000043213 |
| Agent 1 max action difference after folding | 0.0000083745 |
| BC + closed-loop evaluation wall time | 29.44 s |

Export `best.pt` accepts raw 66D actor inputs. Checkpoint SHA-256:
`8e88e30e4d9ec87a462596a46c7aa5fd1e83d7ac51b0bfcfd3f595c380273979`.
Critics và alpha không train bằng BC. Std output head không có supervised target.
Loss curves: `experiments/phase3/bc_pipeline_audit/bc_figures/bc_loss.{png,pdf}`.

## Paired preliminary evaluation

Deterministic policies, same CUDA runtime, seed 20000, **10 episodes × 200 steps**
mỗi checkpoint. Restored dedicated RNG sequence và initial-state hashes match:
`19841a3401962f77201c881bd1c5865b19aa9a104e7c42c02926d6ccedfd7721`.

| Checkpoint | Ever-success | Final-success | Mean return |
| --- | ---: | ---: | ---: |
| Historical scratch MASAC `smoke_seed0/final.pt` (50k) | 0/10 = 0% | 0/10 = 0% | 19.5494 |
| BC only `bc_seed42/best.pt` | 7/10 = 70% | 2/10 = 20% | 30.8481 |

BC mean first-success among successful episodes: 99.86 steps / 4.993 s. Difference
between ever and final shows many successful lifts are not retained until horizon.
This is **one checkpoint per method, 10 paired initializations**, not a 3-seed RL
comparison, statistical significance result, or compute-matched comparison: BC
adds expert supervision. No claim that BC + 300k MASAC improves learning yet.

Bar chart này không chứng minh BC bão hòa hoặc scratch sẽ vượt BC về sau. Protocol
300k × 3-seed đối chứng, cách đọc entropy/MSE và fixed SR milestones nằm trong
[BC-init vs scratch comparison](bc_vs_scratch_protocol.md). Dataset hiện tại là
scripted demonstrations, không phải human demonstrations.

Evidence: BC `evaluation_episodes.csv` / `summary.json`; scratch
`experiments/phase3/bc_pipeline_audit/baseline_paired_eval_cuda/`; comparison figure
`bc_figures/paired_checkpoint_success.{png,pdf}` and `paired_comparison.json`.
An earlier CPU scratch cross-check is retained separately in `baseline_paired_eval/`;
its return differs from CUDA and is not the paired comparison used above.

## Short MASAC integration

Run: `experiments/phase3/bc_pipeline_audit/integration_1000/`.

- 1,000 env steps, 745 updates, five completed training episodes; exit 0.
- **Test-only overrides**: warmup 256, batch 32, replay 2,000, eval every 500,
  eval/final 2 episodes. Nominal `bc_pilot.yaml` remains 300k / warmup 10k / batch
  256 / replay 300k / eval every 10k, final 50.
- Actor-only BC initialization; critics/targets, alpha, replay and optimizers fresh.
- NaN/Inf/action-bound violations = 0; full-horizon timeout bootstrapping exercised.
- Final deterministic evaluation: **ever 0/2, final 0/2**, mean return 22.2328.
  This is a wiring/numerics check; it does not establish improved RL behavior.
- Final alpha values 0.17956 / 0.18010. Actual alpha/entropy/action/Q/TD/gradient
  PNG/PDF curves under `bc_pipeline_audit/integration_figures/`.
- Pilot analyzer correctly returns INCOMPLETE for this 1k run; no aggregate 3-seed
  success rate is manufactured.

The previous user-run scratch pilot controller exception at attempted step 18,713
is still unresolved; collection and short integrations do not prove 300k simulator
reliability. Existing failure traceback/state capture remains available for pilot
diagnosis. No simulator dependency changes or claimed controller repair here.

## Implementation and verification

New implementation: `src/rel301m/imitation/{__init__,scripted_teacher,demonstrations,
collect,bc,train_bc,warm_start}.py`, `configs/imitation/bc.yaml`,
`configs/experiment/bc_pilot.yaml`, `tests/test_imitation.py`, `scripts/plot_bc.py`,
and these two docs. Existing training/logger gain optional actor-only initialization
and explicit method provenance. `scripts/analyze_pilot.py` and its tests gain BC
provenance checks and optional paired scratch-reference comparison. README links
the new workflow. Historical run artifacts are preserved.

MASAC equations, networks, replay implementation, simulator factory/wrapper,
Phase 1 snapshots, original smoke/pilot/final configs, simulator/dependency pins
are unchanged from the before-BC SHA snapshot. No predictor, communication,
delay/dropout, Robomimic dependency or BC auxiliary RL objective added.

**109 tests PASS, 0 failed, 0 skipped, 68.64 s**, full suite exit 0.
Test evidence: `experiments/phase3/bc_pipeline_audit/final_pytest.xml` and
`logs/pytest_bc_pipeline_final.log`. Tests cover actor isolation, normalization
equivalence, episode splits, native successful collection + BC/export/evaluation,
actor-only warm start preserving RNG/Q/alpha/optimizers, malformed checkpoints,
unchanged pilot settings and method-comparison pairing, plus previous Phase 0–3
regressions. Artifact hashes, dataset checks and protected-source comparison are
recorded separately in `bc_pipeline_audit/verification.json`.

Phase 0 isolated environment was rechecked independently: RESULT PASS, Python
3.10.12 / robosuite 1.5.2 / MuJoCo 3.9.0, no PyTorch installed and no ROS pollution
in the research Python path. Evidence under `bc_pipeline_audit/phase0_regression/`
and `phase0_regression.md`; original reproducibility report is preserved.

Reproducibility note: Git has no HEAD yet (`uncommitted (no HEAD yet)`); dataset/run
manifests record this rather than inventing a commit SHA. Data, videos, checkpoints,
plots and logs live under ignored output directories, code/docs remain in repo.
