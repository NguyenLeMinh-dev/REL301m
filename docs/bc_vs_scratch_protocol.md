# BC-init MASAC vs scratch MASAC — comparison protocol

Date: 2026-10-02. Thử nghiệm cần trả lời hai câu riêng: BC initialization có giúp
đạt mức success định trước với ít **RL steps** hơn không, và policy có đạt mức
success đủ ổn định sau 300k không? Code BC → MASAC đã có. Pilot dài do người dùng
tự chạy; tài liệu này không báo kết quả chưa được đo.

## Đối chiếu bản đề xuất với evidence hiện có

| Nội dung | Kết luận đúng ở thời điểm hiện tại |
| --- | --- |
| BC bão hòa rồi scratch MASAC vượt lên | Chưa có evidence. `paired_checkpoint_success` là bar chart của hai checkpoint, không phải SR learning curve. |
| BC-only tốt hơn checkpoint scratch 50k | Trên 10 paired initializations/CUDA đã đo: BC ever 70%, final 20%; scratch ever/final 0%. Chỉ là preliminary checkpoint comparison. |
| BC→MASAC đã cải thiện | Chưa xác nhận. Integration 1k có 745 updates, final eval 0/2 success, không phải 300k learning result. |
| Critic input 119D | State 119D; Q concatenates joint action 14D, nên Q input **133D**. |
| Dataset hiện tại là human `demo.hdf5` | 40 **scripted**, synchronized dual-arm demos, NPZ state/action data; không gọi chúng là human demonstrations hoặc R2BC. |
| Replay có thể prefill hoặc empty tùy run | Trong đối chứng này replay **empty**, optimizers/alpha/critics mới và 10k random-action warmup ở cả hai nhóm. Prefill là ablation khác. |
| 2/3 seeds thắng chứng minh significance | Đây là project criterion cho consistent improvement; 3 seeds không tự tạo ra statistical significance. |

BC có thể gặp covariate shift vì các hành động của policy ảnh hưởng observations
trong tương lai. Đây là vấn đề đã được nghiên cứu, không phải bằng chứng rằng
checkpoint hiện tại bắt buộc có một performance ceiling hoặc sẽ bị scratch RL
vượt qua. Xem [Ross et al., DAgger (2011)](https://arxiv.org/abs/1011.0686).
Điểm hypothesis cần kiểm tra ở project này là **BC-init tăng sample efficiency**;
không mặc định kết luận từ nguyên lý rằng MASAC sẽ học thành công.

Robosuite hỗ trợ [human demonstrations và integration với Robomimic](https://robosuite.ai/docs/algorithms/demonstrations.html).
NPZ collector hiện tại đã ghi actor observations đúng contract; không cần thu
lại demos hoặc thêm HDF5/Robomimic để bắt đầu đối chứng.

## Đọc entropy, alpha và action MSE

SAC tối ưu reward cùng entropy; alpha là trọng số của entropy, automatic tuning
điều chỉnh nó theo entropy target. Xem
[Haarnoja et al., Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905).
Trong code này `entropy_i = -mean(log_pi_i)` của sampled squashed policy trên replay
observations; không phải entropy của deterministic evaluation policy.

Target entropy -7/agent là differential entropy, có thể âm. Với loss
`-log_alpha * (log_pi + target_entropy)`, gradient theo log-alpha bằng
`estimated_entropy - target_entropy`. Entropy cao hơn target làm alpha giảm là
hành vi kỳ vọng của automatic tuning. **Alpha giảm riêng lẻ không chứng minh
exploration collapse**. Entropy cao thể hiện randomness cao hơn theo phân phối đang
đo, có thể làm hành động thiếu hiệu quả; không gọi đó là under-exploration. Kiểm tra
cùng SR, Gaussian std, sampled action std, saturation, gradients và Q/TD residual.

BC training/validation MSE đo imitation trên expert-state distribution. Train MSE
tiếp tục giảm còn validation MSE không cải thiện là dấu hiệu generalization gap;
không đủ để quy nguyên nhân closed-loop failure cho overfit. MSE thấp không bảo
đảm giữ được pot tới horizon. BC selected epoch vẫn là 134 theo validation; held-out
test MSE không tham gia chọn epoch. Gaussian log-std head không được supervised
bằng MSE; shared encoder có thay đổi khi BC train mean.

Chart MSE hiện tại thuộc **BC epochs**; SAC actor loss không phải expert-action
MSE và project chưa log imitation MSE ở mỗi RL checkpoint. Không gán nhãn actor
loss thành MSE. Nếu thêm diagnostic đó sau này, dùng fixed validation demos và
không dùng final test demos để chọn RL checkpoint; MSE tăng trong RL cũng có thể
đi cùng success tăng nếu policy tìm hành động khác expert.

## Frozen comparison

| Item | Scratch | BC-init |
| --- | --- | --- |
| Config | `configs/experiment/pilot.yaml` | `configs/experiment/bc_pilot.yaml` |
| Actors | Random initialization | `bc_seed42/best.pt`, same fixed checkpoint for all RL seeds |
| Training seeds / RL steps | 0, 1, 2 / 300k each | 0, 1, 2 / 300k each |
| Actor / state / action dims | 66 each / 119 / 7 each | Same |
| Twin Q / targets | Input 133, fresh seed-specific weights | Same fresh weights for matching seed, hash-checked |
| Hyperparameters | Existing `configs/algo/masac.yaml` | Same file |
| Replay / optimizers / alpha | Empty / fresh / 0.2 | Empty / fresh / 0.2 |
| Warmup | 10k random actions | 10k random actions |
| Environment / runtime / device | Frozen Panda + BASIC, CUDA | Same |
| Periodic monitoring | 10 episodes / 10k steps, seed 20000 | Same initial hashes |
| Primary final evaluation | `final.pt` at 300k, 50 episodes | Same 50 initial hashes |

Matching seed does not mean matching actor weights: initialization is the deliberate
treatment. Matching critic/target hashes and a RNG-preserving actor-only loader
make this distinction explicit. Use the same source/runtime for all six runs;
do not tune alpha, reward, warmup or batch separately after seeing method results.

Extra BC cost is reported separately: 63 collection attempts × 200 = 12,600 sim
steps, 40 accepted demos = 8,000 transitions, 300 epochs = 7,200 optimizer updates
per actor. An advantage in RL steps is not automatically an advantage in total
simulator samples or wall time. Shared BC checkpoint seed 42 means three RL seeds,
not three independent BC datasets/trainings.

## Milestones and gates

`scripts/analyze_pilot.py` reads/recomputes completed artifacts without training.
For each valid seed it exports `final_results.csv` (50-episode metrics) and
`success_milestones.csv` (monitoring diagnostic):

- Fixed SR threshold **0.50**, decided before the 300k comparison.
- `first_observed_step`: first 10k-grid evaluation with SR ≥0.50.
- `two_consecutive_confirmation_step`: second of two adjacent qualifying evals.
  This is additional monitoring evidence, not proof of sustained population SR.
- The final 300k evaluation has 50 episodes; milestone computation uses **its first
  10** so monitoring sample count stays constant at all 30 grid points. Final SR
  gate still uses all 50 episodes.
- Missing threshold crossing is `null / not_reached`, not 300k or infinity. Step
  savings is computed only when both methods cross; it is a grid-based descriptive
  difference, not exact convergence time. These evals are the monitoring set, not
  an untouched independent test.

Existing nominal learning gate remains: mean final SR ≥50%, ≥2/3 seeds beat their
paired random reference, no zero-SR seed/numeric failure, plus explicit curve
review. These are project thresholds, not a universal SAC standard. They are not
relaxed after looking at results.

The **additional warm-start gate** requires ≥2/3 strict wins over scratch (ties do
not count), a positive mean paired SR difference, and the existing candidate
learning gate. Analyzer reports `warm_start_gate.status` as NOT_MET,
REVIEW_REQUIRED or PASS. `comparison.status=COMPARED` only means artifacts were
comparable, and does not itself mean learning PASS or statistical significance.
Report both methods' per-seed SR/final-SR/return, mean ± sample standard deviation
across three seeds, paired deltas, success-conditioned first-success steps, and
the milestone observations; do not treat evaluation episodes as independent
training seeds.

Primary performance uses final.pt, not best.pt. An optional independent final test
must use a new common evaluation seed (e.g. 30000) for all methods and report it
separately. Fixed-alpha, more demos, replay prefill and longer training are later
ablations, not silent changes to this comparison.

## Commands: six runs, progress and filtered optional warnings

Execute manually from the research root. This alternates methods for each seed,
runs one process at a time, saves separate artifacts and stops on failure. Default
progress/ETA works through `tee`; known unused Panda/BASIC notices are filtered,
unknown warnings and errors retained. Add `--show-all-warnings` for debugging.
Wait for `completed` after final eval/save, not merely 100% env steps.

```bash
cd /home/minh/Documents/REL301m/REL301m-research
set -euo pipefail
comparison_batch="experiments/phase3/bc_vs_scratch_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$comparison_batch/scratch" "$comparison_batch/bc_init"
for seed in 0 1 2; do
  for method in scratch bc_init; do
    case "$method" in
      scratch) config="configs/experiment/pilot.yaml" ;;
      bc_init) config="configs/experiment/bc_pilot.yaml" ;;
    esac
    printf '\n%s — seed %s/3\n' "$method" "$((seed + 1))"
    .venv-phase3/bin/python -I -u -m rel301m.training.train \
      --config "$config" --seed "$seed" --device cuda \
      --run-dir "$comparison_batch/$method/seed$seed" \
      2>&1 | tee "$comparison_batch/$method/seed${seed}.log"
  done
done
printf '\nCompleted: %s\n' "$comparison_batch"
```

If BC-init alone is already running, keep that run; use the separate loops in
[BC guide](bc_warm_start.md) and [scratch protocol](pilot_protocol.md). Do not restart
an existing pilot to adopt this directory naming. After both batches finish,
pass their actual paths to `--batch` and `--reference-batch` below.

```bash
# Same shell: comparison_batch still points to the newly completed six-run batch.
MPLCONFIGDIR=/tmp/rel301m-matplotlib python3 scripts/analyze_pilot.py \
  --batch "$comparison_batch/bc_init" \
  --reference-batch "$comparison_batch/scratch" --plots
```

Default curve review is pending. After actual review, rerun analysis with
`--learning-curve-review pass` or `fail`; missing/invalid runs cannot bypass numeric
or provenance checks. Plots show SR/final-SR/return/first-success, alpha/entropy,
std/saturation, Q/target/TD and gradients for both groups. BC epoch MSE plot remains
separate. No 300k runs were launched to create this protocol.

## Current verification

21 focused analyzer tests PASS, including regression gates, strict wins/ties,
missing/mismatched critic hashes/device/seeds, final-50 vs monitoring-10,
censored milestones, table output and incomplete-run behavior. Evidence:
`experiments/phase3/comparison_protocol_audit/pytest.xml`,
`logs/pytest_comparison_protocol.log`. Synthetic fixtures are used only in tests;
real 1k BC-init artifacts still return INCOMPLETE with no fabricated aggregate.
The previous 119-test full suite PASS is recorded in `progress_audit/`; this
change edits analysis/tests/docs only and leaves runtime training/core/configs intact.

Reviewable source snapshot (code/config/contracts/tests/docs only, no venv/data/models):
`experiments/phase3/comparison_protocol_audit/source_snapshot.zip`. Its file hashes,
archive SHA-256 and CRC verification are recorded in `verification.json`. Git still
has no HEAD; this archive preserves the source revision without inventing a SHA
or committing/pushing on behalf of the user.
