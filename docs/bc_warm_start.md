# Scripted demonstrations → BC → actor-only MASAC warm start

Mục tiêu là cho hai actors học cách tiếp cận, kẹp và nâng pot trước khi bắt đầu
MASAC. Đây là BC trên demonstrations đồng bộ của hai robot, dùng code PyTorch
native của project. Khả năng cải thiện RL qua 3 seeds còn phải đo; kết quả đã chạy
nằm trong [BC pipeline report](bc_pipeline_report.md).

## Giữ nguyên environment và CTDE

Actor mỗi agent chỉ nhận 66D của Phase 2: own proprio 50 + own relative handle 3
+ shared object 13. Không dùng raw object-state, teammate proprio hoặc teammate
relative handle. Critic state vẫn là 119D; Q nhận state + joint action, nên **Q input
là 133D**, không phải 119D. Actions 7 + 7 theo live robot/controller layout, bounds
được lấy từ environment. Không sửa simulator, reward, horizon hoặc success.

Native `TwoArmLift._check_success()` kiểm tra pot bottom cao hơn tabletop 0.10 m;
predicate không trực tiếp yêu cầu cả hai grippers đang grasp. Ever-success và
final-success luôn dùng predicate này, không dùng return threshold. Để chọn demos
tốt hơn, collector yêu cầu final native success và ít nhất một frame success có
cả hai handle được grasp. Đây là **demo quality filter riêng**, không thay đổi task.

## Teacher và dataset

`ScriptedLiftTeacher` chạy state machine approach → descend → grasp → lift. Teacher
có privileged simulator/controller state và phối hợp hai tay; chỉ dùng để tạo
offline labels. Actors BC không đọc teacher state. Teacher xác định body-part
slices, delta frame, input/output scaling từ controller metadata, rồi giới hạn
commands trong live bounds. Panda finger closing direction được xoay vuông góc
handle bar trước khi grasp.

Mỗi accepted episode chứa đủ 200 steps, không dừng khi success. NPZ lưu raw float32
`o0/o1/s/a0/a1/r/next_o0/next_o1/next_s/done/timeout`, cùng native success/grasp trace,
teacher phase và qpos/qvel. Manifest lưu accepted và rejected attempts, seed, config,
measured dimensions/layout/bounds, package versions, Git SHA và SHA-256 mỗi file.
Loader kiểm tra hashes, full horizon, finite values, bounds và alignment current/next.

Split theo **episode**, seed 42: 30 train / 5 validation / 5 test cho 40 demos.
Không shuffle transitions trước khi chia split. Mean/std từng actor chỉ fit trên
train observations; std floor 0.01. Test demos không dùng chọn epoch hoặc fit stats.

Robosuite cũng hỗ trợ [human demonstrations](https://robosuite.ai/docs/algorithms/demonstrations.html).
Nếu chuyển sang Robomimic HDF5, `convert_robosuite.py` chỉ chuẩn hóa metadata;
còn phải chạy observation extraction, theo
[Robomimic dataset documentation](https://robomimic.github.io/docs/datasets/robosuite.html).
Pipeline hiện tại lưu observations trực tiếp, không cần cài thêm Robomimic.
Không triển khai Round-Robin BC/R2BC hoặc giả định mức cải thiện từ paper áp dụng
được cho experiment này.

## BC và normalization khi deploy

Hai Gaussian actors hiện có được train độc lập bằng MSE của bounded deterministic
action `tanh(mu)` so với expert action đã affine-normalize theo live bounds.
Architecture 66 → 256 → 256 → mean/log_std(7) giữ nguyên; lr 3e-4, batch 256,
300 epochs. Chọn checkpoint theo mean validation MSE của hai actors.

BC không train critics, alpha hoặc SAC losses. Log-std head không có supervised
target; shared encoder vẫn thay đổi khi train mean. Vì vậy MSE thấp không chứng
minh stochastic SAC exploration đã được calibrate hoặc closed-loop task ổn định.

Để live policy và replay tiếp tục nhận **raw** observations như baseline, transform
`(x - mean) / std` được fold vào Linear đầu tiên khi export:

```text
W_raw = W / std
b_raw = b - W_raw @ mean
```

Export kiểm tra action equivalence trên held-out data. Checkpoint vẫn dùng format
MASAC hiện tại, đọc được bằng evaluation/visualization cũ; critic weights bên trong
BC checkpoint là untrained và không được dùng làm initialization khi fine-tune.

## Actor-only MASAC initialization

`warm_start_from_bc` kiểm tra stage, raw-input transform, env contract, dimensions,
bounds, architecture và finite weights trước khi copy cả hai actors. Nó không
construct thêm model hoặc consume Torch RNG. MASAC critics/targets giữ initialization
theo RL seed; alpha bắt đầu 0.2, optimizers và replay mới. Không load BC optimizer,
prefill demos vào replay hoặc thêm BC regularization.

`configs/experiment/bc_pilot.yaml` dùng đúng `configs/algo/masac.yaml`: actor/Q/alpha
lr 3e-4, gamma .99, tau .005, batch 256, replay 300k, **10k random-action warmup**,
entropy target -7 mỗi actor, time-limit bootstrapping và một update/step. Warmup
vẫn random như scratch baseline để giữ hyperparameters. Tất cả RL seeds dùng cùng
BC checkpoint seed 42; đây là ba RL seeds với một fixed offline initialization,
không phải ba BC seeds độc lập.

## Reproduce demos và BC

Chạy từ research root. Các output directories dưới đây **mới**, để không overwrite
dataset/checkpoint đã đo. `collect` dùng external EGL renderer cho preview; actor
vẫn state-only, factory renderer flags giữ false.

```bash
cd /home/minh/Documents/REL301m/REL301m-research
demo_dir="data/demonstrations/two_arm_lift_scripted_recollect"
bc_dir="experiments/phase3/bc_seed42_retrain"
.venv-phase3/bin/python -I -m rel301m.imitation.collect \
  --episodes 40 --max-attempts 100 --seed 40000 --output-dir "$demo_dir"
.venv-phase3/bin/python -I -m rel301m.imitation.train_bc \
  --config configs/imitation/bc.yaml --dataset "$demo_dir" --output-dir "$bc_dir"
```

## Chạy RL 300k × 3 seeds bằng checkpoint đã có

Training CLI mặc định có progress bar: completed/total steps, percent, tốc độ,
elapsed, ETA và giai đoạn random baseline / warmup / training / evaluation / saving.
Terminal redraw mỗi giây; khi pipe qua `tee`, in dòng mới mỗi 10 giây và khi đổi
giai đoạn để log dễ đọc. ETA là ước lượng từ wall time và completed env steps,
sẽ thay đổi khi qua warmup hoặc evaluation. 100% steps vẫn có thể đang final eval;
chỉ dòng `completed` xác nhận đã ghi final checkpoint và summary.

CLI lọc các robosuite warning đã xác minh không dùng trong Panda + BASIC: private
macros, optional `robosuite_models`, Mink IK cho GR1 và config body parts không có
trên Panda. Unknown warnings, numeric/controller warnings và errors vẫn hiển thị.
`--show-all-warnings` khôi phục toàn bộ thông báo; `--no-progress` tắt riêng bar.
Không thêm package hoặc đổi dependency versions cho chức năng này.

Loop chạy tuần tự, dừng khi một seed fail; checkpoint không hỗ trợ exact resume
replay/RNG/env. Config tự lưu provenance BC checkpoint, YAML, Git SHA/status,
source hashes, pip freeze, CSV/TensorBoard và final checkpoint mỗi run.

```bash
cd /home/minh/Documents/REL301m/REL301m-research
set -euo pipefail
bc_batch="experiments/phase3/bc_pilot_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$bc_batch"
for seed in 0 1 2; do
  .venv-phase3/bin/python -I -u -m rel301m.training.train \
    --config configs/experiment/bc_pilot.yaml --seed "$seed" --device cuda \
    --run-dir "$bc_batch/seed$seed" \
    2>&1 | tee "$bc_batch/seed${seed}.log"
done
printf 'BC pilot artifacts: %s\n' "$bc_batch"
```

Nếu dùng BC checkpoint mới, thêm `--bc-checkpoint "$bc_dir/best.pt"` cho training.
Scratch 300k × 3 seeds dùng loop ở [pilot protocol](pilot_protocol.md). Hai methods
cần cùng runtime/core learner/env và eval initial states. Demos/BC là compute thêm:
báo riêng 12,600 collection steps, 8,000 accepted transitions và 7,200 BC optimizer
updates mỗi actor; 300k RL steps không bao gồm offline compute đó.

## Evaluation, figures và comparison

Protocol đối chứng đầy đủ, interpretation của entropy/MSE, SR milestones và
additional BC-vs-scratch gate: [BC-init vs scratch](bc_vs_scratch_protocol.md).

Periodic eval 10 episodes/10k steps; random và final eval 50 episodes cùng seed
20000/dedicated RNG snapshot. Primary checkpoint là `final.pt` tại 300k; không dùng
`best.pt` thay final để tính gate. Script recompute CSV metrics và kiểm tra paired
initial hashes, đủ ba seeds, package/core-source consistency, numeric checks và BC
checkpoint provenance. Mean/std và per-seed differences chỉ xuất khi dữ liệu đủ.

```bash
# python3 ở workstation đã có Matplotlib; không đổi research venv dependencies.
MPLCONFIGDIR=/tmp/rel301m-matplotlib python3 scripts/analyze_pilot.py \
  --batch "$bc_batch" --plots
# Khi scratch batch cũng hoàn thành, thay path dưới bằng batch thực tế.
scratch_batch="experiments/phase3/REPLACE_WITH_COMPLETED_SCRATCH_BATCH"
MPLCONFIGDIR=/tmp/rel301m-matplotlib python3 scripts/analyze_pilot.py \
  --batch "$bc_batch" --reference-batch "$scratch_batch" --plots
```

Default qualitative curve review là pending. Chỉ dùng `--learning-curve-review pass`
sau khi review learning curves; numeric gates vẫn giữ nguyên. Missing/partial runs
báo INCOMPLETE. `comparison.status=COMPARED` chỉ là descriptive paired comparison,
không phải significance test hoặc learning PASS.

Analyzer cũng xuất `final_results.csv` (all 50 final episodes) và
`success_milestones.csv` (fixed first 10 monitoring episodes, threshold 50%).
Milestone không đạt ghi null; không nội suy convergence time. Paired comparison
kiểm tra matching seed/device và initial critic/target hashes; warm-start gate
yêu cầu ≥2/3 strict wins, positive mean SR difference và nominal learning PASS.

BC loss và preliminary checkpoint comparison, xuất PNG/PDF:

```bash
MPLCONFIGDIR=/tmp/rel301m-matplotlib python3 scripts/plot_bc.py \
  --run experiments/phase3/bc_seed42 \
  --baseline-evaluation experiments/phase3/bc_pipeline_audit/baseline_paired_eval_cuda
.venv-phase3/bin/python -I -m rel301m.evaluation.visualize \
  --checkpoint experiments/phase3/bc_seed42/best.pt --mode video --episodes 1 --seed 20000
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-phase3/bin/python -I -m pytest -q tests
```

Sau khi có đủ 300k × 3 seeds, so sánh ever/final SR, return, first success,
alpha/entropy, action std/saturation, Q/target/TD residual và gradients. BC giúp
initial deterministic behavior trong kiểm thử hiện tại; RL fine-tuning có giữ
được và cải thiện behavior đó hay không vẫn là hypothesis cần đo.
