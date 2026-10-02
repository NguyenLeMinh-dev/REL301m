# Nominal pilot diagnostics

Checklist trong research report được triển khai dưới dạng instrumentation và
post-processing. Budget/config vẫn là 300k × seeds 0/1/2, actor 66 mỗi robot,
action 7 mỗi robot, centralized state 119, Q input 133; automatic alpha,
entropy target -7/agent và initialization được giữ nguyên.

## Trạng thái có bằng chứng

Run `experiments/phase3/pilot_seed0_20261002T065403546533Z` dừng ở attempted
step **18,713** với `SystemError` từ `opspace_matrices`. Có 93 completed
training episodes và checkpoint 10k, chưa có final 300k hay đủ 3 seeds.
Vì vậy trạng thái pilot là **INCOMPLETE**, không phải learning PASS hoặc
bằng chứng baseline thất bại sau 300k. Không suy nguyên nhân singularity,
Numba hay hardware chỉ từ thông báo `SystemError` này: traceback gốc chưa
được lưu trong artifacts cũ.

Instrumentation mới lưu `failure.json` chứa full traceback,
`diagnostics.json` có trạng thái failed và `failure_state.npz` gồm qpos,
qvel, critic state và joint action khi lỗi xảy ra. Đây là evidence cho
controller debugging, không phải exact-resume checkpoint. Lỗi controller
cũ chưa được chứng minh đã sửa; integration 1k không xác nhận độ ổn định
300k. Source simulator và package versions được giữ nguyên.

## Console progress và warning policy

Training CLI mặc định hiển thị ASCII progress bar theo env steps, elapsed, throughput,
ETA và phase (random baseline, warmup, training, evaluation, saving). Evaluation
có episode counter; `completed` chỉ in sau khi final checkpoint và summary được
ghi. Pipe qua `tee` dùng newline progress định kỳ 10 s, terminal trực tiếp redraw
1 s. `--no-progress` tắt riêng bar; CSV/TensorBoard/diagnostics vẫn được ghi.

Chỉ bỏ các notice đã xác minh không ảnh hưởng Panda + BASIC: private macros,
optional `robosuite_models`, GR1 Mink IK và unused body parts left/torso/head/base/legs.
Filter được cài trước import robosuite để xử lý cả startup notices. Không dùng
`warnings.ignore`, không tăng global log level và không redirect stderr vào null.
Unknown/numeric/controller warnings cùng mọi errors/tracebacks được giữ;
`--show-all-warnings` giữ cả known notices khi debug.

Verification: 119 tests PASS (64.65 s), gồm existing 1k-step training integration,
callback evaluation invariance, ETA/failed/completed states và warning-filter safety.
CLI smoke 400 steps hoàn tất qua `tee` với final eval/save progress và không có known
startup warnings. Đây chỉ là console verification, không phải learning pilot.
Evidence: `experiments/phase3/progress_audit/`, `logs/pytest_progress_full.log` và
`logs/progress_cli_smoke.log`. Không đổi algo/config/dependency versions.

## Initialization và optimizer metadata

`metadata.json` ghi:

- `initialization`: `torch.nn.Linear default reset_parameters`.
- `initial_parameter_sha256`: hash toàn bộ named parameters và từng
  actor/Q/target, lấy ngay sau construction trước optimizer updates.
- `optimizer_class` và `optimizer_settings`: Adam, lr, betas `(0.9,0.999)`,
  epsilon `1e-8`, weight decay `0`, amsgrad `false` theo runtime thực tế.

Không chạy custom initializer hay reseed khi lấy hashes. Actor/Q Linear
weights và biases có bounds theo `1/sqrt(in_features)`, đúng implementation
PyTorch đang cài. Tham khảo
[PyTorch Linear](https://docs.pytorch.org/docs/2.7/generated/torch.nn.Linear.html)
và [Adam](https://docs.pytorch.org/docs/2.7/generated/torch.optim.Adam.html).

## Định nghĩa metrics

`losses.csv` và TensorBoard có thêm các diagnostics, lấy từ current replay
minibatch; cadence là `log_interval=100`, không thêm action samples.

| CSV keys | Ý nghĩa | TensorBoard |
| --- | --- | --- |
| `log_pi_mean_0/1` | Mean log density sau tanh và affine action transform | `train/log_pi_mean_0/1` |
| `entropy_0/1` | Chính xác `-log_pi_mean` | `train/entropy_0/1` |
| `alpha_0/1` | Temperature sau update | `train/alpha_0/1` |
| `action_std_0/1` | Mean Gaussian std **trước tanh**, trước actor optimizer step | `action/std_0/1` |
| `action_sample_std_0/1` | Population std mỗi action dimension trên batch, rồi mean; sau tanh/scaling | `action/sample_std_0/1` |
| `action_saturation_0/1` | Fraction các components có `abs((a-bias)/scale)>=0.95`, sử dụng live bounds | `action/saturation_0/1` |
| `q1_mean`, `q2_mean` | Q trên replay state/action trước critic optimizer step | `critic/q1_mean`, `critic/q2_mean` |
| `target_q_mean` | Mean Bellman target **y**; alias của legacy `target_mean` | `critic/target_q_mean` |
| `td_abs_mean`, `td_abs_p50/p95/p99` | Pool `abs(Q1-y)` và `abs(Q2-y)` trước optimizer step | `critic/td_abs_*` |
| `actor_0/1_grad_norm`, `q_grad_norm` | L2 norm trước optimizer step | `grad/actor_0/1_norm`, `grad/q_norm` |

Action sample std chứa variation giữa states trong batch, không phải
conditional policy std. Pre-tanh std cũng không phải bounded-action entropy.
α và entropy được vẽ riêng; differential entropy có thể âm.
Với alpha loss hiện tại, entropy trên target -7 dẫn tới alpha giảm; entropy
dưới target dẫn tới alpha tăng. Direction được kiểm tra bằng controlled tests,
không kết luận “alpha collapse” chỉ từ temperature nhỏ. Target `-dim(A)`
là heuristic của
[Berkeley SAC reference](https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py).

Gradient minima/maxima theo dõi mọi optimizer update. JSON ghi
`last_diagnostic_step` để biết batch gần nhất được đo chi tiết; trên một run
failed, trường có suffix `_final` nghĩa là giá trị cuối có sẵn, không phải kết
quả completed pilot. Khi chưa có updates, các metrics chưa đo là null và
`loss_check_status=NOT_EXERCISED`.

Numeric checks kiểm tra replay batches, Q/targets, losses/gradients và
parameters; executed actions phải hữu hạn và nằm trong live bounds.
NaN/Inf/bound violations được đếm rồi dừng run; không clip hoặc bỏ transition
bị lỗi. Target gradients, không có updates sau warmup và paired evaluation
hash mismatch cũng dừng run. `global_step` trong diagnostics đếm completed
environment transitions; `attempted_step`/failure step có thể cao hơn một.

## Phân tích và figures

Chạy từ research root. Analyzer không train, không gọi simulator và dùng
`final.pt` để kiểm tra primary artifact, không tự chọn best checkpoint.

```bash
cd /home/minh/Documents/REL301m/REL301m-research
# Đặt đường dẫn batch đã chạy, không tự đoán “newest” directory.
pilot_batch=experiments/phase3/pilot_YOUR_TIMESTAMP
python3 scripts/analyze_pilot.py --batch "$pilot_batch"
```

Để xuất PNG/PDF, dùng system Python có Matplotlib sẵn trên workstation;
không thay dependencies trong research venv:

```bash
MPLCONFIGDIR=/tmp/rel301m-matplotlib python3 scripts/analyze_pilot.py \
  --batch "$pilot_batch" --plots
```

Mỗi lần tạo một analysis folder mới với `pilot_report.json` và figures:
ever-SR, final-SR, return, alpha, entropy, pre-tanh std, Q/Bellman target,
TD percentiles, saturation, gradients và first-success. Mỗi seed có curve
riêng; mean chỉ vẽ tại shared checkpoints khi đủ 3 seeds. Metrics không có
trong CSV được liệt kê `unavailable_metrics`, không tạo số giả. First-success
không có successful episodes được bỏ trống.

Có thể phân tích các standalone run directories bằng `--runs PATH0 PATH1
PATH2`, hoặc chỉ một run failed để xem partial curves. Đã tạo figures từ
interrupted seed 0 tại
`experiments/phase3/pilot_diagnostics_audit/interrupted_seed0_analysis/`.
Run cũ chưa có action-std/saturation và TD percentiles, nên các plots đó
được đánh dấu unavailable.

## Learning gate và status

Analyzer recompute ever/final SR, return và first-success trực tiếp từ
50-episode final CSV, so sánh summary và random reference, kiểm tra horizon,
hashes, numeric counters, update count 290001, periodic schedule và frozen
config. Versions, algorithm/source hashes và evaluation sequences phải
match giữa ba seeds. Có 150 final policy episodes nhưng seed-level std là
sample std giữa **3 seeds**, không gộp như 150 independent training runs.

- `INCOMPLETE`: thiếu một seed/completed 300k result, hoặc run bị gián đoạn.
- `INVALID`: completed artifacts không vượt protocol/numeric consistency checks.
- `LEARNING_FAIL`: mean SR <50%, dưới 2 seeds beat random, có zero-SR seed,
  hoặc qualitative learning-curve review fail.
- `REVIEW_REQUIRED`: numeric learning gates pass, curve review còn pending.
- `PASS`: tất cả checks và learning-curve review pass.

“Curve tăng rõ” là qualitative gate đã ghi trong frozen protocol; analyzer
không tự thêm một threshold hay kiểm định mới sau khi có kết quả. Sau khi
xem curves, ghi quyết định review rõ ràng:

```bash
MPLCONFIGDIR=/tmp/rel301m-matplotlib python3 scripts/analyze_pilot.py \
  --batch "$pilot_batch" --plots --learning-curve-review pass
```

Chỉ dùng `pass` khi review đáp ứng protocol; lựa chọn này không bỏ qua numeric
hay consistency gates. Lệnh chạy pilot nguyên bản nằm trong
[pilot_protocol.md](pilot_protocol.md). Fresh teammate 53D, fixed alpha,
target entropy -3.5 và longer seed là diagnostic ablations sau một completed
pilot không đạt gate, chưa được triển khai hoặc chạy ở bước instrumentation.
