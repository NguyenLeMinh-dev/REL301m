# Nominal MASAC pilot — frozen protocol

Mục tiêu: xác định baseline có học `TwoArmLift` nominal đủ ổn định trước robustness.
**Pilot chưa được chạy bởi Codex**; người dùng tự chạy các lệnh dưới đây sau audit.
Hiện có một run do người dùng khởi chạy bị gián đoạn ở step 18,713 do
`opspace_matrices`; chưa có completed pilot 300k × 3 seeds. Logging/test follow-up
và trạng thái controller được ghi tại [pilot_diagnostics.md](pilot_diagnostics.md).
Engineering PASS không đồng nghĩa learning PASS. Historical 50k smoke có internal
1/10 ever-success và independent seed 20000 0/10, chưa giải được task ổn định.

## Frozen settings

- Training seeds: **0, 1, 2**, **300,000 steps mỗi seed**, một env mỗi run.
- Hai independent MLP actors 66→256→256→Gaussian/tanh(7), shared-reward twin centralized
  Q có input 133 và target Q. Không GRU, delay/dropout, predictor hoặc communication.
- Actor/Q/alpha lr 3e-4; gamma 0.99; tau 0.005; batch 256; buffer 300k; warmup 10k;
  một update/environment step; initial alpha 0.2; entropy target -7 mỗi actor.
- `bootstrap_time_limits=true`; raw done/timeout và success semantics giữ nguyên.
- `configs/algo/masac.yaml` không đổi; runtime versions giữ nguyên stack đã smoke.

## Common evaluation

Mọi run dùng **eval_seed=20000**, deterministic `tanh(mean)`, initialization RNG
snapshot được restore trước mỗi eval. Mỗi episode lưu hash initial critic state;
random và final policy phải có cùng initialization-sequence hash trước khi so sánh.
Training seeds khác nhau vẫn nhận cùng evaluation initializations.

- Periodic evaluation: **10 episodes / 10,000 steps** để theo dõi curve.
- Random reference và final pilot evaluation: **50 episodes** mỗi trained seed.
- Best checkpoint chọn trên cùng first 10 episodes; report primary learning gate
  bằng **final.pt** sau 300k, không chọn checkpoint tốt nhất post hoc.
- Dataset 20000 khác training nhưng được dùng làm validation/monitoring. Independent
  final test nếu cần phải dùng seed khác, giống nhau giữa mọi method (ví dụ 30000).
- Primary SR là ever-success fraction; final-success SR được báo riêng. Return là
  shaped reward sum. First-success seconds = step/20; không success thì null.
- Communication cost **N/A**. Chưa tính RD; SR0=0 thì RD không xác định.

## Chạy pilot tuần tự

Training CLI có progress bar mặc định: steps, percent, tốc độ, elapsed/ETA và
giai đoạn evaluation/saving. Qua `tee` sẽ in progress định kỳ mỗi 10 giây; terminal
trực tiếp redraw mỗi giây. Đợi trạng thái `completed`, vì 100% steps vẫn có thể
đang final evaluation. ETA là ước lượng và sẽ thay đổi sau warmup.
Các warning optional không dùng trong Panda/BASIC được lọc theo message/module;
warning khác và mọi error vẫn hiển thị. Thêm `--show-all-warnings` để debug.

Chạy từ research root. Loop dưới đây chạy 3 seeds lần lượt, dừng khi bất kỳ run
fail, tạo batch folder mới để tránh overwrite. Giữ terminal/process chạy đến hết;
checkpoint hiện tại không hỗ trợ exact resume vì không lưu replay/RNG/env state.

```bash
cd /home/minh/Documents/REL301m/REL301m-research
set -euo pipefail
pilot_batch="experiments/phase3/pilot_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$pilot_batch" logs
for seed in 0 1 2; do
  .venv-phase3/bin/python -I -u -m rel301m.training.train \
    --config configs/experiment/pilot.yaml \
    --seed "$seed" --device cuda \
    --run-dir "$pilot_batch/seed$seed" \
    2>&1 | tee "$pilot_batch/seed${seed}.log"
done
printf 'Pilot artifacts: %s\n' "$pilot_batch"
```

Mỗi run tự ghi 50-episode final evaluation trong `summary.json`, `evaluation.csv`
và `eval_0300000_episodes.csv`; không cần rerun cùng 50 episodes để có primary metric.
`random_baseline.json` dùng cùng sequence. TensorBoard và CSV loss logs có alpha,
entropy, Q/target means và gradients; `diagnostics.json` có gradient minima,
action bounds, timeout counts và initialization hashes. Instrumentation bổ sung
`log_pi_mean`, TD percentiles, action std/saturation, gradient maxima và initial
parameter hashes; định nghĩa metrics nằm trong [pilot diagnostics](pilot_diagnostics.md).

Nếu muốn kiểm tra độc lập seed 30000 sau khi có checkpoint:

```bash
# Thay đường dẫn bằng pilot_batch đã in và seed muốn evaluate.
.venv-phase3/bin/python -I -m rel301m.evaluation.evaluate \
  --checkpoint "$pilot_batch/seed0/final.pt" \
  --episodes 50 --seed 30000 --device cuda
```

Theo benchmark historical smoke, 3 seeds pilot dự kiến khoảng 4–5 giờ tuần tự,
chỉ là extrapolation, chưa phải runtime đã đo. Final eval/reference 50 episodes
thêm overhead nhỏ; throughput phụ thuộc system load và trạng thái simulation.
Không cần vectorized env workers cho pilot này.

## Learning gate trước khi tăng compute

Đây là **project gate**, không phải threshold chuẩn từ literature. Sau ba runs,
report SR mỗi seed, mean/std giữa seeds, random reference, final-success SR,
first-success và learning curves. Kiểm tra hashes/common episode count trước khi
so sánh; không suy success từ return.

Chỉ chuyển sang nominal final/robustness khi:

1. Periodic SR learning curves tăng rõ; return tăng đơn lẻ không đủ.
2. Ít nhất **2/3 seeds** vượt random SR trên cùng 50-episode evaluation sequence.
3. Không có seed collapse hoàn toàn (final ever-success SR=0) hoặc numeric failure.
4. Mean nominal ever-success SR giữa 3 seeds đạt mục tiêu **≥50%**; báo riêng
   final-success để thấy success có được duy trì hay chỉ xảy ra thoáng qua.

Nếu ≥2/3 seeds vẫn SR<20% sau 300k, baseline còn yếu: kiểm tra reward → action
scaling → done/timeout → Bellman target → alpha/entropy → gradients. Không thêm
GRU/prediction hoặc tự tăng budget để che baseline chưa học được. Fixed-alpha chỉ
là diagnostic ablation nếu curves cho thấy auto-alpha có vấn đề, chưa triển khai
hoặc chạy ablation trong audit này.

Final preset 700k, seeds 0/1/2, final evaluation 100 episodes, chỉ dùng khi pilot
đạt learning gate. Engineering/test PASS hiện tại chưa cho phép kết luận learning
PASS; kết quả pilot còn pending.

Sau khi có đủ artifacts, dùng `python3 scripts/analyze_pilot.py --batch "$pilot_batch"`
để recompute gate; thêm `--plots` với interpreter có Matplotlib để xuất PNG/PDF.
Curve review là bước explicit, không tự đổi gate sau khi có kết quả.
