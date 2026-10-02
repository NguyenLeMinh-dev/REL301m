# Phase 3 — Smoke run measured report

Measured 2026-10-02 on RTX 4060 Ti, NVIDIA driver 610.43.02, Python 3.10.12.
PyTorch 2.7.1+cu128, CUDA runtime 12.8, CUDA available=true;
TensorBoard 2.20.0. robosuite 1.5.2 / MuJoCo 3.9.0 / NumPy 1.26.4 unchanged.

**Engineering validation PASS; smoke budget completed.** Có success thoáng qua,
chưa có evidence rằng policy lift ổn định hoặc generalize tốt. Không chạy pilot,
final, predictor, uncertainty hay adaptive communication.

## Validation

- **65 passed, 0 failed, 0 skipped** (41.10s): 46 Phase 1/2 regression tests + 19 new Phase 3 tests.
- Live 1k-step integration: 745 updates, 5 full training episodes, CSV/TensorBoard và checkpoint round-trip PASS.
- Phase 0 check PASS trong cả `.venv-phase0` (không Torch) và `.venv-phase3` (CUDA available).
- Source và snapshots Phase 1/2 không thay đổi; code/config/contract hashes của smoke không drift trong quá trình chạy.
- Git SHA: `uncommitted (no HEAD yet)`. Mỗi run có source SHA256, git status và pip freeze; chưa tạo commit/push.

## Smoke và success semantics

Run: `experiments/phase3/smoke_seed0/`, training seed 0, 50,000 steps,
40,001 gradient updates, 250 episodes đủ horizon 200; warmup 10k, batch 256,
capacity 300k. Networks: 86,542 mỗi actor, 100,353 mỗi Q/target Q;
tổng 574,496 network parameters + 2 log-alpha scalars.

Random reference trước training: mean return 6.156,
ever-success 0/10. Dedicated eval environment dùng seed 10000, RNG tiếp tục qua
resets/evaluation checkpoints; không phải paired identical initial states với random reference.

| Training steps | Eval mean return | Ever-success rate | Final-success rate | First-success mean step |
| ---: | ---: | ---: | ---: | ---: |
| 10000 | 5.195 | 0% | 0% | null |
| 20000 | 0.339 | 0% | 0% | null |
| 30000 | 5.368 | 0% | 0% | null |
| 40000 | 10.163 | 0% | 0% | null |
| 50000 | 16.145 | 10% | 0% | 96.0 |

Training có 2/250 episodes ever-success,
0/250 final-success. Evaluation cuối có
1/10 ever-success, first-success step 96, nhưng 0/10 final-success. Success được
đo trực tiếp từ pot-height predicate `_check_success()`, không từ return/tilt/threshold.

CLI evaluation độc lập, deterministic actors, seed **20000**, 10 episodes:
mean return **19.549**, ever-success **0/10**, final-success **0/10**,
first-success=null. Số episodes/seeds này chỉ cho smoke evidence; chưa đủ để kết
luận generalization hay cải thiện thành công ổn định. Return chỉ là shaped reward signal.

## Benchmark thực tế

- Wall time: **815.4s (13.59 phút)**.
- End-to-end: 61.32 training steps/s, bao gồm evaluations và artifacts.
- Simulation/reset: 420.7s, 118.85 steps/s.
- Gradient updates: 254.7s, 157.03 updates/s.
- Evaluations: 108.4s.
- PyTorch peak allocated: **24.55 MiB**;
  peak reserved: **30.00 MiB**.
- `nvidia-smi` samples cho training process: **188 MiB** (không gọi sampled memory là peak).
- Replay capacity tối đa trên CPU: 622,800,000 bytes (~593.95 MiB).

CPU simulation chiếm nhiều wall time hơn updates trong run này. Các ước tính
pilot/final ban đầu chưa phải kết quả; hiệu năng có thể thay đổi theo robot state,
load workstation và tỷ lệ updates sau warmup. Chưa chạy budgets 300k×3 hoặc 700k×3.

## Diagnostic checkpoint cuối

Probe seed 30000, 200 deterministic steps; chỉ update in-memory copy để kiểm tra
gradients, không overwrite hoặc tiếp tục train `final.pt`.

- Wrapper reward bằng upstream reward, max absolute difference **0**.
- Live action bounds [-1,1]; concat đúng robot order; không action vượt bounds.
- Live OSC_POSE delta scale: translation ±0.05 m, rotation ±0.5 rad, frame `base`.
  Một gripper command mỗi actor được upstream GRIP controller xử lý; không thêm scaling.
- Tỷ lệ sampled components |action|≥0.99: **0.25%** trong probe.
- Một raw done và một timeout ở step 200; Bellman terminal mask **0** khi bootstrap_time_limits=true.
  Target dùng next observation trước reset.
- Bellman target so với công thức tính riêng, max absolute difference **0**;
  twin-Q minimum và tổng entropy hai actors được giữ đúng.
- Alpha cuối: 0.00049851, 0.00057905; target entropy -7 mỗi actor.
- Toàn bộ loss logs hữu hạn; Q/actor gradient norms khác 0. Các kiểm tra này không chứng minh critic ước lượng chính xác ở mọi state.

Artifacts chính: `final.pt`, `best.pt`, periodic checkpoints, `summary.json`,
`diagnostics.json`, `resolved_config.yaml`, `env.yaml`, `metadata.json`,
`requirements.freeze.txt`, `episodes.csv`, `losses.csv`, `evaluation.csv`,
`evaluation_seed20000/`, và `tensorboard/`. Xem [baseline contract](baseline_masac.md)
để chạy lại hoặc đánh giá checkpoint với seed khác.
