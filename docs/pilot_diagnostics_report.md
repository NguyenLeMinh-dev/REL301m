# Pilot instrumentation follow-up — 2026-10-02

**Instrumentation/regression PASS. Pilot INCOMPLETE. Learning gate PENDING.**
Codex hoàn thiện code/tests và commands theo lựa chọn trước đó của người dùng,
chưa chạy pilot 300k × 3 seeds hoặc các diagnostic ablations.

## Kết quả đo thực tế

| Check | Result |
| --- | --- |
| Full suite | **97 passed, 0 failed, 0 skipped**, 57.11 s |
| Live integration | **1,000 environment steps, 745 updates**, 5 full horizon-200 episodes, CUDA |
| Numeric integration | NaN=0, Inf=0, action-bound violations=0 |
| Phase 0 regression | System/simulator PASS trong `.venv-phase0`; PyTorch absent, CUDA unspecified, ROS pollution none |
| CUDA peak allocated | 28,736,512 bytes (~27.41 MiB) |
| Old/new MASAC update comparison | 5 updates trên CPU và 5 trên CUDA: all parameters, RNG và legacy metrics exactly equal |
| Instrumentation on/off comparison | RNG, parameters và Adam states exactly equal |
| Alpha loss sign | Entropy below -7 tăng alpha; above -7 giảm alpha |
| Metadata | Reproducible seed hashes; hashing không consume RNG; actual Linear/Adam defaults recorded |
| Guards/failure artifacts | Injected NaN, target gradient và controller failures dừng run, giữ diagnostics/traceback/state |
| Analyzer | Recompute episode metrics, verify paired hashes; incomplete/invalid data không PASS; curve review explicit |
| Protected files | **16 files unchanged**: configs, Phase 1 snapshots, env factory/wrapper, networks, replay, dependency pins |

Integration artifacts nằm tại
`experiments/phase3/pilot_diagnostics_audit/pytest_tmp/test_one_thousand_step_trainin0/integration/`.
Test evidence: `experiments/phase3/pilot_diagnostics_audit/pytest.xml`,
`logs/pytest_pilot_diagnostics.log`. Analyzer tests dùng synthetic schema fixtures;
các fixture PASS không phải kết quả learning thật.
Phase 0 evidence được ghi riêng dưới
`experiments/phase3/pilot_diagnostics_audit/phase0_regression/`, giữ historical report.

## Thay đổi

- `algorithms/masac.py`: detached diagnostics và finite guards; loss/target/optimizer
  math và action-sampling sequence giữ nguyên khi dữ liệu hợp lệ.
- `utils/diagnostics.py`: TD quantiles, action statistics, numeric error counts,
  TensorBoard tag mapping, valid failure JSON.
- `utils/logger.py`: initial parameter hashes và optimizer/default-init metadata.
- `training/train.py`: extended CSV/TensorBoard, gradient min/max, running/failure
  diagnostics, eval hash guard, simulator failure evidence.
- `scripts/analyze_pilot.py`: read-only run analysis; final 50-episode CSV thay vì
  return threshold; 3-seed mean/sample std; PNG/PDF curves, missing metrics explicit.
- `tests/test_pilot_diagnostics.py`, `tests/test_pilot_analysis.py` và existing
  integration test: numeric definitions, no RNG/gradient side effects, failure
  persistence và rejection of corrupt/incomplete pilot artifacts.
- Documentation và README liên kết commands/definitions/report mới.

## Pilot hiện có và figures

User-started seed 0 run `pilot_seed0_20261002T065403546533Z` dừng ở attempted
step **18,713** với `SystemError` từ native `opspace_matrices`. Chỉ có 93
completed training episodes, periodic eval/checkpoint 10k và không có final.pt
hay summary completed. Lỗi chưa được xác định root cause hoặc chứng minh đã sửa.
Traceback cũ không nằm trong run artifacts; các run mới sẽ lưu đầy đủ.

Đã xuất 7 nhóm figures thực tế của interrupted run, mỗi nhóm PNG + PDF:
ever-SR, final-SR, return, alpha, entropy, Q/target, gradients. Output:
`experiments/phase3/pilot_diagnostics_audit/interrupted_seed0_analysis/`.
Action std/saturation, TD percentiles chưa có trong logs cũ; first-success
không có successful eval episodes: ghi unavailable, không nội suy hay fabricate.
Report ghi **INCOMPLETE**, không gộp missing seeds vào mean/std.

Alpha và entropy dùng separate axes/figures. Curves trước khi run dừng cho
thấy entropy còn trên target -7, vì vậy chiều alpha giảm phù hợp với loss;
không có đủ evidence để kết luận đây là nguyên nhân task chưa được giải.

Các figures có diagnostics mới từ integration 1k nằm trong
`experiments/phase3/pilot_diagnostics_audit/integration_figures/` và được ghi rõ
INCOMPLETE đối với pilot gate. Chúng kiểm tra pipeline xuất figures, không
thay thế 300k learning evidence.

Protocol budget/hyperparameters, versions, state observations, reward/success
và done/timeout semantics không đổi. HDF5 replay persistence, fresh-teammate
119D actors, fixed alpha, entropy-target ablation, GRU và robustness chưa được
triển khai. Commands và learning gate giữ tại
[pilot protocol](pilot_protocol.md); [diagnostic definitions](pilot_diagnostics.md).
