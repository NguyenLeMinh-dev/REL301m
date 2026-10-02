# MASAC baseline audit — 2026-10-02

**Engineering PASS. Learning PENDING. Pilot chưa chạy**, theo lựa chọn người dùng:
chỉ audit/tests và chuẩn bị lệnh pilot. Không chạy lại smoke 50k hoặc thay đổi
nominal training budget; không triển khai GRU/delay/dropout/prediction.

## Validation thực tế

| Check | Result |
| --- | --- |
| Full Phase 1–3 suite | **76 passed, 0 failed, 0 skipped**, 50.72s |
| Live integration | **1,000 steps / 745 updates**, 5 full horizon-200 training episodes |
| Phase 0 venv | System/simulator PASS, PyTorch vẫn không cài |
| Phase 3 venv | System/simulator PASS, CUDA available=true |
| Device | RTX 4060 Ti, driver 610.43.02, torch 2.7.1+cu128, runtime 12.8 |
| Dimensions | actor_obs 66 mỗi agent; actions 7+7=14; critic state 119, critic input 133 |
| Parameters | actor 86,542 mỗi net; Q/target 100,353 mỗi net; total 574,496 + 2 log-alpha |
| Common eval RNG | Same initial-state hashes across random/policy, repeated eval và training seeds |
| Legacy checkpoint | Old smoke checkpoint load/evaluate PASS; first two seed-20000 trajectories/metrics unchanged |
| Numeric/outputs | Losses finite, gradients nonzero; CSV/TensorBoard/diagnostics/checkpoints PASS |

Test evidence: `experiments/phase3/baseline_audit/pytest.xml` và
`logs/pytest_baseline_audit.log`. Integration artifacts lưu tại
`experiments/phase3/baseline_audit/integration_1000/`.

## Những thay đổi đã hoàn thành

- Giữ một implementation trong `algorithms`; `agents` cung cấp facades tương thích.
- Tách `utils/seed.py`, `utils/logger.py`, `utils/checkpoint.py` từ behavior hiện có.
- Thêm `evaluation/metrics.py`: ever/final-success, first-success seconds, initial-state hashes.
- Evaluation sequence mới cố định **seed 20000** cho mọi training seed/method;
  restore dedicated environment RNG snapshot trước mỗi eval, không reseed training RNG.
- Random reference và final eval có cùng initialization sequence và episode count.
- Periodic eval 10 episodes; final pilot **50**, final preset **100**.
- Best selection dùng cùng first 10 episodes; learning gate report final checkpoint.
- Run metadata ghi software versions, evaluation protocol, deterministic flag và
  communication cost **N/A**. Mỗi run có `diagnostics.json` với gradient minima,
  bounds, timeout counts, alpha và evaluation hashes.
- Checkpoint thêm top-level `global_step`, vẫn load old format; không claim exact resume.
- Direct dependency pins trong `pyproject.toml` khóa đúng versions hiện có; không
  upgrade/downgrade packages. `configs/requirements/phase3.txt` chỉ định CUDA wheel index.
- Added tests cho protocol, success timing/semantics, utils, critic storage isolation,
  old checkpoint compatibility và presets. Existing Phase 1/2 tests giữ nguyên.

Đã đối chiếu hashes của **16 protected files** và AST của MASAC
constructor/target/loss/update/Polyak. Phase 1/2 factory/wrapper/contracts, networks,
replay buffer và `configs/algo/masac.yaml` không đổi. Old smoke report/checkpoints
được giữ nguyên; số liệu historical smoke không bị thay bằng evaluation protocol mới.

## Việc người dùng chạy tiếp

[Protocol và lệnh pilot](pilot_protocol.md) đã sẵn sàng: **300k × seeds 0/1/2**,
sequential, one env/run, same hyperparameters và common evaluation set.
Learning gate vẫn pending đến khi có pilot curves/results; chưa cho phép kết luận
MASAC giải được task. Nominal SR ≥50% là project target, không phải literature standard.

Checkpoint không lưu replay/RNG/environment state, nên không exact-resume interrupted
training. Seed và pinned stack không đảm bảo bitwise equality giữa platforms/releases;
metadata snapshots và hashes giúp nhận diện run đã thực hiện.

## Follow-up sau khi người dùng khởi chạy pilot

Đây là historical 76-test audit snapshot. Run seed 0 sau đó dừng ở step 18,713;
chưa có completed 300k × 3-seed result. Xem [97-test instrumentation follow-up](pilot_diagnostics_report.md)
và [diagnostic definitions](pilot_diagnostics.md). Lỗi controller chưa được chứng minh đã sửa.
