# REL301m — Cooperative Two-Arm Manipulation

Hai robot Panda phối hợp nâng pot trong robosuite `TwoArmLift`. Repo gồm simulator contract (Phase 0–2) và baseline MASAC (Phase 3), độc lập với source robosuite.

**Trạng thái:** simulator và multi-agent interface đã được kiểm thử. Code MASAC/BC đã có; chưa đạt learning PASS. BC-init pilot seed 0 dừng ở 138,757 steps do lỗi collision MuJoCo; chưa hoàn thành pilot 300k × 3 seeds. Xem [chẩn đoán](docs/bc_pilot_seed0_diagnosis.md).

## 1. Chuẩn bị

Dùng Linux và Python 3.10. Tất cả lệnh dưới đây chạy tại root của repo:

```bash
git clone https://github.com/NguyenLeMinh-dev/REL301m.git
cd REL301m
python3.10 -I -m venv .venv-phase0
.venv-phase0/bin/python -I -m pip install -e '.[dev]'
.venv-phase0/bin/python -I -m pip check
```

Stack được pin trong `pyproject.toml`: robosuite 1.5.2, MuJoCo 3.9.0, NumPy 1.26.4. Phase 0–2 không cần PyTorch. `-I` loại Python path kế thừa từ ROS và user site-packages; dùng trực tiếp Python trong venv nên không cần activate.

## 2. Phase 0 — System Freeze

Kiểm tra environment sạch, reset/step simulator và lưu thông tin máy, versions, Git SHA/status, `pip freeze`. GPU/driver được ghi qua `nvidia-smi`; PyTorch/CUDA chỉ được kiểm tra nếu đã cài.

```bash
nvidia-smi
.venv-phase0/bin/python -I scripts/check_install.py \
  --output-dir experiments/phase0 \
  --report experiments/phase0/reproducibility.md
```

Đọc `RESULT PASS` và các file trong `experiments/phase0/`. Báo cáo workstation ban đầu: [reproducibility](docs/reproducibility.md). Lỗi GPU trên máy khác cần được đọc theo điều kiện workstation của Phase 0, không suy ra simulator đã hỏng.

## 3. Phase 1 — Environment Contract

Cấu hình: hai Panda `opposed`, controller `BASIC`, state observations, shaped reward, 20 Hz, horizon 200, không camera/renderer.

```bash
.venv-phase0/bin/python -I scripts/inspect_env.py
.venv-phase0/bin/python -I scripts/random_rollout.py --episodes 3 --seed 0
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-phase0/bin/python -I -m pytest -q \
  tests/test_env_smoke.py tests/test_env_contract.py
```

`inspect_env.py` lấy observation keys, shape/dtype, action bounds/layout và observables từ live environment. Kết quả nằm trong `artifacts/*.json` và [environment specification](docs/environment_spec.md); random rollout lưu trong `experiments/phase1/`.

Success dùng kiểm tra task của robosuite: pot vượt mặt bàn 0.10 m. **Không dùng return threshold để định nghĩa success.** Episode vẫn chạy tới horizon; logger tách return, ever success, first success step, final success và episode length.

## 4. Phase 2 — Multi-Agent Wrapper

| Input/output | Agent 0 | Agent 1 |
| --- | ---: | ---: |
| Local observation | 53 | 53 |
| Shared object | 13 | 13 |
| Actor observation | 66 | 66 |
| Action | 7 | 7 |

Centralized `critic_state` có 119 phần tử; joint action có 14 theo thứ tự robot 0 rồi robot 1. Actors chỉ nhận proprioception/relative-handle state của chính mình và shared object; không nhận raw `object-state` chứa thông tin cả hai robot.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-phase0/bin/python -I -m pytest -q \
  tests/test_env_smoke.py tests/test_env_contract.py \
  tests/test_observation_split.py tests/test_action_split.py
```

API trong [multi_agent_wrapper.py](src/rel301m/envs/multi_agent_wrapper.py): `reset()`, `step(agent_0_action, agent_1_action)`, `close()`. Dimensions/bounds được xác minh từ metadata; reward/done/info giữ semantics robosuite. Chi tiết: [multi-agent contract](docs/multi_agent_contract.md).

## 5. Phase 3 — Baseline MASAC

Hai decentralized Gaussian actors, twin centralized critics và target critics, automatic entropy tuning. Tạo venv riêng; không thay stack Phase 0.

```bash
python3.10 -I -m venv .venv-phase3
.venv-phase3/bin/python -I -m pip install -r configs/requirements/phase3.txt
.venv-phase3/bin/python -I -m pip check
.venv-phase3/bin/python -I -c "import torch; print(torch.__version__, torch.cuda.is_available())"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-phase3/bin/python -I -m pytest -q tests
```

Requirements dùng PyTorch 2.7.1 CUDA 12.8 wheel và TensorBoard. Không cần cài system CUDA toolkit chỉ để dùng wheel; NVIDIA driver phải tương thích. `--device cpu` cho phép chạy bằng CPU.

Chạy smoke 50k steps trong một thư mục mới:

```bash
.venv-phase3/bin/python -I -u -m rel301m.training.train \
  --config configs/experiment/smoke.yaml \
  --run-dir experiments/phase3/smoke_my_run
```

CLI có progress, steps/s và ETA; evaluation/save có phase riêng. Known optional warnings của Panda/BASIC được lọc, errors vẫn hiện. Dùng `--show-all-warnings` để xem đầy đủ. Chờ process thoát thành công và có `final.pt`/`summary.json` mới coi run hoàn tất.

Evaluate checkpoint và xem TensorBoard:

```bash
.venv-phase3/bin/python -I -m rel301m.evaluation.evaluate \
  --checkpoint experiments/phase3/smoke_my_run/final.pt --episodes 10
.venv-phase3/bin/tensorboard --logdir experiments/phase3
```

Không mặc định tăng compute nếu success không lên. Kiểm tra reward, action scaling, done mask, critic targets, entropy và gradients trước. Checkpoint hiện không lưu replay/environment state để resume chính xác.

Các workflow tiếp theo được chạy thủ công theo tài liệu:

- [Baseline MASAC và hyperparameters](docs/baseline_masac.md).
- [Pilot 300k × 3 seeds](docs/pilot_protocol.md), [diagnostics](docs/pilot_diagnostics.md).
- [Thu scripted demos → BC → actor-only initialization](docs/bc_warm_start.md), [kết quả BC](docs/bc_pipeline_report.md).
- [So sánh BC-init và scratch trên cùng seeds](docs/bc_vs_scratch_protocol.md).
- [Xem hai robot trực tiếp hoặc xuất MP4](docs/visualization.md).

BC là workflow tùy chọn. Demos/checkpoints không được đóng gói trong Git; cần tạo chúng theo hướng dẫn trước khi chạy BC pilot. Predictor/uncertainty/adaptive communication chưa được triển khai thành phương pháp đã kiểm chứng.

## Files và kết quả

| Folder | Nội dung |
| --- | --- |
| `src/rel301m/` | Environment, MASAC, training/evaluation, BC, utilities |
| `configs/` | Environment/algo/experiment configs và requirements |
| `scripts/`, `tests/` | Inspection, analysis và regression tests |
| `artifacts/` | Observation/action/environment contract JSON |
| `docs/` | Specifications, protocols và báo cáo đo thực tế |
| `experiments/`, `data/`, `logs/` | Outputs local, không upload lên Git |

Mỗi training run lưu config, Git SHA, dependency freeze, metrics và checkpoint. Không xóa những file này khi đổi run; dùng `--run-dir` mới. `.gitignore` loại venv, caches, logs, datasets, checkpoints, video và secrets local khỏi repository.
