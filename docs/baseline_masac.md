# Phase 3 — Baseline MASAC

Baseline dùng continuous SAC cho shared reward: hai actor độc lập, một cặp
centralized Q và target Q. Actor chỉ dùng `actor_obs` của chính agent; coordinator
mới đọc `critic_state`. Không có predictor, delay/dropout hoặc communication policy.
`GRU128` chỉ là cấu hình dự kiến (`enabled: false`), chưa khởi tạo network.

## Kiến trúc đo thực tế

| Network | Input | Hidden layers | Output | Parameters |
| --- | ---: | --- | --- | ---: |
| Actor 0 | 66 | 256, 256, ReLU | mean(7), log_std(7) | 86,542 |
| Actor 1 | 66 | 256, 256, ReLU | mean(7), log_std(7) | 86,542 |
| Q1 / Q2 / target Q1 / target Q2, mỗi net | 119 + 14 | 256, 256, ReLU | scalar | 100,353 |

Tổng network parameters: **574,496**; thêm hai scalar `log_alpha` trainable.
Target critics không nhận gradient. Actor không nhận raw observation dictionary,
critic state, hay private state của teammate. Action dimensions và bounds lấy từ
wrapper/live Phase 1 metadata. Actor tạo Normal distribution với reparameterized
sample, áp dụng tanh rồi affine transform theo action bounds; log probability có
cả tanh Jacobian và affine Jacobian. `log_std` giới hạn [-20, 2]. Evaluation dùng
tanh(mean), training dùng stochastic sample.

## Update và time limits

Đây là shared-Q cooperative MASAC implementation theo công thức
[SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html), với cách tách actor
và critic theo tinh thần [CTDE](https://arxiv.org/abs/1706.02275). Không dùng code hay
claims performance của MADDPG để gọi đây là MASAC đã được benchmark trong paper.

Với `a' = [a0', a1']`, critic target là:

```text
terminal = done * (1 - timeout)  # default bootstrap_time_limits=true
entropy_term = alpha0 * log_pi0(a0'|o0') + alpha1 * log_pi1(a1'|o1')
y = r + gamma * (1 - terminal) * (min(target_Q1(s',a'), target_Q2(s',a')) - entropy_term)
Q_loss = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
actor_i_loss = mean(alpha_i * log_pi_i(ai|oi) - min(Q1(s,a), Q2(s,a)))
alpha_i_loss = -mean(log_alpha_i * detach(log_pi_i + target_entropy_i))
target_Q = (1 - tau) * target_Q + tau * Q
```

Mỗi actor có temperature riêng, initial alpha=0.2 và target entropy lấy từ action
size, hiện là **-7 mỗi agent**. Critic target dùng tổng entropy hai policy. Khi
update actor i, teammate action được sample dưới `no_grad`; critics đóng băng
parameters nhưng vẫn truyền gradient từ Q về own action. Hai actor update tuần tự;
actor 1 dùng policy actor 0 đã update. Không truyền gradient qua teammate hay target.

Upstream `done` của task hiện tại chỉ xảy ra ở horizon 200. Đây là time limit của
continuing manipulation task, nên default target vẫn bootstrap từ observation
cuối trước reset. Replay lưu `done` nguyên gốc và thêm `timeout`, không thay đổi
wrapper reward/done/info. `bootstrap_time_limits: false` cho finite-horizon target
mask; lựa chọn này phải được giữ nhất quán khi so sánh experiment.

Replay float32 nằm trong CPU RAM, ring buffer uniform sampling. Fields theo yêu cầu:

```text
o0(66), o1(66), s(119), a0(7), a1(7), r(1),
next_o0(66), next_o1(66), next_s(119), done(1), timeout(1)
```

Capacity 300,000 chiếm **622,800,000 bytes (~594 MiB)** cho arrays; chỉ sampled
batch được chuyển lên GPU. Không có reward rescaling hoặc observation normalization
ngoài environment contract. lr actor/Q/alpha=3e-4, gamma=0.99, tau=0.005,
batch=256, warmup=10,000, một gradient update mỗi environment step sau warmup.
Warmup steps 1..10,000 dùng random actions; update đầu tiên ở step 10,000 nếu đủ batch.
50,000 steps tạo 40,001 updates.

## Environment và commands

Chạy các lệnh sau **từ research root**. `.venv-phase0` được giữ nguyên; baseline dùng
`.venv-phase3` riêng. Python `-I` và pytest plugin isolation tránh ROS path/plugins.
CUDA wheel được pin `torch==2.7.1+cu128`, không yêu cầu cài CUDA toolkit hệ thống.
Wheel cho Python 3.10 có trong [official CUDA 12.8 index](https://download.pytorch.org/whl/cu128/torch/).
Direct dependencies hiện pin chính xác robosuite 1.5.2, MuJoCo 3.9.0, NumPy 1.26.4,
PyYAML 6.0.3, Torch 2.7.1+cu128, TensorBoard 2.20.0 và pytest 9.1.1 trong
`pyproject.toml`; `configs/requirements/phase3.txt` cung cấp official CUDA index.
Không thay version đã chạy. Pin này nhằm đóng băng baseline chạy được trên workstation; không phải claim rằng
version này nhanh hoặc tốt hơn version khác. TensorBoard pin 2.20.0.

```bash
cd /home/minh/Documents/REL301m/REL301m-research
python3.10 -I -m venv .venv-phase3
.venv-phase3/bin/python -I -m pip install -r configs/requirements/phase3.txt
.venv-phase3/bin/python -I -m pip check
.venv-phase3/bin/python -I -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'

# Full regression, math/gradient/save-load tests và live 1k-step integration.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-phase3/bin/python -I -m pytest -q tests

# 50k steps, seed 0. Mặc định tạo folder run riêng theo timestamp.
.venv-phase3/bin/python -I -m rel301m.training.train --config configs/experiment/smoke.yaml

# Đo success độc lập bằng deterministic decentralized actors từ checkpoint.
.venv-phase3/bin/python -I -m rel301m.evaluation.evaluate \
  --checkpoint experiments/phase3/smoke_seed0/final.pt --episodes 10 --seed 20000

# Mở dashboard CSV/TensorBoard tương ứng với run.
.venv-phase3/bin/tensorboard --logdir experiments/phase3
```

Sau khi activate venv, có thể dùng `python -I -m ...` với cùng arguments. `--device
cuda` sẽ fail rõ nếu CUDA unavailable; `auto` chọn CUDA nếu khả dụng, hoặc CPU và
ghi device thực tế. `--steps`, `--seed` và `--device` override được ghi trong YAML.
`--run-dir` phải là folder mới, tránh overwrite run cũ.

Pilot preset: 300k steps, final eval 50 episodes; final preset: 700k steps, final eval
100 episodes. Periodic eval vẫn 10 episodes mỗi 10k steps; không nhân evaluation
overhead lên 50/100 tại mọi checkpoint. Chạy seed 0/1/2 bằng `--seed` cho
mỗi config tương ứng. Audit chỉ chạy tests/live 1k integration theo yêu cầu; pilot chưa được chạy. Các preset
là budget configurations, không phải kết quả đã hoàn thành.

```bash
for seed in 0 1 2; do
  .venv-phase3/bin/python -I -m rel301m.training.train --config configs/experiment/pilot.yaml --seed "$seed"
done
# Khi quyết định thực hiện final: đổi pilot.yaml thành final.yaml.
```

## Run artifacts và metrics

Mỗi run lưu `resolved_config.yaml`, `env.yaml`, `metadata.json` (git SHA/status,
versions, CUDA/GPU, seed, dimensions, bounds trong checkpoint, parameter counts,
source/config/contract SHA256), `git_status.txt`, `requirements.freeze.txt`,
`pip_freeze_stderr.txt`, `episodes.csv`, `losses.csv`, `evaluation.csv`, TensorBoard,
random-policy reference, từng evaluation episode, `diagnostics.json` và periodic/best/final checkpoints.
Git chưa có HEAD thì ghi đúng `uncommitted (no HEAD yet)`; source hashes vẫn xác
định code đã chạy. Không tự tạo commit để thay nhãn này.

`episodes.csv` có return, ever_success, first_success_step (null nếu chưa thành công),
final_success, length và rolling **ever-success rate** của tối đa 100 completed
training episodes. Episode cuối chưa đủ horizon được ghi `partial=true`, loại khỏi
rolling rate. `evaluation.csv` dùng dedicated environment với deterministic actors;
`success_rate` = fraction ever_success, `final_success_rate` được ghi riêng. Success
luôn dùng `_check_success()`, không suy ra từ return. Baseline random chạy trước
training. Evaluation mới dùng **eval_seed=20000 chung** cho mọi training seed và
method. Trước mỗi eval, khôi phục bit-generator state của dedicated eval environment
về snapshot sau constructor; giữ nguyên generator object vì native samplers giữ
reference tới nó. Không reseed Torch/global RNG của training. Random reference và
final policy dùng cùng số episodes và cùng initialization sequence.

Mỗi episode ghi `initial_state_sha256` từ critic state ngay sau reset; summary có
`initialization_sequence_sha256`. Tests kiểm tra random/policy, eval lặp lại và
training seeds khác nhau có cùng initializations. Đây là initialization sequence
khác training, nhưng được dùng để monitor/select checkpoints; không gọi nó là
independent test set chưa dùng để chọn model. CLI có thể evaluate seed mới (ví dụ
30000) dành riêng cho final reporting của mọi method. Historical 50k smoke dùng
protocol cũ; [smoke report](phase3_smoke_report.md) giữ đúng số đo cũ.

`first_success_time_s = first_success_step / control_freq`, hiện tại chia 20;
không success thì null/CSV trống. Metadata ghi communication cost `N/A`; không ghi
communication-rate hoặc RD thành 0 khi communication experiment chưa tồn tại.

`losses.csv` ghi Q/actor/alpha losses, alpha/entropy, Q/target means và gradient norms.
`summary.json` chỉ có `completed` sau khi đủ budget và lưu final checkpoint;
exception được ghi `failure.json`. `best.pt` chọn lexicographic eval ever-success
rate rồi mean_return trên cùng first 10 episodes tại mọi checkpoint (final 50/100
không thay denominator selection). `final.pt` vẫn là model cuối đủ budget;
learning gate đánh giá final checkpoint, tránh chọn kết quả post hoc. Checkpoints chứa tất
cả networks, optimizers, alpha và update count để kiểm tra/load/evaluate; chưa có
CLI resume training vì replay và RNG/environment state chưa được checkpoint.

PyTorch peak allocated/reserved trong summary không bao gồm toàn bộ CUDA context,
driver hoặc desktop GPU usage. Số đo `nvidia-smi` của process được ghi riêng khi
benchmark. Budget dự kiến phải dựa trên simulation/update/evaluation timings thực tế;
1k integration dùng warmup/batch nhỏ để kiểm tra update, không thay smoke config.

## Kiểm chứng và diễn giải

Tests kiểm tra tensor shapes, parameter counts, affine tanh density/bounds, numeric
stability, replay ordering/copies/atomic rejection, actor/critic/target gradient
isolation, twin-Q min/entropy/time-limit target, chính xác Polyak update,
model+optimizer round-trip, CUDA batch và full 1k-step training qua simulator thật.
46 tests Phase 1/2 tiếp tục chạy; Phase 0 system check chạy riêng trên cả hai venv.

Nếu smoke không có success, kiểm tra lần lượt reward từ simulator, action bounds /
OSC_POSE + gripper scale, raw done/time-limit mask, Bellman target, alpha/entropy và
gradient norms trước khi tăng compute. Return cao hơn chỉ là shaped reward signal,
không thay thế task success. `RD(d)` và communication-rate chưa log vì chưa có
communication perturbation; không gán RD=0 hoặc chia cho SR0=0. Predictor,
uncertainty và adaptive communication thuộc experiment sau baseline, chưa triển khai.

## Kết quả smoke đã chạy

Xem [measured smoke report](phase3_smoke_report.md): 65 tests PASS, smoke 50k hoàn thành;
eval cuối ever-success 1/10, held-out seed 20000 ever-success 0/10.

## Audit completion và public API

`rel301m.agents/{networks,replay_buffer,masac}.py` là public facades trỏ tới classes
trong `rel301m.algorithms`; chỉ có một implementation. Imports cũ vẫn hoạt động.
`evaluation/metrics.py` quản lý success/return/first-success timing;
`utils/seed.py`, `utils/logger.py`, `utils/checkpoint.py` tách utility hiện có để
reuse. Checkpoint mới thêm top-level `global_step`; checkpoint smoke cũ vẫn load
được. Replay/RNG/environment state chưa được lưu để exact resume.

Giữ nguyên `__init__`, critic target, actor loss, update và Polyak math của MASAC;
không thay architectures, hyperparameters hoặc Phase 1/2 environment semantics.
Xem [pilot protocol](pilot_protocol.md) để tự chạy 300k × seeds 0/1/2 và đánh giá
engineering gate riêng với learning gate.

Python/NumPy/Torch coordinator được seed tại đầu run; robosuite/replay có generator
riêng. Metadata ghi deterministic-algorithms flag thực tế. Seed và package pin
không đảm bảo bitwise equality giữa hardware/releases; xem
[PyTorch 2.7 reproducibility documentation](https://docs.pytorch.org/docs/2.7/notes/randomness.html).
