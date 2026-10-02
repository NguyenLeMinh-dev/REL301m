# BC-assisted MASAC: kiểm chứng ngắn ngày 2026-10-02

**Code/integration PASS; learning FAIL trong integration rút ngắn. Config 10k mặc định chưa được chạy.** Không chạy 10k/30k/300k trong lần triển khai này. Baseline configs, dependency pins, original BC checkpoint và simulator contracts được giữ nguyên.

## Những gì đã thêm

Variant riêng dùng train-demo prefill và retained demo replay, BC auxiliary loss, khởi tạo log_std=-3, cửa sổ fixed std, critic pretraining và actor/alpha freeze. Có paired deterministic/stochastic evaluation trước updates, validation drift, reward components và `best.pt` bao gồm checkpoint BC step 0. Details và lệnh chạy: [fine-tune protocol](bc_finetune.md).

## Tests

- Full suite: **145 passed**, 74.13 s; bao gồm các tests Phase 0–3 trước đó.
- Environment Phase 0 không PyTorch: **48 passed**, 14.00 s; contracts, observation/action split và reward decomposition.
- Tests mới xác minh train/validation/test separation, retained demos sau ring overwrite, BC loss tác động thật lên actor update, gradient isolation, critic-only Adam freeze, fixed alpha/std schedule, checkpoint load và stochastic evaluation RNG isolation.
- Reward decomposition được đối chiếu với upstream reward trên cả random full episode và scripted grasp/lift/success. Không đổi reward đưa vào replay.

Logs: `experiments/phase3/bc_finetune_audit_20261002/`.

## Integration 1k

Run cuối: `experiments/phase3/bc_finetune_verified_20261002/`.

Dùng cùng 40 demos và BC checkpoint trước đó. Chỉ 30 train episodes / 6,000 transitions dùng để learning; 5 validation episodes đo drift; 5 test episodes không dùng. Live dimensions vẫn là actor 66/agent, action 7/agent, critic state 119, joint action 14.

Để exercise đủ optimizer paths trong 1k budget, integration **override**:

| Parameter | Integration | Config 10k mặc định |
| --- | ---: | ---: |
| Critic pretrain updates | 25 | 1,000 |
| Actor freeze online updates | 50 | 1,000 |
| Fixed log_std total updates | 100 | 3,000 |
| Replay capacity | 12,000 | 300,000 |
| Evaluation interval | 500 | 1,000 |
| Checkpoint interval | 500 | 1,000 |
| Log interval | 50 | 100 |

Giữ lambda_BC=0.5, initial alpha=0.02 auto, batch=256, random warmup=0, demo fraction schedule 0.75→0.5, và paired evaluation seed=20,000 / 10 episodes.

Kết quả kỹ thuật: hoàn tất 1,000 environment steps / 5 full episodes, 1,025 critic updates gồm 25 offline, **950 actor updates**. NaN=0, Inf=0, bounds violations=0. Có Adam state cho Q/actors/alpha sau run; offline pretraining không thay actors/alpha hay optimizer state của chúng. Run mất 107.74 s, peak CUDA allocated ≈29 MiB; đây không phải benchmark tốc độ pilot dài.

## Kết quả learning

| Policy / step | Ever-success | Final-success | Mean return | Validation MSE agent 0 / 1 |
| --- | ---: | ---: | ---: | --- |
| Initial BC deterministic, step 0 | 7/10 | 2/10 | 30.848 | 0.000583 / 0.000953 |
| Initial BC stochastic, log_std=-3 | 3/10 | 1/10 | 24.672 | cùng actor mean |
| Fine-tune step 500 | 0/10 | 0/10 | 0.546 | 0.085457 / 0.051798 |
| Fine-tune step 1,000 | 0/10 | 0/10 | 0.627 | 0.063859 / 0.068065 |

Tất cả cùng initialization-sequence SHA-256:
`19841a3401962f77201c881bd1c5865b19aa9a104e7c42c02926d6ccedfd7721`.

Stochastic rollout dùng Torch RNG stream của seed 0, được fork/restore để không đổi training RNG. Không so trực tiếp tỷ lệ 3/10 này với probe 4/5 seed/RNG khác rồi kết luận policy đã cải thiện hoặc suy giảm.

**BC loss=0.5 kết hợp schedules rút ngắn chưa ngăn drift.** Tests PASS chứng minh pipeline chạy đúng các paths đã kiểm tra, không chứng minh policy học tốt. Đây không phải bằng chứng config đầy đủ với 1,000 pretrain + 1,000 freeze sẽ có cùng kết quả; config đó chưa được đánh giá. Không tăng budget chỉ vì optimizer chạy ổn.

`best.pt` giữ **step 0 / 7-of-10 ever-success**. `final.pt` vẫn lưu policy fine-tune suy giảm để audit; không nhầm final với best. Original BC checkpoint vẫn có SHA-256:
`8e88e30e4d9ec87a462596a46c7aa5fd1e83d7ac51b0bfcfd3f595c380273979`.

Machine-readable checks: `experiments/phase3/bc_finetune_audit_20261002/verification.json`; source hashes lúc train khớp với source đã kiểm chứng. Probe config/script và logs được giữ trong cùng audit folder.

## Collision: reproduced, chưa repaired

Script `scripts/reproduce_collision.py` nạp qpos/qvel từ failure của pilot seed 0 rồi gọi `mj_forward` trên model không đổi. Đã tái hiện:

```text
mj_narrowphase: collision function returned 10 contacts for geom pair (150, 153),
expected at most 8 from mj_maxContact
```

Hai geom là `gripper1_right_finger1_pad_collision` / `gripper1_right_finger2_pad_collision`, đều box. Đây là failure native, không phải optional warning. Theo [source MuJoCo 3.9.0](https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_collision_driver.c), box-box có giới hạn 8 contacts; không có XML `mj_maxContact=16` để sửa kiểm tra đó.

Evidence: `experiments/phase3/bc_finetune_audit_20261002/collision_reproduction.json`. `collision_fix_verified=false` được ghi trong metadata/diagnostics. Integration ngắn không bị collision không chứng minh lỗi đã hết. Chưa thay packages, physics hoặc collision masks; chưa đủ điều kiện chạy pilot dài.
