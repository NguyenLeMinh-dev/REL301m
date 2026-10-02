# Xem hai robot và xuất video

Entry point `rel301m.evaluation.visualize` chạy deterministic actions từ một
checkpoint MASAC đã lưu. Mỗi episode vẫn dùng state observations 66/agent,
critic state 119 và action 7 + 7. Camera chỉ phục vụ hiển thị, không đưa image
vào actors và không thay đổi config training hay source robosuite.

Chạy các lệnh bên dưới từ research root:

```bash
cd /home/minh/Documents/REL301m/REL301m-research
```

## Xem trực tiếp

Lệnh này mở cửa sổ MuJoCo trên desktop, chạy một episode 200 steps ở control
frequency 20 Hz. Có thể xoay/zoom camera bằng chuột; đóng cửa sổ để dừng sớm.

```bash
.venv-phase3/bin/python -I -m rel301m.evaluation.visualize \
  --checkpoint experiments/phase3/smoke_seed0/final.pt \
  --mode viewer --episodes 1 --seed 20000
```

Viewer cần một desktop display. Mỗi episode tạo viewer mới sau `reset()` vì
robosuite thay model/data khi hard reset. Nếu đóng sớm, `summary.json` ghi
`partial: true` và không tính episode đó vào completed metrics.

## Xuất MP4

```bash
.venv-phase3/bin/python -I -m rel301m.evaluation.visualize \
  --checkpoint experiments/phase3/smoke_seed0/final.pt \
  --mode video --episodes 3 --seed 20000
```

Mặc định mỗi lần chạy tạo folder timestamp mới bên cạnh checkpoint, chứa
`episode_000.mp4`, các PNG start/middle/end, ffmpeg log và `summary.json`.
Có thể dùng `--output-dir PATH` với một folder chưa tồn tại. Với checkpoint
khác, thay đường dẫn `--checkpoint`; config môi trường được đọc từ checkpoint
và dimensions/action bounds được đối chiếu với live environment.

Video dùng EGL offscreen, không cần cửa sổ desktop. Workstation hiện tại đã có
`ffmpeg` và OpenCV trong `.venv-phase3`, không cần cài thêm dependencies.
Độ phân giải scene mặc định 960×540, cộng footer 96 pixels, đầu ra H.264
960×636, 20 fps, 10 giây cho horizon 200. Không chạy thêm physics steps để
render. Overlay hiển thị step, thời gian simulation, reward, return,
first-success và current/ever success.

## Camera và cách đọc kết quả

- `--camera overview`: mặc định, nhìn trọn hai Panda và pot.
- `--camera agentview`: cận cảnh grippers và pot.
- Các camera native khác gồm `frontview`, `birdview`, `sideview`,
  `robot0_robotview`, `robot1_robotview`, `robot0_eye_in_hand`,
  `robot1_eye_in_hand`.
- `--device cpu` hoặc `--device cuda`: chọn device chạy actors. Rendering vẫn
  là một phần riêng của MuJoCo.

Success lấy trực tiếp từ task, không suy ra từ return hay việc một tay nhấc
handle. `current_success` ở frame cuối tương ứng `final_success`; episode
tiếp tục tới horizon ngay cả khi đã success.

## Kiểm tra thực tế

Checkpoint `experiments/phase3/smoke_seed0/final.pt`, seed 20000:

- Video: `experiments/phase3/smoke_seed0/video_preview_seed20000/episode_000.mp4`.
- Viewer: `experiments/phase3/smoke_seed0/live_viewer_check/summary.json`.
- Cả hai chạy đủ 200 steps, return **27.33457810298125**,
  `ever_success=false`, `final_success=false`.
- Initial state SHA-256 giống nhau:
  `49ce12456655966572dabd5ec8859bf9a3613a06d8aad04b054660adfdd51c75`.
- MP4 được kiểm tra metadata, decode và xem PNG ở đầu/giữa/cuối episode.

Đây là hình ảnh của smoke policy hiện có; một đoạn video chưa thành công
không thay thế evaluation Success Rate qua nhiều episodes/seeds.
