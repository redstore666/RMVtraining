# RMVtraining 视觉组大作业

RoboMaster 自瞄视觉链路：相机取流 → 装甲板检测 → PnP 位姿解算 → EKF 跟踪预测，
结果写回 64 字节通信结构体。题目只覆盖装甲板自瞄，串口、Web Debug 和双线程
框架由教师提供，其余在 `student/` 实现。

当前进度：取流、标定（内参）、检测、PnP、EKF 跟踪与预测均已实现；
`ctest -E camera_grab` 8/8 通过。剩余实机项：相机→云台外参标定、全链路联机验收。

## 目录结构

```
├── main.cpp              双线程框架（读线程/操作线程），框架代码未动
├── student/              本作业的全部实现
│   ├── StudentTasks.cpp    get_pic / armor_detect / armor_solve / ekf_predict
│   ├── ArmorDetectorNet.*  OpenVINO 推理 + YOLO 关键点解码（任务 2）
│   ├── RotatingArmorModel.* 旋转车体运动模型（任务 4 的公式真源）
│   ├── RotatingArmorEkf.*   9 维 EKF（初始化契约/预测/更新/NIS）
│   ├── RotatingArmorTracker.* 观测关联/门限/状态机/瞄准板选择（任务 4）
│   └── AimGeometry.*        云台系↔世界系换算与极角定义
├── armor/               教师提供的 EKF 与弹道模型（仅作参考，未接入主流程）
├── serialPort/          串口读写（教师提供）
├── web_debug/           Web Debug 数值曲线（教师提供）
├── models/              离线模型与说明，见 models/README.md
├── tests/               调试与验收工具 + 实验报告，见 tests/README.md
├── tools/               标定相关工具（棋盘格 PDF、内参标定、IR 手术脚本）
├── config/camera.yaml   2 号机内参（MATLAB 标定，均值重投影误差 0.07px）
├── calib_data/          标定用图（21 张）
└── docs/                任务书与评分标准（教师提供）
```

## 构建

依赖见 [环境依赖.md](环境依赖.md)：OpenCV 4.5.4、Eigen3、MVS SDK、OpenVINO
运行时（pip 装）。注意本机 OpenCV 未编译 Inference Engine，不能读 OpenVINO IR，
检测走原生 OpenVINO。

```bash
cmake -S . -B build -DOpenVINO_DIR=<venv>/lib/python3.X/site-packages/openvino/cmake
cmake --build build -j
```

## 运行

```bash
./build/rmv_training /dev/ttyACM0 115200 config/camera.yaml 8080
```

Web Debug 在 `http://127.0.0.1:8080`。相机→云台外参尚未标定，`camera.yaml` 里
仍是占位单位阵；`armor_solve` 会按"相机光学系(右/下/前) → 云台系(前/左/上)"的固定
旋转重排坐标轴并在首帧打印一条警告，因此相对几何（中心/半径/角速度）可用，但绝对
瞄准角仍有系统偏差。标定后填入真实 `R/t` 即自动走 `R*p + t`（见 `docs/云台外参.md`）。

## 测试工具

| 工具 | 用途 |
| --- | --- |
| `test_camera_live [--red\|--blue] [--no-display]` | 实时相机全链路：取流+检测+PnP+显示；状态栏显示相机固件帧率 fps_cam 与实际帧率 fps_nodisp |
| `test_detect 图 --red [--save out.png]` | 单图检测，输出角点叠加图 |
| `test_batch 目录 --red` | 批量检出率评估（调参/回归用） |
| `test_solve 图 --red` | 检测→PnP 全链路，打印距离/角度/重投影误差 |
| `test_camera` | 相机取流自检（ctest camera_grab） |
| `test_ekf_viewer [--world] [--out DIR]` | EKF 俯视图（上帝视角）演示：红点车心 + 四块模型板 + 绿点实测板 + 瞄准射线；`--no-show --out DIR` 无显示环境导出 PNG |
| `test_camera_playback v.avi --red --ekf --topdown` | 视频回放 + 俯视图窗口（`--topdown-world` 固定原点、`--topdown-out DIR` 逐帧落盘） |
| `calib_capture` / `calibrate_intrinsics` | 标定采集 / 内参解算 |

常用环境变量（检测器）：`RMV_DET_MODEL=yolo11` 切备选模型；
`RMV_DET_SIZE` 切换推理输入（默认 `640x480`，与 4:3 画面同比例；`640` 回到正方形输入，`512` 再提速但远处角点抖动增大）；
`RMV_DET_DEVICE=GPU/AUTO` 切推理设备（编译失败自动回退 CPU）。

## 检测路线

神经网络四点检测：YOLO 关键点模型（sp_vision_25 的 yolov5，OpenVINO IR）直接
回归装甲板四角点，同时解出颜色、编号、大小板。实拍验收：红 64/64、蓝 141/141
检出率 100%，CPU 单帧 18~25ms，PnP 重投影误差 1.6~1.8px。

传统视觉路线（HSV 掩码 + 灯条配对 + 数字分类）的调试过程与结论见实验报告，
代码已随路线切换移除。

## 文档

- [实验报告（调试全过程记录）](./tests/README.md)
- [模型说明](models/README.md)（yolov5/yolov5_512/yolo11，解码格式，A/B 记录）
- [环境依赖](环境依赖.md)
- [通信结构体](通信结构体.md)（64 字节帧，字段不得改）
- [云台外参](云台外参.md)、[轨迹预测说明](轨迹预测说明.md)
- [任务书](docs/大作业任务书.md)、[验收与评分](docs/验收与评分.md)
- [EKF 设计说明](docs/EKF设计说明.md)、[实验报告](docs/实验报告.md)

## 阶段规划

**先走完大流程，再做精细优化。** 检测模型已满足验收，模型重训不是当前阶段的
阻塞项。顺序：相机联机实跑 → 外参标定 → 全链路实机验收（EKF 已实现）；
之后再进精细优化段（模型针对性重训、INT8 量化、比赛机 GPU、远近双模型自适应，
每项按检出率 + 角点偏差 + 重投影三项指标 A/B 后决定是否合入）。

## 已知问题

1. 相机→云台外参待标定填写（`config/camera.yaml` 的 `calibrated` 故意保持 0）；
   未标定时 EKF 的相对几何仍自洽，但绝对距离/半径会偏移。
2. EKF 的车体 yaw 只能确定到 `2*pi/N`（N 块板几何全同，属模型固有对称性），
   详见 [EKF 设计说明](docs/EKF设计说明.md)。
3. EKF 是单半径、等高模型；前哨站交替高度/半径的目标需按槽位扩展状态。
4. 推理输入默认 `640x480`（与画面同比例，省掉 25% 无效卷积），`640` 与 `512` 模型都保留；合成素材 `synth_small_1` 的绝对距离与文档预期不符（宽高比不自洽），属素材问题。
5. 实拍素材目前只有哨兵单目标，多目标场景待补素材回归。
