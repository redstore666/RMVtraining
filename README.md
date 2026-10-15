# RMVtraining 视觉组大作业

RoboMaster 自瞄视觉链路：相机取流 → 装甲板检测 → PnP 位姿解算 → EKF 跟踪预测，
结果写回 64 字节通信结构体。题目只覆盖装甲板自瞄，串口、Web Debug 和双线程
框架由教师提供，其余在 `student/` 实现。

当前进度：取流、标定、检测、PnP 已完成并实拍验收；EKF 尚未开始。

## 目录结构

```
├── main.cpp              双线程框架（读线程/操作线程），框架代码未动
├── student/              本作业的全部实现
│   ├── StudentTasks.cpp    get_pic / armor_detect / armor_solve / ekf_predict
│   └── ArmorDetectorNet.*  OpenVINO 推理 + YOLO 关键点解码（任务 2）
├── armor/               教师提供的 EKF 与弹道模型（任务 4 用，暂未接线）
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

Web Debug 在 `http://127.0.0.1:8080`。外参未标定前 `camera.yaml` 的
`calibrated` 保持 0，程序拒绝占位外参，属预期行为。

## 测试工具

| 工具 | 用途 |
| --- | --- |
| `test_camera_live [--red\|--blue] [--no-display]` | 实时相机全链路：取流+检测+PnP+显示；状态栏显示相机固件帧率 fps_cam 与实际帧率 fps_nodisp |
| `test_detect 图 --red [--save out.png]` | 单图检测，输出角点叠加图 |
| `test_batch 目录 --red` | 批量检出率评估（调参/回归用） |
| `test_solve 图 --red` | 检测→PnP 全链路，打印距离/角度/重投影误差 |
| `test_camera` | 相机取流自检（ctest camera_grab） |
| `calib_capture` / `calibrate_intrinsics` | 标定采集 / 内参解算 |

常用环境变量（检测器）：`RMV_DET_MODEL=yolo11` 切备选模型；
`RMV_DET_SIZE=512` 切小输入提速（远处角点抖动增大，默认 640）；
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

## 阶段规划

**先走完大流程，再做精细优化。** 检测模型已满足验收，模型重训不是当前阶段的
阻塞项。顺序：相机联机实跑 → 外参标定 → EKF 实现 → 全链路实机验收；
之后再进精细优化段（模型针对性重训、INT8 量化、比赛机 GPU、远近双模型自适应，
每项按检出率 + 角点偏差 + 重投影三项指标 A/B 后决定是否合入）。

## 已知问题

1. EKF（任务 4）未实现，`ekf_predict` 为空桩；旋转车体多目标跟踪方案已有草稿。
2. 相机→云台外参待标定填写。
3. 512 输入在远处（约 2m）角点抖动明显，默认值保持 640。
4. 实拍素材目前只有哨兵单目标，多目标场景待补素材回归。
