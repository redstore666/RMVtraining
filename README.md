# RMVtraining 视觉组大作业

RoboMaster 自瞄视觉链路：相机取流 → 装甲板检测 → PnP 位姿解算 → EKF 跟踪预测，
结果写回 64 字节通信结构体。题目只覆盖装甲板自瞄，串口、Web Debug 和双线程
框架由教师提供，其余在 `student/` 实现。

当前进度：取流、标定（内参+外参，车上相机 2026-10-25 配置）、检测、PnP、EKF
跟踪与预测均已实现，上车测试（精简）版。回归/单元测试套件在 `test` 分支。

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
├── tests/               现场调试工具（test_camera_live / test_camera_system）
│                          与迷你俯视图网页组件（web_system / SystemView）
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

## 测试工具（精简版：上车现场用）

| 工具 | 用途 |
| --- | --- |
| `rmv_training 串口 波特率 camera.yaml 8080` | 上车主程序；迷你俯视图网页自动开在 **8081** |
| `test_camera_live [--red\|--blue] [--no-display]` | 实时相机全链路：取流+检测+PnP+显示；状态栏显示相机固件帧率 fps_cam 与实际帧率 fps_nodisp |
| `test_camera_system [--red\|--blue] [--video v.avi] [--id N]` | 系统视图网页工作台（遥测 + 迷你俯视图，8081）；`--video` 可离线回放 |
| `calib_capture` / `calibrate_intrinsics` | 标定采集 / 内参解算 |

> 精简说明：本分支为上车测试版——合成回归/单元测试套件、离线分析工具与
> 过程文档保留在 `test` 分支（含 docs/任务进度备份.md 全量记录），此处不携带。

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

- [模型说明](models/README.md)（yolov5/yolov5_512/yolo11，解码格式，A/B 记录）
- [环境依赖](环境依赖.md)
- [通信结构体](通信结构体.md)（64 字节帧，字段不得改）
- [云台外参](云台外参.md)、[轨迹预测说明](轨迹预测说明.md)
- [任务书](docs/大作业任务书.md)、[验收与评分](docs/验收与评分.md)
- 过程文档（实验报告、任务进度备份、EKF 设计说明等）在 `test` 分支，本精简版不携带

## 阶段规划

**先走完大流程，再做精细优化。** 检测模型已满足验收，模型重训不是当前阶段的
阻塞项。顺序：相机联机实跑 → 外参标定 → 全链路实机验收（EKF 已实现）；
之后再进精细优化段（模型针对性重训、INT8 量化、比赛机 GPU、远近双模型自适应，
每项按检出率 + 角点偏差 + 重投影三项指标 A/B 后决定是否合入）。

## 已知问题

1. 相机→云台外参已填车上相机值（2026-10-25：R=90° 置换 + t=(70.82, 0, 84.31) mm，
   来源见 config/camera.yaml 注释）；支架装配偏角如实测存在需另行手眼标定。
2. EKF 的车体 yaw 只能确定到 `2*pi/N`（N 块板几何全同，属模型固有对称性）。
3. EKF 是单半径、等高模型；前哨站交替高度/半径的目标需按槽位扩展状态。
4. 推理输入默认 `640x480`（与画面同比例，省掉 25% 无效卷积），`640` 与 `512` 模型都保留。
