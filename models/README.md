# models/ 离线模型资源

## yolov5.xml / yolov5.bin（神经网络四点检测，主用，640 输入）

装甲板四点关键点检测模型，OpenVINO IR（version 11），供 `ArmorDetectorNet` 使用。
来源：同济大学 SuperPower 战队 25 赛季开源视觉框架
[redstore666/sp_vision_25](https://github.com/redstore666/sp_vision_25)
（standard3 上场默认模型，流程解析见同目录 `自瞄识别全流程解析.md`）。

| 项 | 值 |
| --- | --- |
| 输入 | 1x3x640x640 FP32，RGB 顺序，/255 归一化 |
| 预处理 | 等比缩放（scale = min(640/rows, 640/cols)），左上对齐放入 640x640 黑底画布（letterbox） |
| 输出 | 1x25200x22 FP32，每行一个候选 |

输出行布局（与 sp_vision_25 `tasks/auto_aim/yolos/yolov5.cpp` 解码严格一致）：

```
[0:8)   4 个关键点（letterbox 坐标，像素）：
        (0,1)=TL  (2,3)=BL  (4,5)=BR  (6,7)=TR
        拼成工程约定 TL,TR,BR,BL 的取列顺序为 (0,1),(6,7),(4,5),(2,3)
[8]     objectness（需 sigmoid，阈值 0.7）
[9:13)  颜色 one-hot 4 类 argmax：0=blue 1=red 2=extinguish 3=purple
        （注意：one-hot 顺序与 yolo11 的属性表枚举不同，源自
          sp_vision_25 Armor(color_id, num_id,...) 构造函数的映射）
[13:22) 编号 one-hot 9 类 argmax：
        0=sentry 1=one 2=two 3=three 4=four 5=five
        6=outpost 7=base 8=not_armor(丢弃)
```

后处理：score(sigmoid) ≥ 0.7 → 关键点 ÷scale 还原原图坐标 → NMS(conf 0.7 / IoU 0.3)。
语义映射到本工程：blue/red → `TeamColor`；sentry→7(7guard)、one..five→1..5、
outpost→6、base→8；extinguish/purple（熄灭/紫）直接丢弃（不可击打状态）。
大小板：输出不含大小类，用四角点几何 ratio（外缘宽/条高 ≈2.37 小 / 4.04 大，
与传统路线 classifyArmorSize 同标准）判定。

### 部署链路验证记录（2026-10-14）

- 推理引擎：OpenVINO 运行时（pip openvino 2026.4.1，CPU LATENCY 模式）。
  系统 OpenCV 4.5.4 未编译 Inference Engine，不能读 IR；IR→ONNX 转换
  （openvino2onnx）会因 HardSwish/ReduceMean/Resize(opset18) 语义在本版
  OpenCV 上图解析失败，故采用原生 OpenVINO，避免关键点精度受损；
- 实拍视频抽帧回归（每 30 帧抽 1 张，2500μs 曝光 60fps）：
  红色 64/64、蓝色 141/141 检出率 100%，单帧推理 13~25ms；
- PnP 链路（test_solve）：距离抖动 ~1cm/帧，重投影误差 1.6~1.8px
  （同帧传统路线 2.6px）。

## yolov5_512.xml / yolov5_512.bin（同网络 512 输入版）

由 `tools/make_small_ir.py` 从 yolov5.xml 离线手术生成：Detect 头 3 个 Reshape
常量改 `[1,3,22,0,-1]`（special_zero 动态化）、3 个 meshgrid 常量换成
"从 Multiply 输出形状动态生成"的子图，再 reshape 输入到 512。
数值验证：与 640 原模型在同一实拍帧上检出完全一致、置信度一致、
角点偏差 2.5~7.6px（低分辨率固有定位损失）。

复现：`python3 tools/make_small_ir.py models/yolov5.xml models/yolov5_512 512`

## yolov5_640x480.xml / .bin（同网络 640x480 输入版，默认使用）

相机是 4:3（1440x1080），640x640 输入把画面等比缩到 640x480 后，剩下 160 行
全是黑边——图像内容、缩放比例与 640x480 输入完全相同，但那 25% 的卷积是白算的。
本模型就是把输入网格裁成与画面同比例，**精度不变、计算量 -25%**。

由 `tools/make_small_ir.py` 生成（与 512 版同一套手术，额外支持非正方形网格）：

```bash
python3 tools/make_small_ir.py models/yolov5.xml models/yolov5_640x480 640x480
```

关键实现：Reshape 常量动态化后，meshgrid 不能只换"动态子图"（那样分支里
Multiply 的 [1,3,h,w] 会按形状缓存广播，非正方形时张量只涨 25% 而不是 75%，
结果错误但形状合法、不报错），必须在 reshape 到目标尺寸后**烘焙**成常量
`[1,1,gh,gw,22]`（gh=h/stride、gw=w/stride）。顺序上是先换成 [1,1,1,1,22]
占位常量、reshape、再换成目标尺寸常量，否则会撞形状冲突。

数值验证（同一批实拍帧，与 640x640 逐项对比）：

| 项 | 结果 |
| --- | --- |
| 检出数量 / 编号 / 大小板判定 | 5 张实拍帧完全一致 |
| PnP 距离 | 相对差 ≤0.8%（如 9.5888 vs 9.6676 m） |
| 角点 | 同帧偏差 1~4px（0.1~0.4% 画幅） |
| 重投影误差 | 0.02~0.13px，量级相同 |
| 端到端耗时（同进程 AB 交替） | 最小 4.78 ms vs 5.96 ms（-20%），中位 7.03 vs 8.81 ms |

注意：`tests/data/synth_small_1.png` 上两个尺寸的 PnP 距离差 4.7%
（1.320 vs 1.261 m），原因是该合成图自身宽高比与 2.37/4.04 都不符（检出框
3.40，被判成大板），属合成素材问题，两个尺寸表现一致，非本次改动引入。

### 运行参数（环境变量，ArmorDetectorNet）

| 变量 | 取值 | 说明 |
| --- | --- | --- |
| `RMV_DET_SIZE` | `640`（正方形）或 `640x480`（宽x高，需 32 倍数） | 推理输入尺寸，默认 **640x480**；缺对应模型文件自动回退 640x640 |
| `RMV_DET_DEVICE` | CPU/GPU/AUTO | 推理设备，默认 CPU；GPU 编译失败自动回退 CPU（有 iGPU 渲染节点 `/dev/dri` 的机器可用 GPU） |
| `RMV_DET_MODEL` | yolo11 | 切换备选模型 yolo11.xml |

### 512 输入 A/B 验收（2026-10-14，实拍视频抽帧）

| 输入 | 红 64 帧 | 蓝 141 帧 | 平均耗时 |
| --- | --- | --- | --- |
| 640 | 64/64 | 141/141 | 16.2 / 21.8 ms |
| 512 | 64/64 | 141/141 | 11.1 / 16.0 ms |

2026-10-14 用户实机验收：512 在远处（~2m）角点抖动明显（有效像素 -36%，关键点
回归噪声增大，实测同帧角点偏差 2.5~7.6px），**默认值定回 640**（定位稳优先）；
512 模型与 `RMV_DET_SIZE=512` 保留，需要吞吐时可切。

GPU 路径结论：本机 Intel iGPU 有 i915 驱动但容器内无 `/dev/dri` 渲染节点、
`mknod` 被禁，OpenVINO 仅识别 CPU；NVIDIA 显卡 OpenVINO 不支持。
GPU 回退逻辑已实装，宿主环境（有渲染节点）可直接 `RMV_DET_DEVICE=GPU` 使用。

## yolo11.xml / yolo11.bin（备选四点检测）

同来源（sp_vision_25），与 yolov5 并列的备选检测模型。

| 项 | 值 |
| --- | --- |
| 输入 | 1x3x640x640 FP32，预处理同 yolov5 |
| 输出 | 1x50x8400 FP32，transpose 后每行一个候选 |

输出行布局（transpose 后，与 `yolo11.cpp` 解码一致）：

```
[0:4)   xywh 检测框（letterbox 坐标）
[4:42)  38 类分数（颜色x编号x大小组合，armor_properties 表，见 ArmorDetectorNet.cpp）
[42:50) 4 个关键点（letterbox 坐标），需按 y 分组、组内按 x 排序为 TL,TR,BR,BL
```

yolo11 解码器已实现但未经实拍 A/B（默认模型为 yolov5）；切换方式
`RMV_DET_MODEL=yolo11`，实拍验收通过后再作为主用。
