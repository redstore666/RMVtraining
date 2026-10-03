# models/ 离线模型资源

## armor_cls_64.onnx

装甲板数字分类器（9 类），供 `armor_detect` 使用。

| 项 | 值 |
| --- | --- |
| 网络 | YOLO26n-cls（Ultralytics 官方预训练 `yolo26n-cls.pt` 微调） |
| 训练数据 | 课程提供数据集 `RMVtrainingHomework/datasets/`，9 类：`1,2,3,4,5,6outpost,7guard,8base,9neg` |
| 训练配置 | imgsz=64，batch=256，flr=0，50 epochs 上限（早停于 34，最佳 14），seed=0 |
| 验证结果 | 去重划分后 val top1 = 100%（1775 张）|
| 导出 | ONNX opset 12，输入 1x3x64x64，输出 1x9 softmax（已含 softmax） |
| 大小 | 6.2 MB |

### 类别顺序（输出向量下标 → 含义）

```
0:'1'  1:'2'  2:'3'  3:'4'  4:'5'  5:'6outpost'  6:'7guard'  7:'8base'  8:'9neg'
```

对应 `ArmorDetection::target_id = index + 1`；`9neg` 表示"不是数字"，必须丢弃。

### 推理预处理（必须与训练一致，ArmorClassifier 已实现）

1. 灰度化（训练图是单通道贴纸裁片）；
2. Otsu 二值化（训练图是"白字黑底"的二值图，二值化让实拍裁剪贴近训练分布）；
3. 短边缩放到 64（截断取整），中心裁剪 64x64；
4. /255 归一化，RGB 顺序，NCHW。

### 重要：裁剪必须紧贴数字

训练裁片里数字几乎充满整幅图。实测把"灯条间整个区域"直接送去分类
（数字仅占画面 ~8%）会被判成 `9neg`。正确做法：二值化后取最大连通域
（或面积 ≥ 最大域 10% 的连通域并集）的外接矩形，外扩约 15% 再送入分类。

### 复现步骤

```bash
# 1) 划分（按 md5 去重，避免同帧重复样本跨 train/val）
cd RMVtrainingHomework/train_cls && python3 split_data.py
# 2) 训练
yolo classify train data=dataset model=yolo26n-cls.pt imgsz=64 epochs=50 batch=256 \
    fliplr=0.0 patience=20 cache=True seed=0
# 3) 导出
yolo export model=runs_cls/cls64_v2/weights/best.pt format=onnx opset=12 imgsz=64
```

训练脚本与完整结果（results.csv、混淆矩阵）在 `RMVtrainingHomework/train_cls/`。

### 部署链路验证记录（2026-10-03）

- Python：手动预处理 + `cv2.dnn` 与 Ultralytics 推理在 54/54 张 val 图上 argmax 一致，概率最大偏差 0.0000；
- C++：系统 OpenCV 4.5.4 `cv::dnn::readNetFromONNX` 加载成功，`tests/test_classifier` 验证通过。
