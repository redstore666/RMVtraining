#!/usr/bin/env python3
"""把 sp_vision_25 yolov5 IR 的 Detect 头硬编码网格改为动态/目标输入尺寸，导出新 IR。

用法:
    python3 make_small_ir.py <输入.xml> <输出前缀> <尺寸>

原理:
  1. Detect 头 3 个 Reshape 的 shape 常量写死 [1,3,22,80,80] 等网格，算子
     special_zero=True —— 换 [1,3,22,0,-1]：0 拷贝输入第 3 维（随输入尺寸
     自动变化），-1 推断最后一维，任何输入尺寸都合法；
  2. Detect 头 3 个 meshgrid 常量（[1,1,g,g,22]，通道 0/2/4/6=x*stride、
     1/3/5/7=y*stride）是纯数据且尺寸写死 —— 替换为"从 Multiply 输出形状
     动态生成 meshgrid"的子图（ShapeOf→Gather→Range→Reshape→Concat），
     任何输入尺寸都自动正确；
  3. model.reshape 输入到目标尺寸，另存 IR。

输出: <输出前缀>.xml / <输出前缀>.bin
依赖: pip install openvino numpy（见 环境依赖.md）
"""
import sys

import numpy as np
import openvino as ov
import openvino.opset1 as ops


def build_static_mesh(gh: int, gw: int, stride: float) -> np.ndarray:
    """按目标网格尺寸直接烘出 [1,1,gh,gw,22] 的常量 meshgrid。

    早期版本用 Range/shape_of 造"动态"mesh，只有在正方形输入时才对得上；
    非正方形（如 640x480）会让模型的坐标解码错位（实测检出框整体跑偏、
    还会多出假框）。这里改成和原模型一样的"烘焙常量"，只是把网格尺寸换成
    目标尺寸，语义与原图完全一致。
    """
    ys = (np.arange(gh, dtype=np.float32) * np.float32(stride)).reshape(1, 1, gh, 1, 1)
    xs = (np.arange(gw, dtype=np.float32) * np.float32(stride)).reshape(1, 1, 1, gw, 1)
    ys = np.broadcast_to(ys, (1, 1, gh, gw, 1))
    xs = np.broadcast_to(xs, (1, 1, gh, gw, 1))
    xy8 = np.concatenate([xs, ys, xs, ys, xs, ys, xs, ys], axis=4)  # [1,1,gh,gw,8]
    zeros = np.zeros((1, 1, gh, gw, 14), dtype=np.float32)
    return np.concatenate([xy8, zeros], axis=4)  # [1,1,gh,gw,22]


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("用法: python3 make_small_ir.py <输入.xml> <输出前缀> <尺寸|宽x高>\n"
                 "  例: make_small_ir.py models/yolov5.xml models/yolov5_640x480 640x480")
    src, dst_prefix, size = sys.argv[1], sys.argv[2], sys.argv[3]
    # 尺寸支持 "640" 或 "640x480"：相机是 4:3，正方形输入会把 1/4 的
    # 画布浪费在黑边上（1440x1080 -> 640x480 + 160 行黑边）。
    if "x" in size.lower():
        parts = size.lower().split("x")
        width, height = int(parts[0]), int(parts[1])
    else:
        width = height = int(size)
    for value in (width, height):
        if value <= 0 or value % 32 != 0:
            sys.exit("宽高都必须是 32 的正倍数（YOLO 三次下采样）")
    core = ov.Core()
    m = core.read_model(src)
    n_reshape = n_mesh = 0
    stride_by_g = {80: 8, 40: 16, 20: 32}

    # 分两步，顺序不能反：
    #  ① 先把 Detect 头的 Reshape 目标形状改成 [1,3,22,0,-1]（0=照抄输入对应维，
    #    -1=自动推断），这样它对新尺寸永远合法，reshape 才能通过形状校验；
    #  ② 把输入改到目标尺寸，让整图传播；
    #  ③ 再把 meshgrid 换成目标尺寸的**烘焙常量**（此时才与广播形状兼容）。
    # 早期版本在第 ③ 步用 Range/shape_of 造"动态"网格，正方形时恰好对，非正方形
    # 会让坐标解码错位（检出框整体偏移、还多出假框），所以这里坚持烘常量。
    for op in m.get_ordered_ops():
        if op.get_type_name() != "Reshape":
            continue
        try:
            vals = list(op.input_value(1).get_node().get_vector())
        except Exception:
            continue
        if len(vals) == 5 and list(vals[:3]) == [1, 3, 22] and vals[3] in stride_by_g:
            # 这个 Reshape 把 [1,66,gh,gw] 拆成 [1,3,22,gh,gw]。不能写成常量
            # [1,3,22,0,-1]：special_zero 的 0 只会照抄"同一维"，非正方形时
            # gh/gw 会被交换（这正是 640x480 坐标整体跑偏的原因）。改成从输入
            # 的 shape 里动态取最后两维，尺寸无关且轴序保持。
            src_shape = ops.shape_of(op.input_value(0))
            tail = ops.gather(src_shape, ops.constant(np.array([2, 3], np.int64)),
                              ops.constant(0, np.int64))  # [2] = [gh, gw]
            pattern = ops.concat([ops.constant(np.array([1, 3, 22], np.int64)), tail],
                                 axis=0)  # [5]
            op.input(1).replace_source_output(pattern.output(0))
            n_reshape += 1
    print(f"patch: Reshape x{n_reshape}")

    # 先把 meshgrid 换成"什么尺寸都能广播"的占位常量（[1,1,1,1,22]），否则
    # 下一步 reshape 会因为旧常量还是 80x80 而形状冲突；reshape 之后再换成
    # 目标尺寸的烘焙常量。
    pending = []
    for op in m.get_ordered_ops():
        if op.get_type_name() != "Add" or not op.get_friendly_name().startswith("/m/model.26/Add"):
            continue
        old_convert = op.input_value(1).get_node()
        old_const = old_convert.input_value(0).get_node()
        g_old = list(old_const.get_output_shape(0))[2]
        stride = stride_by_g[g_old]
        placeholder = ov.op.Constant(np.zeros((1, 1, 1, 1, 22), dtype=np.float32))
        op.input(1).replace_source_output(placeholder.output(0))
        pending.append((op, stride))
    print(f"patch: meshgrid 占位 x{len(pending)}")

    m.reshape({m.input(): ov.PartialShape([1, 3, height, width])})

    grids = {8: (height // 8, width // 8), 16: (height // 16, width // 16),
             32: (height // 32, width // 32)}
    n_mesh = 0
    for op, stride in pending:
        gh, gw = grids[stride]
        mesh = ov.op.Constant(build_static_mesh(gh, gw, float(stride)))
        op.input(1).replace_source_output(mesh.output(0))
        n_mesh += 1

    print(f"patch: meshgrid x{n_mesh} (目标 {width}x{height})")
    if n_reshape != 3 or n_mesh != 3:
        sys.exit("patch 数量不对（应各 3 个），模型结构不是预期的 yolov5 Detect 头")

    ov.save_model(m, dst_prefix + ".xml", compress_to_fp16=False)
    print(f"saved: {dst_prefix}.xml / {dst_prefix}.bin (input {width}x{height})")


if __name__ == "__main__":
    main()
