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


def build_dynamic_mesh(P: ov.Output, stride: float) -> ov.Output:
    """从 P [1,3,g,g,22] 动态构造 [1,1,g,g,22] meshgrid（ch0/2/4/6=x*stride，1/3/5/7=y*stride）。"""
    shape = ops.shape_of(P)  # i64 [5]
    g1 = ops.gather(shape, ops.constant(np.array([3], np.int64)),
                    ops.constant(0, np.int64))  # [1] i64，网格边长
    g = ops.squeeze(g1, np.array([0], np.int64))  # 0-d（Range 要求标量）
    g_f = ops.convert(g, np.float32)
    rng = ops.range(ops.constant(np.float32(0)), g_f,
                    ops.constant(np.float32(1)))  # [g] f32
    xs = ops.multiply(ops.reshape(rng, [1, 1, 1, -1, 1], False),
                      ops.constant(stride, np.float32))  # [1,1,1,g,1]
    ys = ops.multiply(ops.reshape(rng, [1, 1, -1, 1, 1], False),
                      ops.constant(stride, np.float32))  # [1,1,g,1,1]
    # 先各自广播到 [1,1,g,g,1] 再沿通道拼 8 份
    target1 = ops.concat(
        [ops.constant(np.array([1, 1], np.int64)), g1, g1,
         ops.constant(np.array([1], np.int64))], axis=0)  # [5]
    xs_b = ops.broadcast(xs, target1)
    ys_b = ops.broadcast(ys, target1)
    xy8 = ops.concat([xs_b, ys_b, xs_b, ys_b, xs_b, ys_b, xs_b, ys_b],
                     axis=4)  # [1,1,g,g,8]
    target14 = ops.concat(
        [ops.constant(np.array([1, 1], np.int64)), g1, g1,
         ops.constant(np.array([14], np.int64))], axis=0)  # [5]
    zeros = ops.broadcast(ops.constant(0.0, np.float32), target14)  # [1,1,g,g,14]
    return ops.concat([xy8, zeros], axis=4)  # [1,1,g,g,22]


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("用法: python3 make_small_ir.py <输入.xml> <输出前缀> <尺寸>")
    src, dst_prefix, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if size % 32 != 0:
        sys.exit("尺寸必须是 32 的倍数（YOLO 三次下采样）")
    stride_by_g = {80: 8, 40: 16, 20: 32}

    core = ov.Core()
    m = core.read_model(src)
    n_reshape = n_mesh = 0

    for op in m.get_ordered_ops():
        t = op.get_type_name()
        if t == "Reshape":
            try:
                vals = list(op.input_value(1).get_node().get_vector())
            except Exception:
                continue
            # Detect 头 3 个 [1,3,22,g,g] → [1,3,22,0,-1]（special_zero 已 True）
            if len(vals) == 5 and list(vals[:3]) == [1, 3, 22] and vals[3] in stride_by_g:
                new_c = ov.op.Constant(np.array([1, 3, 22, 0, -1], dtype=np.int64))
                op.input(1).replace_source_output(new_c.output(0))
                n_reshape += 1
        elif t == "Add" and op.get_friendly_name().startswith("/m/model.26/Add"):
            # 输入 0 = Multiply 输出 [1,3,g,g,22]；输入 1 = meshgrid Convert。
            # 按旧 meshgrid 常量的网格尺寸确定 stride，然后换成动态 mesh。
            old_convert = op.input_value(1).get_node()
            old_const = old_convert.input_value(0).get_node()
            g_old = list(old_const.get_output_shape(0))[2]
            stride = float(stride_by_g[g_old])
            mesh = build_dynamic_mesh(op.input_value(0), stride)
            op.input(1).replace_source_output(mesh.output(0))
            n_mesh += 1

    print(f"patch: Reshape x{n_reshape}, dynamic meshgrid x{n_mesh}")
    if n_reshape != 3 or n_mesh != 3:
        sys.exit("patch 数量不对（应各 3 个），模型结构不是预期的 yolov5 Detect 头")

    m.reshape({m.input(): ov.PartialShape([1, 3, size, size])})
    ov.save_model(m, dst_prefix + ".xml", compress_to_fp16=False)
    print(f"saved: {dst_prefix}.xml / {dst_prefix}.bin (input {size}x{size})")


if __name__ == "__main__":
    main()
