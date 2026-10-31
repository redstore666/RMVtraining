#!/usr/bin/env python3
"""从 test_camera_system 抓的 SSE dump 里算"跟踪行为"指标。

用法：
    tools/analyze_sse.py out/base.sse out/pair.sse          # 并排对比
    tools/analyze_sse.py out/base.sse --yaw                 # 再打印 yaw 随时间（每 2 s）
    tools/analyze_sse.py out/base.sse --keys yaw_deg,omega  # 只看某几个键的分布

SSE 格式（tests/web_system/src/SystemWebLogger.cpp）：
    data: [{"k":"omega","v":0.01,"t":1698...}, ...]

指标口径（和 docs/v4_webdebug_report.md 一致）：
  * dYaw      —— yaw_deg 逐帧差，先 unwrap。判"模型有没有在转"就看它：
                 中位 0 且净转角 0 = 冻住；p90 大 = 抖（用户说的"振荡"）
  * net       —— 整段净转角。真车在转、模型不转 → 0
  * omega     —— EKF 的 state(7)。|w|>=5 的比例是"方波"指标（修前 20.6%）
  * phase_lag —— 观测法向 − 模型槽位法向。静止车系统性 −45°（几何/约定偏置，
                 不是噪声，见报告 §7），所以要看分布宽度而不是绝对值
  * R         —— 半径。正常 0.25~0.27 m；崩到 0.15（clamp 下限）说明状态坏了
"""
import json
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("需要 numpy：pip install numpy")


def load(path):
    keys = {}
    with open(path, errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                batch = json.loads(line[5:].strip())
            except Exception:
                continue
            for rec in batch:
                keys.setdefault(rec["k"], []).append(rec["v"])
    return {k: np.asarray(v, dtype=float) for k, v in keys.items()}


def pct(arr, p):
    return float(np.percentile(arr, p)) if arr.size else float("nan")


def has(data, key):
    return key in data and data[key].size > 0


def report(path, tag):
    d = load(path)
    out = {"file": path, "tag": tag}
    n = len(d.get("frame_dt_ms", np.zeros(0)))
    out["frames"] = n
    if has(d, "frame_dt_ms"):
        out["video_s"] = float(np.nansum(d["frame_dt_ms"])) / 1000.0
    if has(d, "yaw_deg"):
        yaw = np.unwrap(np.radians(d["yaw_deg"]))
        dy = np.degrees(np.diff(yaw))
        out["dYaw_med"] = float(np.median(np.abs(dy))) if dy.size else float("nan")
        out["dYaw_p90"] = pct(np.abs(dy), 90)
        out["jump5"] = float(np.mean(np.abs(dy) > 5.0) * 100.0) if dy.size else float("nan")
        out["jump20"] = float(np.mean(np.abs(dy) > 20.0) * 100.0) if dy.size else float("nan")
        out["net_deg"] = float(np.degrees(yaw[-1] - yaw[0])) if yaw.size else float("nan")
    if has(d, "omega"):
        w = d["omega"][np.isfinite(d["omega"])]
        out["w_med"] = float(np.median(w))
        out["w_p90abs"] = pct(np.abs(w), 90)
        out["w_ge5"] = float(np.mean(np.abs(w) >= 5.0) * 100.0)
        out["w_min"], out["w_max"] = float(np.min(w)), float(np.max(w))
    if has(d, "phase_lag_deg"):
        lag = d["phase_lag_deg"][np.abs(d["phase_lag_deg"]) < 1e3]
        out["lag_med"] = float(np.median(lag))
        out["lag_p10"], out["lag_p90"] = pct(lag, 10), pct(lag, 90)
        out["lag_in20"] = float(np.mean(np.abs(lag) < 20.0) * 100.0)
    if has(d, "radius"):
        r = d["radius"][np.isfinite(d["radius"])]
        out["R_med"], out["R_p10"] = float(np.median(r)), pct(r, 10)
    if has(d, "tracking"):
        out["tracking"] = float(np.nanmean(d["tracking"]))
    if has(d, "updates"):
        out["zero_upd"] = float(np.mean(d["updates"] <= 0.0) * 100.0)
    for k in ("pair_solves", "realigns", "body_refits", "missed"):
        if has(d, k):
            out[k] = float(np.nansum(d[k]))
    if has(d, "aim_miss"):
        out["aim_miss_med"] = float(np.median(d["aim_miss"]))
    for k in ("loop_ms", "detect_ms"):
        if has(d, k):
            out[k + "_med"] = float(np.median(d[k]))
    return out, d


def show(rows):
    def fmt(value, spec="6.2f"):
        if value is None:
            return "-"
        if isinstance(value, float):
            return format(value, spec)
        return str(value)

    head = ("tag", "frames", "video_s", "dYaw_med", "dYaw_p90", "jump5%", "net_deg",
            "w_med", "w_p90abs", "w_ge5%", "lag_med", "lag_in20%", "R_med",
            "pair_solves", "zero_upd%", "aim_miss_med")
    widths = (16, 8, 8, 9, 9, 7, 8, 7, 9, 7, 8, 9, 7, 12, 10, 12)
    print("".join(h.ljust(w) for h, w in zip(head, widths)))
    for r in rows:
        cells = (fmt(r.get("tag")), fmt(r.get("frames")), fmt(r.get("video_s")),
                 fmt(r.get("dYaw_med")), fmt(r.get("dYaw_p90")), fmt(r.get("jump5"), "5.1f"),
                 fmt(r.get("net_deg"), "7.0f"), fmt(r.get("w_med"), "+.2f"),
                 fmt(r.get("w_p90abs")), fmt(r.get("w_ge5"), "5.1f"),
                 fmt(r.get("lag_med"), "+.1f"), fmt(r.get("lag_in20"), "5.1f"),
                 fmt(r.get("R_med"), ".3f"), fmt(r.get("pair_solves"), ".0f"),
                 fmt(r.get("zero_upd"), "5.1f"), fmt(r.get("aim_miss_med"), ".0f"))
        print("".join(str(c).ljust(w) for c, w in zip(cells, widths)))


def show_yaw(d, step_s=2.0):
    if not has(d, "yaw_deg"):
        return
    yaw = np.degrees(np.unwrap(np.radians(d["yaw_deg"])))
    dt = float(np.median(d["frame_dt_ms"])) / 1000.0 if has(d, "frame_dt_ms") else 0.01
    stride = max(1, int(round(step_s / dt)))
    print("  yaw 轨迹（每 %.0f s）:" % step_s)
    prev = None
    for i in range(0, len(yaw), stride):
        delta = "" if prev is None else " (Δ%+6.1f)" % (yaw[i] - prev)
        print("    t=%5.1fs yaw=%+8.1f%s" % (i * dt, yaw[i], delta))
        prev = yaw[i]


def main(argv):
    files = [a for a in argv[1:] if not a.startswith("--") and not a.endswith(".csv")]
    want_yaw = "--yaw" in argv
    only_keys = None
    if "--keys" in argv:
        only_keys = argv[argv.index("--keys") + 1].split(",")
    if not files:
        print(__doc__)
        return 1
    rows = []
    for path in files:
        tag = path.rsplit("/", 1)[-1].replace(".sse", "")
        row, data = report(path, tag)
        rows.append(row)
        if want_yaw:
            show_yaw(data)
        if only_keys:
            for k in only_keys:
                if has(data, k):
                    v = data[k][np.isfinite(data[k])]
                    print("  %-20s n=%5d med=%+9.3f p10=%+9.3f p90=%+9.3f min=%+9.3f max=%+9.3f"
                          % (tag + ":" + k, v.size, np.median(v), pct(v, 10), pct(v, 90),
                             np.min(v), np.max(v)))
    print()
    show(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
