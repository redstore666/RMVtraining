#!/usr/bin/env python3
"""生成可打印的相机标定棋盘格 PDF（A4 横向，9x6 内角点，20mm 方格）。

用法: python3 tools/gen_checkerboard.py [输出路径]
打印要求: 必须选择"实际大小 / 100% 缩放"，打印后用尺子量校验线（应为 100.0 mm）。
"""
import sys

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# 中文字体（系统自带文鼎明体，中文与拉丁数字均完整）
CJK_FONT = "CJK"
pdfmetrics.registerFont(
    TTFont(CJK_FONT, "/usr/share/fonts/truetype/arphic/uming.ttc", subfontIndex=0)
)

COLS, ROWS = 9, 6          # 内角点数（列 x 行）
SQUARE_MM = 20.0           # 每个方格边长
SQUARES_X, SQUARES_Y = COLS + 1, ROWS + 1  # 10 x 7 个方格

out_path = sys.argv[1] if len(sys.argv) > 1 else "tools/checkerboard_a4.pdf"

page_w, page_h = landscape(A4)  # 297mm x 210mm
board_w = SQUARES_X * SQUARE_MM * mm
board_h = SQUARES_Y * SQUARE_MM * mm
x0 = (page_w - board_w) / 2
y0 = (page_h - board_h) / 2

c = canvas.Canvas(out_path, pagesize=landscape(A4))
c.setTitle("Camera calibration checkerboard 9x6 20mm A4")

# 棋盘格：左上角为黑格，黑白交替
for row in range(SQUARES_Y):
    for col in range(SQUARES_X):
        if (row + col) % 2 == 1:
            continue
        c.setFillGray(0)
        x = x0 + col * SQUARE_MM * mm
        # PDF 原点在左下，棋盘第 0 行在顶部
        y = y0 + (SQUARES_Y - 1 - row) * SQUARE_MM * mm
        c.rect(x, y, SQUARE_MM * mm, SQUARE_MM * mm, stroke=0, fill=1)

# 淡灰外框：便于裁剪/检查边界（在图案外，不影响角点检测）
c.setStrokeGray(0.75)
c.setLineWidth(0.5)
c.rect(x0, y0, board_w, board_h, stroke=1, fill=0)

# 顶部说明
c.setFillGray(0)
c.setFont("Helvetica", 9)
c.drawCentredString(
    page_w / 2, page_h - 12 * mm,
    "Calibration checkerboard: 9x6 inner corners, 20 mm squares. "
    "Print at 100%% (actual size), then verify the 100 mm scale bar below.",
)
c.setFont(CJK_FONT, 9)
c.drawCentredString(
    page_w / 2, page_h - 17 * mm,
    "相机标定棋盘格：9x6 内角点，20 mm 方格。务必按 100% 实际大小打印；"
    "打印后先用尺子量页面底部的校验线，应为 100.0 mm。",
)

# 底部 100mm 校验线
bar_y = 10 * mm
bar_x0 = (page_w - 100 * mm) / 2
c.setLineWidth(0.8)
c.line(bar_x0, bar_y, bar_x0 + 100 * mm, bar_y)
for tick in (0, 100):
    x = bar_x0 + tick * mm
    c.line(x, bar_y - 2 * mm, x, bar_y + 2 * mm)
c.setFont("Helvetica", 8)
c.drawCentredString(page_w / 2, bar_y - 6 * mm, "verify: 100.0 mm")

c.save()
print(f"written: {out_path}")
print(f"board: {SQUARES_X}x{SQUARES_Y} squares of {SQUARE_MM} mm "
      f"({board_w / mm:.0f} x {board_h / mm:.0f} mm), inner corners {COLS}x{ROWS}")
