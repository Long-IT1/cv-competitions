# -*- coding: utf-8 -*-
# make_error_examples.py —— 典型错误实例图: 四张最有代表性的错图 + 一句人话说明
#   用于第4周周报"误差分析"一节(人眼验证的证据)
#   只读, 不改任何现有文件。输出: figures/error_examples.png
import os
import sys
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)
import dataset as ds

OUT = os.path.join(os.path.dirname(HERE), "figures")

# (图片名, 类别标签, 显示用短名, 判断详情, 说明行)
EXAMPLES = [
    ("WOMEN-Dresses-id_00002013-02_1_front.png",
     "叠穿", "WOMEN-Dresses 00002013",
     "属性8: 真0 → 预1 (96%)",
     ["碎花连衣裙外搭棒球外套，", "“属性说的是哪一件”？"]),
    ("WOMEN-Sweatshirts_Hoodies-id_00005200-01_7_additional.png",
     "图案·姿势", "WOMEN-Sweatshirts 00005200",
     "属性8: 真0 → 预1 (96%)",
     ["扎染长袖+破洞牛仔裤、", "坐凳子姿势（侧后方视角）"]),
    ("WOMEN-Dresses-id_00006755-01_2_side.png",
     "稀有类", "WOMEN-Dresses 00006755",
     "属性10: 真1 → 预5 (93%)",
     ["该类的 17 张验证图全部判错；", "照片本身看不出任何“异样”"]),
    ("WOMEN-Graphic_Tees-id_00005522-01_4_full.png",
     "多层同框", "WOMEN-Graphic_Tees 00005522",
     "属性9: 真3 → 预0 (92%)",
     ["外套+字母T恤+碎花短裙三层，", "说哪一层都说得通"]),
]


def load_font(size):
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttf",
               "C:/Windows/Fonts/simhei.ttf"]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    _, _, index = ds.official_split()
    TW, CAP = 230, 108         # 图宽 230 -> 高 460; 说明区 108px
    GAP, M = 12, 14
    cols = len(EXAMPLES)
    W = M * 2 + cols * TW + (cols - 1) * GAP
    H = M * 2 + 46 + 2 * TW + CAP          # 2*TW: 1:2 竖图
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_title = load_font(21)
    f_tag = load_font(15)
    f_name = load_font(12)
    f_body = load_font(13)

    d.text((M, M), "典型错误实例（5 种子集成+TTA）—— 错的不是“看不清”，是“本身难”",
           fill="black", font=f_title)

    y0 = M + 46
    for i, (name, tag, short, detail, desc) in enumerate(EXAMPLES):
        x = M + i * (TW + GAP)
        img = Image.open(index[name]).convert("RGB").resize((TW, TW * 2))
        canvas.paste(img, (x, y0))
        d.rectangle([x, y0, x + TW, y0 + TW * 2], outline=(170, 170, 170))
        cy = y0 + TW * 2 + 6
        d.text((x + 2, cy), "【%s】" % tag, fill=(176, 83, 25), font=f_tag)
        d.text((x + 2, cy + 21), short, fill="black", font=f_name)
        d.text((x + 2, cy + 39), detail, fill="black", font=f_body)
        for li, line in enumerate(desc):
            d.text((x + 2, cy + 59 + li * 17), line, fill=(82, 81, 78), font=f_body)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "error_examples.png")
    canvas.save(p)
    print("已生成: figures/error_examples.png  (%dx%d)" % (W, H))


if __name__ == "__main__":
    main()
