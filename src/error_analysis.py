# -*- coding: utf-8 -*-
# error_analysis.py —— 误差分析(只读, 不训练、不改任何现有文件)
#   目的: 回答"错在哪、错得有多自信、是不是数据/标注的天花板"
#   方法: 本地 5 个 EXP009 种子模型 + TTA(原图+镜像) 在官方验证集 2296 张上做"错题本"
#   产出(全部写进 logs/):
#     error_summary.md      文字总结(错误分布/置信度分解/集中度/最难图名单)
#     error_confusions.txt  每个属性的混淆矩阵(真值×预测)
#     error_samples.csv     全部错误的逐条明细(图名,属性,真值,预测,置信度)
#     error_attr8/9/10.png  最弱三个属性的错题图集(按置信度排序)
#     error_overall.png     错得最多的图集(人眼看是图难还是标注错)
import os
import sys
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)

import dataset as ds
exp = importlib.import_module("EXP009")

DEVICE = exp.DEVICE
ATTR = ds.ATTR_CLASSES
K = len(ATTR)
TAGS = ["EXP009", "EXP009_s1", "EXP009_s2", "EXP009_s3", "EXP009_s4"]
LOGDIR = os.path.join(HERE, "logs")
CONF_HI = 0.90   # "自信错了"的线
CONF_LO = 0.60   # 低于它 = 模型自己也没把握


def load_font(size):
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttf",
               "C:/Windows/Fonts/simhei.ttf"]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_sheet(path, title, tiles, cols=4, TW=160, TH=320, CAP=48):
    """把 (图片, 说明文字) 列表拼成一张大图"""
    GAP, M = 8, 10
    rows = (len(tiles) + cols - 1) // cols
    W = M * 2 + cols * TW + (cols - 1) * GAP
    H = M * 2 + 40 + rows * (TH + CAP) + (rows - 1) * GAP
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    ft, fc = load_font(20), load_font(13)
    d.text((M, M), title, fill="black", font=ft)
    y0 = M + 40
    for idx, (im, cap) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x = M + c * (TW + GAP)
        y = y0 + r * (TH + CAP + GAP)
        canvas.paste(im.resize((TW, TH)), (x, y))
        d.rectangle([x, y, x + TW, y + TH], outline=(170, 170, 170))
        for li, line in enumerate(cap.split("\n")):
            d.text((x + 2, y + TH + 3 + li * 15), line, fill="black", font=fc)
    canvas.save(path)
    print("  图集已存: %s" % path)


def main():
    print("===== 误差分析开始 (5 x EXP009 集成 + TTA) =====")
    _, val_items, index = ds.official_split()
    N = len(val_items)
    ds_val = ds.FFAttrDataset(val_items, index, exp.val_tf)
    loader = torch.utils.data.DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=3)

    # ---- ① 攒平均概率 ----
    prob_sum = [torch.zeros(N, c) for c in ATTR]
    y_true = torch.zeros(N, K, dtype=torch.long)
    for tag in TAGS:
        model = exp.FFResNet(ATTR).to(DEVICE)
        sd = torch.load(os.path.join(HERE, tag + "_best.pt"), map_location="cpu")
        model.load_state_dict(sd)
        model.eval()
        pos = 0
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="  " + tag, leave=False):
                imgs = imgs.to(DEVICE)
                logit1 = model(imgs)
                logit2 = model(torch.flip(imgs, dims=[3]))     # TTA: 左右镜像
                b = imgs.size(0)
                for i in range(K):
                    p = (F.softmax(logit1[i], 1) + F.softmax(logit2[i], 1)) / 2
                    prob_sum[i][pos:pos + b] += p.cpu()
                if tag == TAGS[0]:
                    y_true[pos:pos + b] = labels
                pos += b
        del model
        torch.cuda.empty_cache()
        print("  %s 跑完" % tag)

    probs = [p / len(TAGS) for p in prob_sum]
    yt = y_true.numpy()
    preds = np.zeros_like(yt)
    confs = np.zeros(yt.shape, dtype=np.float32)
    for i in range(K):
        pi = probs[i].numpy()
        preds[:, i] = pi.argmax(1)
        confs[:, i] = pi.max(1)

    correct = (preds == yt)
    acc_per = correct.mean(0)          # 每个属性的准确率
    mean_acc = acc_per.mean()
    n_err_per = (~correct).sum(0)      # 每个属性的错误数
    total_err = int(n_err_per.sum())

    print("\n集成+TTA 验证准确率: %.2f%%   (EXP009_ensemble.csv 存档是 94.07%%, 对上就没算错)"
          % (mean_acc * 100))

    # ---- ② 每个属性: 错误数 + 置信度分档 ----
    rows = []
    tot_hi = tot_mid = tot_lo = 0
    for i in range(K):
        c = confs[~correct[:, i], i]
        hi = int((c >= CONF_HI).sum())
        lo = int((c < CONF_LO).sum())
        mid = len(c) - hi - lo
        tot_hi += hi; tot_mid += mid; tot_lo += lo
        rows.append((i, ATTR[i], acc_per[i], len(c), hi, mid, lo))

    lines = []
    lines.append("# 误差分析报告 (5 x EXP009 集成 + TTA, 官方验证集 %d 张)" % N)
    lines.append("")
    lines.append("- 集成准确率(复现校验): **%.2f%%** (存档记录 94.07%%)" % (mean_acc * 100))
    lines.append("- 属性级错误总数: **%d** / %d (占 %.2f%%)" % (total_err, N * K, total_err / (N * K) * 100))
    lines.append("")
    lines.append("## 1. 每个属性的错误分布")
    lines.append("")
    lines.append("| 属性 | 类别数 | 准确率 | 错误数 | 其中自信错(≥90%%) | 中间(60~90%%) | 不确定错(<60%%) |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, n_cls, a, n_e, hi, mid, lo in rows:
        lines.append("| 属性%d | %d | %.2f%% | %d | %d | %d | %d |" % (i + 1, n_cls, a * 100, n_e, hi, mid, lo))
    lines.append("")
    lines.append("## 2. 错误有多\"自信\"(全部错误合计)")
    lines.append("")
    lines.append("- 自信错(≥90%%): **%d 个, 占 %.1f%%**" % (tot_hi, tot_hi / total_err * 100))
    lines.append("- 中间(60~90%%): %d 个, 占 %.1f%%" % (tot_mid, tot_mid / total_err * 100))
    lines.append("- 不确定(<60%%): %d 个, 占 %.1f%%" % (tot_lo, tot_lo / total_err * 100))
    lines.append("")

    # ---- ③ 错误集中度: 一张图错几个属性 ----
    wrong_n = (~correct).sum(1)
    hist = np.bincount(wrong_n, minlength=K + 1)
    n_bad = int((wrong_n >= 2).sum())
    err_in_bad = int(wrong_n[wrong_n >= 2].sum())
    lines.append("## 3. 错误集中度(同一张图错了几个属性)")
    lines.append("")
    lines.append("- 完全答对: %d 张 (%.1f%%)" % (hist[0], hist[0] / N * 100))
    for k in range(1, 5):
        lines.append("- 错 %d 个属性: %d 张" % (k, hist[k]))
    lines.append("- 错 ≥5 个属性: %d 张" % int(hist[5:].sum()))
    lines.append("- **%d 张\"问题图\"(错≥2个) 吃掉了全部错误的 %.1f%%**" % (n_bad, err_in_bad / total_err * 100))
    lines.append("")

    # 最难的 15 张图名单
    score = wrong_n + ((~correct) * confs).sum(1) * 0.01
    order = np.argsort(-score)
    lines.append("### 最难的 15 张图(错得最多 → 人眼重点看这几张)")
    lines.append("")
    for si in order[:15]:
        bad = [(i, yt[si, i], preds[si, i], confs[si, i]) for i in range(K) if not correct[si, i]]
        bad.sort(key=lambda x: -x[3])
        info = "  ".join("属%d: 真%d→预%d(%.0f%%)" % (i + 1, t, p, c * 100) for i, t, p, c in bad[:4])
        lines.append("- `%s` 错%d个: %s" % (val_items[si][0], len(bad), info))
    lines.append("")

    # ---- ④ 混淆矩阵 + 最常混的类对 ----
    conf_txt = []
    lines.append("## 4. 每个属性最容易混的类对(前 3)")
    lines.append("")
    for i in range(K):
        cm = np.zeros((ATTR[i], ATTR[i]), dtype=int)
        for t, p in zip(yt[:, i], preds[:, i]):
            cm[t, p] += 1
        conf_txt.append("=== 属性%d (共%d类, 准确率%.2f%%) ===" % (i + 1, ATTR[i], acc_per[i] * 100))
        conf_txt.append("        " + "".join("  预%02d" % j for j in range(ATTR[i])))
        for t in range(ATTR[i]):
            conf_txt.append("  真%02d  " + " ".join("%5d" % cm[t, j] for j in range(ATTR[i])))
        pairs = [(int(cm[t, p]), t, p) for t in range(ATTR[i]) for p in range(ATTR[i])
                 if t != p and cm[t, p] > 0]
        pairs.sort(reverse=True)
        s = " / ".join("真%d→错判%d(%d次)" % (t, p, n) for n, t, p in pairs[:3]) or "(无混淆)"
        lines.append("- **属性%d**: %s" % (i + 1, s))
        conf_txt.append("最常混: " + s)
        conf_txt.append("")
    lines.append("")

    # ---- ⑤ 写文件 ----
    with open(os.path.join(LOGDIR, "error_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(LOGDIR, "error_confusions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(conf_txt) + "\n")
    with open(os.path.join(LOGDIR, "error_samples.csv"), "w", encoding="utf-8") as f:
        f.write("name,attr,true,pred,conf(%)\n")
        for si in range(N):
            for i in range(K):
                if not correct[si, i]:
                    f.write("%s,%d,%d,%d,%.1f\n" % (val_items[si][0], i + 1,
                                                     yt[si, i], preds[si, i], confs[si, i] * 100))
    print("  文字报告已存: logs/error_summary.md, error_confusions.txt, error_samples.csv")

    # ---- ⑥ 错题图集 ----
    # 6a. 最弱三属性(8/9/10), 按置信度排序取前16
    for ai in [7, 8, 9]:
        widx = np.where(~correct[:, ai])[0]
        widx = widx[np.argsort(-confs[widx, ai])][:16]
        tiles = []
        for si in widx:
            im = Image.open(index[val_items[si][0]]).convert("RGB")
            cap = "%s\n真%d → 预%d  (%.0f%%)" % (val_items[si][0][:20], yt[si, ai],
                                                 preds[si, ai], confs[si, ai] * 100)
            tiles.append((im, cap))
        make_sheet(os.path.join(LOGDIR, "error_attr%d.png" % (ai + 1)),
                   "属性%d 错题图集(最自信的16个错): 真值 -> 预测" % (ai + 1), tiles)
    # 6b. 错得最多的图
    tiles = []
    for si in order[:16]:
        im = Image.open(index[val_items[si][0]]).convert("RGB")
        bad = [(i, yt[si, i], preds[si, i], confs[si, i]) for i in range(K) if not correct[si, i]]
        bad.sort(key=lambda x: -x[3])
        cap = "%s  (错%d个)\n" % (val_items[si][0][:20], len(bad)) + "  ".join(
            "属%d:%d→%d" % (i + 1, t, p) for i, t, p, c in bad[:3])
        tiles.append((im, cap))
    make_sheet(os.path.join(LOGDIR, "error_overall.png"),
               "错得最多的图 —— 人眼判断: 图本身难? 还是标注可能错?", tiles)

    print("\n===== 完成. 请打开 logs/ 里的 error_summary.md 和 4 张图集 =====")


if __name__ == "__main__":
    main()
