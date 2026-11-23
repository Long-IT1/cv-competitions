# -*- coding: utf-8 -*-
# make_confusion_heatmap.py —— 12 个属性的混淆矩阵热力图
#   数据来源: logs/ens_probs.npz (probe_prior_correction.py 存档的 5 种子集成概率, 无需跑模型)
#   每行归一化: 数字 = 该"真实类"被判给各"预测类"的比例(%)
#   只读, 不改任何现有文件。输出: figures/confusion.png
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "logs", "ens_probs.npz")
OUT = os.path.join(os.path.dirname(HERE), "figures")

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
ATTR = [6, 5, 4, 3, 5, 3, 3, 3, 5, 7, 3, 3]

# 与项目其它图一致的单色蓝序(浅->深 = 占比低->高)
blue_steps = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
              "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
              "#184f95", "#104281", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("blue_seq", blue_steps)

z = np.load(NPZ)
yt = z["labels"]
preds = [z["p%d" % i].argmax(1) for i in range(12)]

fig, axes = plt.subplots(3, 4, figsize=(13.5, 9.2), dpi=150)
fig.patch.set_facecolor(SURFACE)
for i, ax in enumerate(axes.flat):
    c = ATTR[i]
    cm = np.zeros((c, c))
    for t, p in zip(yt[:, i], preds[i]):
        cm[t, p] += 1
    row = np.maximum(cm.sum(1, keepdims=True), 1)
    norm = cm / row * 100.0
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for t in range(c):
        for p in range(c):
            v = norm[t, p]
            col = "#ffffff" if v > 60 else (INK if v > 4 else "#c3c2b7")
            ax.text(p, t, "%.0f" % v, ha="center", va="center", fontsize=7, color=col)
    acc = (preds[i] == yt[:, i]).mean() * 100
    ax.set_title("属性%d (%d 类, 准确率 %.1f%%)" % (i + 1, c, acc), fontsize=9.5, color=INK)
    ax.set_xticks(range(c))
    ax.set_xticklabels(range(c), fontsize=7, color=MUTED)
    ax.set_yticks(range(c))
    ax.set_yticklabels(range(c), fontsize=7, color=MUTED)
    if i >= 8:
        ax.set_xlabel("预测类", fontsize=8, color=INK2)
    if i % 4 == 0:
        ax.set_ylabel("真实类", fontsize=8, color=INK2)
    for s in ax.spines.values():
        s.set_color("#e1e0d9")

fig.suptitle("12 属性混淆矩阵热力图(5 种子集成+TTA)—— 每行=一个真实类, 数字=它被判给各预测类的比例(%), 对角线=判对",
             fontsize=11.5, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(os.path.join(OUT, "confusion.png"), facecolor=SURFACE)
print("已生成: figures/confusion.png")

# 顺手打印几个亮点, 供报告引用
print("\n每属性准确率与最弱真实类(召回最低):")
for i in range(12):
    cm = np.zeros((ATTR[i], ATTR[i]))
    for t, p in zip(yt[:, i], preds[i]):
        cm[t, p] += 1
    row = np.maximum(cm.sum(1, keepdims=True), 1)
    rec = (cm / row)[:, :].diagonal() * 100
    worst = int(np.argmin(rec))
    print("  属性%-2d 准确率 %5.1f%%  最弱类: 类%d (该真实类共%d张, 召回%.0f%%)"
          % (i + 1, (preds[i] == yt[:, i]).mean() * 100, worst,
             int(cm[worst].sum()), rec[worst]))
