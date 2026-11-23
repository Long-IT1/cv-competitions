# -*- coding: utf-8 -*-
# plot_overfit.py —— 训练 vs 验证 双曲线(过拟合视图), 小倍数图
#   每格 = 一个代表实验, 蓝线=训练正确率, 橙线=验证正确率; 两线缺口 = 过拟合
#   数据来源: logs/EXP*_log.csv 第3列(验证)与第5列(训练正确率, EXP006 起才有)
#   只读, 不改任何现有文件。输出: figures/train_val.png
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
OUT = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"
C_TRAIN = "#2a78d6"    # 训练: 蓝
C_VAL = "#eb6834"      # 验证: 橙 (与蓝的色差已通过色觉安全校验)

PANELS = [
    ("EXP006", "EXP006 +LabelSmoothing"),
    ("EXP008", "EXP008 长宽比448x224"),
    ("EXP009", "EXP009 叠加640x320"),
    ("EXP010", "EXP010 ConvNeXt-Tiny"),
    ("EXP012", "EXP012 ResNet-50"),
    ("EXP014", "EXP014 DenseNet-121"),
]


def read_log(name):
    path = os.path.join(LOGS, name + "_log.csv")
    if not os.path.exists(path):
        return None
    ep, tr, va = [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if parts[0] == "epoch":
            continue
        if len(parts) < 5:      # 老日志没有 train_acc 列
            return None
        ep.append(int(parts[0]))
        va.append(float(parts[2]))
        tr.append(float(parts[4]))
    return (ep, tr, va) if ep else None


fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), dpi=150, sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)
first = True
for ax, (name, title) in zip(axes.flat, PANELS):
    ax.set_facecolor(SURFACE)
    d = read_log(name)
    if d is None:
        ax.set_title(title + "(无数据)", fontsize=9.5, color=INK2)
        continue
    ep, tr, va = d
    ax.plot(ep, tr, color=C_TRAIN, linewidth=2, label="训练正确率" if first else None)
    ax.plot(ep, va, color=C_VAL, linewidth=2, label="验证正确率" if first else None)
    gap = tr[-1] - va[-1]
    ax.set_title("%s\n训练 %.1f / 验证 %.1f(差 %.1f)" % (title, tr[-1], va[-1], gap),
                 fontsize=9.5, color=INK)
    ax.grid(color=GRID, linewidth=0.7)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    if first:
        leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        for t in leg.get_texts():
            t.set_color(INK2)
        first = False

for ax in axes[:, 0]:
    ax.set_ylabel("准确率 (%)", color=INK2, fontsize=9)
for ax in axes[1, :]:
    ax.set_xlabel("训练轮次 epoch", color=INK2, fontsize=9)

fig.suptitle("训练 vs 验证 曲线(过拟合视图)—— 两线缺口越大, 说明模型越在\"背题\"",
             color=INK, fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "train_val.png"), facecolor=SURFACE)
print("已生成: figures/train_val.png")
for name, title in PANELS:
    d = read_log(name)
    if d:
        print("  %s 末轮: 训练 %.1f / 验证 %.1f / 差 %.1f" % (title, d[1][-1], d[2][-1], d[1][-1] - d[2][-1]))
