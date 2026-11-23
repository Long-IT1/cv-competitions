# 读取 logs/*.csv 自动生成实验结果图表(PNG) -> 项目根目录 figures/
# 用法(在 exp 目录下): python plot_results.py    跑完新实验后重新运行即可刷新图表
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
OUT = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(OUT, exist_ok=True)

# 图表配色(固定顺序取色, 与"实验"这个身份绑定, 不随数量变化重新上色)
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"
MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
SERIES = {"EXP001": "#2a78d6", "EXP002": "#eb6834", "EXP002b": "#1baf7a",
          "EXP003": "#eda100", "EXP004": "#e87ba4", "EXP005": "#8b5cf6",
          "EXP006": "#0d9488", "EXP007": "#dc2626", "EXP008": "#4b5563",
          "EXP009": "#0284c7", "EXP010": "#65a30d", "EXP011": "#b45309",
          "EXP012": "#7c3aed", "EXP013": "#be185d", "EXP014": "#166534",
          "EXP015": "#78716c"}
ORDER = ["EXP001", "EXP002", "EXP002b", "EXP003", "EXP004", "EXP005", "EXP006",
         "EXP007", "EXP008", "EXP009", "EXP010", "EXP011", "EXP012", "EXP013",
         "EXP014", "EXP015"]
LABEL = {"EXP001": "EXP001 基线", "EXP002": "EXP002 强增强",
         "EXP002b": "EXP002b 温和增强", "EXP003": "EXP003 +余弦退火",
         "EXP004": "EXP004 +预训练", "EXP005": "EXP005 +CBAM",
         "EXP006": "EXP006 +LabelSmoothing", "EXP007": "EXP007 分辨率320x320",
         "EXP008": "EXP008 长宽比448x224", "EXP009": "EXP009 叠加640x320",
         "EXP010": "EXP010 ConvNeXt-T", "EXP011": "EXP011 可变形卷积",
         "EXP012": "EXP012 ResNet-50", "EXP013": "EXP013 7x7大卷积",
         "EXP014": "EXP014 DenseNet-121", "EXP015": "EXP015 ConvNeXt-S"}

def read_log(name):
    path = os.path.join(LOGS, name + "_log.csv")
    if not os.path.exists(path):
        return None
    epochs, accs, best, per = [], [], None, None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.search(r"per_attr_acc\(%\): ([0-9.,]+)", line)
            if m:
                per = [float(x) for x in m.group(1).split(",")]
            m2 = re.search(r"best=([0-9.]+)%", line)
            if m2:
                best = float(m2.group(1))
            continue
        parts = line.split(",")
        if len(parts) >= 4 and parts[0] != "epoch":
            epochs.append(int(parts[0]))
            accs.append(float(parts[2]))
    if not epochs:
        return None
    return {"epochs": epochs, "accs": accs, "best": best if best else max(accs),
            "per": per, "finished": best is not None}

data = {}
for name in ORDER:
    d = read_log(name)
    if d:
        data[name] = d
print("读到数据的实验:", list(data.keys()))

# ---------- 图1: 验证准确率曲线 ----------
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for name in ORDER:
    if name not in data:
        continue
    d = data[name]
    ax.plot(d["epochs"], d["accs"], color=SERIES[name], linewidth=1.8,
            marker="o", markersize=2.5, label=LABEL[name])
    # 线条末端直接标注(身份不只靠颜色); 未跑完的标注"进行中"
    note = "" if d["finished"] else " (进行中)"
    ax.annotate("%.1f%s" % (d["accs"][-1], note), (d["epochs"][-1], d["accs"][-1]),
                xytext=(5, 0), textcoords="offset points",
                fontsize=8, color=INK2, va="center")
ax.set_title("验证准确率对比(12属性平均)", color=INK, fontsize=12)
ax.set_xlabel("训练轮次 epoch", color=INK2, fontsize=9)
ax.set_ylabel("验证准确率 (%)", color=INK2, fontsize=9)
ax.grid(color=GRID, linewidth=0.8)
ax.tick_params(colors=MUTED, labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
for t in leg.get_texts():
    t.set_color(INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "val_curve.png"), facecolor=SURFACE)
print("已生成: figures/val_curve.png")

# ---------- 图2: 最佳成绩对比(点图, 不用截断的柱状图) ----------
fig2, ax2 = plt.subplots(figsize=(7.2, 3.4), dpi=150)
fig2.patch.set_facecolor(SURFACE); ax2.set_facecolor(SURFACE)
names = [n for n in ORDER if n in data][::-1]     # 从上到下按实验顺序
ys = list(range(len(names)))
base = data["EXP001"]["best"] if "EXP001" in data else None
if base:
    ax2.axvline(base, color=MUTED, linewidth=1, linestyle="--")
    ax2.text(base, len(names) - 0.55, " 基线 %.2f%%" % base, fontsize=8, color=INK2)
for i, name in enumerate(names):
    v = data[name]["best"]
    note2 = "" if data[name]["finished"] else " (进行中)"
    ax2.scatter(v, i, s=90, color=SERIES[name], zorder=3)
    ax2.annotate("%.2f%%%s" % (v, note2), (v, i), xytext=(9, 0), textcoords="offset points",
                 fontsize=9, color=INK2, va="center")
ax2.set_yticks(ys)
ax2.set_yticklabels([LABEL[n] for n in names], fontsize=9, color=INK2)
ax2.set_xlabel("最佳验证准确率 (%)", color=INK2, fontsize=9)
ax2.set_title("各实验最佳成绩对比", color=INK, fontsize=12)
ax2.grid(axis="x", color=GRID, linewidth=0.8)
ax2.tick_params(colors=MUTED, labelsize=8)
ax2.set_xlim(min([data[n]["best"] for n in names]) - 3,
             max([data[n]["best"] for n in names]) + 3)
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
ax2.spines["bottom"].set_color(AXIS)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "best_acc.png"), facecolor=SURFACE)
print("已生成: figures/best_acc.png")

# ---------- 图3: 12属性准确率热力图 ----------
rows = [n for n in ORDER if n in data and data[n].get("per")]
if rows:
    matrix = [data[n]["per"] for n in rows]
    blue_steps = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                  "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                  "#184f95", "#104281", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("blue_seq", blue_steps)
    vmin, vmax = 40.0, 100.0
    fig3, ax3 = plt.subplots(figsize=(8.4, 0.85 * len(rows) + 1.6), dpi=150)
    fig3.patch.set_facecolor(SURFACE)
    im = ax3.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(len(rows)):
        for j in range(12):
            v = matrix[i][j]
            t = (v - vmin) / (vmax - vmin)
            ax3.text(j, i, "%.0f" % v, ha="center", va="center", fontsize=8,
                     color="#ffffff" if t > 0.6 else INK)
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(["属性%d" % (j + 1) for j in range(12)], fontsize=8, color=INK2)
    ax3.set_yticks(range(len(rows)))
    ax3.set_yticklabels([LABEL[n] for n in rows], fontsize=9, color=INK2)
    ax3.set_title("各属性准确率热力图(%)", color=INK, fontsize=12)
    cb = fig3.colorbar(im, ax=ax3, fraction=0.035, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUT, "attr_heatmap.png"), facecolor=SURFACE, bbox_inches="tight")
    print("已生成: figures/attr_heatmap.png")
else:
    print("暂无逐属性数据, 跳过热力图")
