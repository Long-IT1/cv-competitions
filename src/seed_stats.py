# 汇总多种子结果: 每个实验收集所有种子的"最佳验证准确率" -> 均值 / 标准差 / 最低 / 最高
# 用法(在 exp 目录下): python seed_stats.py
# 会同时打印表格, 并写入 logs/seed_stats.csv
import os, re, csv, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")

PAT = re.compile(r"^(EXP[A-Za-z0-9]+?)(?:_s(\d+))?_log\.csv$")


def read_best(path):
    """从日志里取最佳验证准确率: 优先读末尾的 '# best=xx.xx% at epoch N', 没有就取所有轮的最大值"""
    best = None
    vals = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                m = re.search(r"best=([0-9.]+)%", line)
                if m:
                    best = float(m.group(1))
                continue
            parts = line.split(",")
            if len(parts) >= 3 and parts[0] != "epoch":
                try:
                    vals.append(float(parts[2]))
                except ValueError:
                    pass
    if best is not None:
        return best
    return max(vals) if vals else None


groups = {}   # exp -> [(seed标签, 最佳分), ...]
for fn in sorted(os.listdir(LOGS)):
    m = PAT.match(fn)
    if not m:
        continue
    exp, seed = m.group(1), (int(m.group(2)) if m.group(2) else 42)
    v = read_best(os.path.join(LOGS, fn))
    if v is None:
        continue
    groups.setdefault(exp, []).append((seed, v))

if not groups:
    print("没有找到任何日志文件")
    raise SystemExit


def sort_key(item):
    exp = item[0]
    m = re.match(r"EXP(\d+)([a-z]*)", exp)
    return (int(m.group(1)), m.group(2)) if m else (999, exp)


print("实验      | 种子数 |  均值  | 标准差 |  最低  |  最高  | 各种子明细")
print("-" * 88)
rows = []
for exp, items in sorted(groups.items(), key=sort_key):
    items.sort()
    seeds = [s for s, _ in items]
    vals = [v for _, v in items]
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    detail = " ".join("s%d:%.2f" % (s, v) for s, v in items)
    print("%-9s | %5d  | %6.2f | %6.2f | %6.2f | %6.2f | %s"
          % (exp, len(vals), mean, std, min(vals), max(vals), detail))
    rows.append([exp, len(vals), "%.2f" % mean, "%.2f" % std,
                 "%.2f" % min(vals), "%.2f" % max(vals), detail])

out = os.path.join(LOGS, "seed_stats.csv")
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["exp", "n_seeds", "mean", "std", "min", "max", "detail"])
    w.writerows(rows)
print("-" * 88)
print("已写入 logs/seed_stats.csv")
print("说明: 标准差 = 种子之间的波动。两个实验的均值差如果小于 2 个标准差, 就别当真。")
