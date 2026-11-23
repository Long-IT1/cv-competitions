# 集成评估: 把多个模型的预测概率平均, 看能不能比任何单个模型都准
#   - 集成 = 多个模型看同一张图, 12 个属性各取概率平均, 再判类
#   - 可选叠加 TTA (每个模型再看一遍翻转图), 收益可以叠加
# 用法(在 exp 目录下):
#   python eval_ensemble.py                    # 默认用 EXP009 的 5 个种子
#   python eval_ensemble.py EXP009 EXP009_s1   # 也可以手动指定
import sys, os, csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import official_split, FFAttrDataset, ATTR_CLASSES

# EXP009 的五个种子: 第一次跑的不带后缀(种子42), 其余是 _s1~_s4
DEFAULT = ["EXP009", "EXP009_s1", "EXP009_s2", "EXP009_s3", "EXP009_s4"]


def load_models(base_name, suffixes):
    """按 base_name + 后缀 依次加载存档模型"""
    mod = __import__(base_name)                 # 只需要它的 FFResNet / DEVICE / val_tf
    models = []
    for suf in suffixes:
        path = base_name + suf + "_best.pt"
        if not os.path.exists(path):
            print("  !! 找不到存档:", path)
            continue
        m = mod.FFResNet(ATTR_CLASSES).to(mod.DEVICE)
        m.load_state_dict(torch.load(path, map_location=mod.DEVICE))
        m.eval()
        models.append((suf or "_seed42", m))
        print("  已加载 %-16s" % path)
    return mod, models


def evaluate_one(model, loader, device, with_tta):
    """单个模型的准确率"""
    correct = [0] * 12
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            if with_tta:
                lf = model(torch.flip(imgs, dims=[3]))
                logits = [(F.softmax(logits[i], 1) + F.softmax(lf[i], 1)) / 2 for i in range(12)]
            for i in range(12):
                correct[i] += (logits[i].argmax(1) == labels[:, i]).sum().item()
            total += labels.size(0)
    per = [c / total for c in correct]
    return sum(per) / 12


def evaluate_ensemble(models, loader, device, with_tta):
    """集成: 所有模型的概率先平均, 再判类"""
    correct = [0] * 12
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            acc = [0.0] * 12
            for _, m in models:
                logits = m(imgs)
                if with_tta:
                    lf = m(torch.flip(imgs, dims=[3]))
                    for i in range(12):
                        acc[i] = acc[i] + (F.softmax(logits[i], 1) + F.softmax(lf[i], 1)) / 2
                else:
                    for i in range(12):
                        acc[i] = acc[i] + F.softmax(logits[i], 1)
            for i in range(12):
                correct[i] += (acc[i].argmax(1) == labels[:, i]).sum().item()
            total += labels.size(0)
    per = [c / total for c in correct]
    return sum(per) / 12, per


def main():
    names = sys.argv[1:] or DEFAULT
    base = names[0]
    suffixes = [n[len(base):] for n in names]      # "" / "_s1" / "_s2" ...

    print("===== 集成评估 =====")
    mod, models = load_models(base, suffixes)
    if len(models) < 2:
        print("!! 至少要有 2 个模型才能集成")
        return

    _, val_items, index = official_split()
    val_ds = FFAttrDataset(val_items, index, mod.val_tf)
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    print("验证集 %d 张, 模型 %d 个\n" % (len(val_items), len(models)))

    # 1) 每个模型单独的成绩(不做 TTA)
    print("--- 单个模型(普通预测) ---")
    single = {}
    for tag, m in models:
        a = evaluate_one(m, loader, mod.DEVICE, with_tta=False)
        single[tag] = a
        print("  %-10s %.2f%%" % (tag, a * 100))
    best_single = max(single.values())
    print("  最好单个模型: %.2f%%\n" % (best_single * 100))

    # 2) 集成(不做 TTA)
    print("--- 集成(概率平均) ---")
    ens, per_ens = evaluate_ensemble(models, loader, mod.DEVICE, with_tta=False)
    print("  集成准确率: %.2f%%   (比最好单个 %+.2f)\n" % (ens * 100, (ens - best_single) * 100))

    # 3) 集成 + TTA
    print("--- 集成 + TTA(每个模型再看一遍翻转图) ---")
    ens_tta, per_ens_tta = evaluate_ensemble(models, loader, mod.DEVICE, with_tta=True)
    print("  集成+TTA 准确率: %.2f%%   (比单集成 %+.2f)\n" % (ens_tta * 100, (ens_tta - ens) * 100))

    # 4) 逐属性
    print("--- 逐属性 (集成+TTA) ---")
    print("  " + " ".join("属%d:%.2f" % (i + 1, per_ens_tta[i] * 100) for i in range(12)))

    # 5) 存结果
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "EXP009_ensemble.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc(%)"])
        for tag, a in sorted(single.items(), key=lambda x: -x[1]):
            w.writerow([tag, "%.2f" % (a * 100)])
        w.writerow(["best_single", "%.2f" % (best_single * 100)])
        w.writerow(["ensemble", "%.2f" % (ens * 100)])
        w.writerow(["ensemble+tta", "%.2f" % (ens_tta * 100)])
        w.writerow([])
        w.writerow(["attr", "acc(%)"])
        for i in range(12):
            w.writerow([i + 1, "%.2f" % (per_ens_tta[i] * 100)])
    print("\n已保存 logs/EXP009_ensemble.csv")


if __name__ == "__main__":
    main()
