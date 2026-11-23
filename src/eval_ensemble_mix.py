# 异源集成: 把"不同架构 / 不同输入尺寸"的模型放在一起投票
#   比同源种子集成更强 —— 因为不同架构的"错法"差得远, 互补性大
#   每个模型用自己那份预处理(比如 ResNet 用 640x320, ConvNeXt 用 448x224), 概率再平均
# 用法(在 exp 目录下):
#   python eval_ensemble_mix.py                                   # 默认: 5个ResNet + ConvNeXt
#   python eval_ensemble_mix.py EXP009 EXP009_s1 EXP010           # 也可以手动指定
import sys, os, csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import official_split, FFAttrDataset, ATTR_CLASSES

DEFAULT = ["EXP009", "EXP009_s1", "EXP009_s2", "EXP009_s3", "EXP009_s4", "EXP010"]
BATCH = 64


def base_of(tag):
    """EXP009_s1 -> EXP009 ; EXP010 -> EXP010 (种子文件没有自己的 .py)"""
    return tag.split("_")[0]


def main():
    args = sys.argv[1:]
    WITH_TTA = "--tta" in args                      # 加上 --tta 就同时用"原图+翻转"
    args = [a for a in args if not a.startswith("--")]
    names = args or DEFAULT
    _, val_items, index = official_split()
    N = len(val_items)
    labels = torch.tensor([lab for _, lab in val_items], dtype=torch.long)   # [N, 12]
    print("验证集 %d 张, 模型 %d 个" % (N, len(names)))
    print()

    acc_all = [torch.zeros(N, c) for c in ATTR_CLASSES]     # 累积所有模型的概率
    singles = {}

    for tag in names:
        path = tag + "_best.pt"
        if not os.path.exists(path):
            print("  !! 找不到存档:", path)
            continue
        mod = __import__(base_of(tag))                       # 用该实验自己的模型结构和预处理
        model = mod.FFResNet(ATTR_CLASSES).to(mod.DEVICE)
        model.load_state_dict(torch.load(path, map_location=mod.DEVICE))
        model.eval()

        loader = DataLoader(FFAttrDataset(val_items, index, mod.val_tf),
                            batch_size=BATCH, shuffle=False, num_workers=0)
        cur = [torch.zeros(N, c) for c in ATTR_CLASSES]
        p = 0
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(mod.DEVICE)
                logits = model(imgs)
                b = imgs.size(0)
                if WITH_TTA:
                    lf = model(torch.flip(imgs, dims=[3]))     # 再看一遍翻转图
                    probs = [(F.softmax(logits[i], 1) + F.softmax(lf[i], 1)) / 2 for i in range(12)]
                else:
                    probs = [F.softmax(logits[i], 1) for i in range(12)]
                for i in range(12):
                    cur[i][p:p + b] = probs[i].cpu()
                p += b
            if p != N:
                print("  !! %s 只跑了 %d 张 (应为 %d)" % (tag, p, N))

        a = sum((cur[i].argmax(1) == labels[:, i]).float().mean().item() for i in range(12)) / 12
        singles[tag] = a
        print("  %-12s 单独 %.2f%%   (输入 %s)" % (tag, a * 100, "x".join(map(str, list(mod.val_tf.transforms[0].size)))))
        for i in range(12):
            acc_all[i] += cur[i]

    if len(singles) < 2:
        print("!! 至少要有 2 个模型")
        return

    best_single = max(singles.values())
    ens = sum((acc_all[i].argmax(1) == labels[:, i]).float().mean().item() for i in range(12)) / 12
    per_attr = [(acc_all[i].argmax(1) == labels[:, i]).float().mean().item() for i in range(12)]

    print()
    print("=" * 56)
    print("  最好单个模型 : %.2f%%" % (best_single * 100))
    print("  异源集成     : %.2f%%   (比最好单个 %+.2f)" % (ens * 100, (ens - best_single) * 100))
    print("=" * 56)
    print("  逐属性: " + " ".join("%.2f" % (a * 100) for a in per_attr))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "ensemble_mix.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc(%)"])
        for tag, a in sorted(singles.items(), key=lambda x: -x[1]):
            w.writerow([tag, "%.2f" % (a * 100)])
        w.writerow(["best_single", "%.2f" % (best_single * 100)])
        w.writerow(["ensemble_mix", "%.2f" % (ens * 100)])
        w.writerow([])
        w.writerow(["attr", "acc(%)"])
        for i in range(12):
            w.writerow([i + 1, "%.2f" % (per_attr[i] * 100)])
    print("  已保存 logs/ensemble_mix.csv")


if __name__ == "__main__":
    main()
