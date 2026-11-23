# 评估某实验最优模型的"训练正确率 + 验证正确率"(12属性平均),
# 并把结果追加/更新到 logs/metrics_summary.csv (指标汇总表)
# 用法(exp 目录下): python eval_metrics.py EXP001
import sys, os, csv, datetime
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import official_split, FFAttrDataset, ATTR_CLASSES


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "EXP001"
    mod = __import__(exp)

    model = mod.FFResNet(ATTR_CLASSES).to(mod.DEVICE)
    model.load_state_dict(torch.load(exp + "_best.pt", map_location=mod.DEVICE))

    # 官方切分: 训练 = train_ann 全部(9188张), 验证 = val_ann 全部(2296张)
    train_items, val_items, index = official_split()

    train_ds = FFAttrDataset(train_items, index, mod.val_tf)
    val_ds = FFAttrDataset(val_items, index, mod.val_tf)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    tr_acc, _ = mod.evaluate(model, train_loader)
    va_acc, _ = mod.evaluate(model, val_loader)
    print(exp, "训练正确率: %.2f%%" % (tr_acc * 100))
    print(exp, "验证正确率: %.2f%%" % (va_acc * 100))

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "metrics_summary.csv")
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) >= 5 and r[0] != "exp":
                    rows[r[0]] = r
    rows[exp] = [exp, "%.2f" % (tr_acc * 100), "%.2f" % (va_acc * 100),
                 "待定", datetime.date.today().isoformat()]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp", "train_acc(%)", "val_acc(%)", "test_acc(%)", "date"])
        for k in sorted(rows):
            w.writerow(rows[k])
    print("已写入 logs/metrics_summary.csv")


if __name__ == "__main__":
    main()
