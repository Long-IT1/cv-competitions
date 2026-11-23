# 评估某个实验已存档的最优模型 -> 打印 12 属性准确率, 并追加写入 logs/EXPxxx_log.csv
# 用法(在 exp 目录下):  python eval_best.py EXP001
import sys, os, random
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import read_ann, keep_existing, FFAttrDataset, ANN_TRAIN, IMG_TRAIN, ATTR_CLASSES


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "EXP001"
    mod = __import__(exp)

    model = mod.FFResNet(ATTR_CLASSES).to(mod.DEVICE)
    model.load_state_dict(torch.load(exp + "_best.pt", map_location=mod.DEVICE))

    # 复刻 make_train_val_loaders 的切分(seed=42, 前10%验证), 只建验证集且不开多进程
    items = keep_existing(read_ann(ANN_TRAIN), IMG_TRAIN)
    idx = list(range(len(items)))
    random.Random(42).shuffle(idx)
    n_val = int(len(items) * 0.1)
    val_ds = FFAttrDataset([items[i] for i in idx[:n_val]], IMG_TRAIN, mod.val_tf)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    acc, per = mod.evaluate(model, val_loader)
    per_s = ",".join("%.2f" % (a * 100) for a in per)
    print(exp, "per_attr_acc(%):", per_s)
    print(exp, "mean_acc(%):", "%.2f" % (acc * 100))

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", exp + "_log.csv")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("# per_attr_acc(%): " + per_s + "\n")
        f.write("# 复评来源: " + exp + "_best.pt\n")


if __name__ == "__main__":
    main()
