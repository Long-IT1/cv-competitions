# TTA 测试: 同一张图"原图 + 左右翻转"各预测一次, 12个属性的概率取平均再判类
# 用法(exp 目录下): python eval_tta.py EXP005
# 输出: 屏幕上的普通 vs TTA 对比 + logs/EXPxxx_tta.csv
import sys, os, csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import official_split, FFAttrDataset, ATTR_CLASSES


def evaluate_both(model, loader, device):
    """跑一遍数据, 同时算两个分数: 普通预测 & TTA(原图+翻转概率平均)
    返回 (普通逐属性准确率, TTA逐属性准确率), 每项都是 0~1 的小数"""
    model.eval()
    c_plain = [0] * 12
    c_tta = [0] * 12
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)                              # 第一遍: 原图
            logits_f = model(torch.flip(imgs, dims=[3]))      # 第二遍: 左右镜像
            for i in range(12):
                c_plain[i] += (logits[i].argmax(1) == labels[:, i]).sum().item()
                prob = (F.softmax(logits[i], 1) + F.softmax(logits_f[i], 1)) / 2
                c_tta[i] += (prob.argmax(1) == labels[:, i]).sum().item()
            total += labels.size(0)
    return [c / total for c in c_plain], [c / total for c in c_tta]


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "EXP005"
    mod = __import__(exp)

    model = mod.FFResNet(ATTR_CLASSES).to(mod.DEVICE)
    model.load_state_dict(torch.load(exp + "_best.pt", map_location=mod.DEVICE))

    # 用官方验证集(和 eval_metrics.py 是同一批图, 数字可以直接对比)
    _, val_items, index = official_split()
    val_ds = FFAttrDataset(val_items, index, mod.val_tf)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    print("验证集共", len(val_items), "张")

    plain, tta = evaluate_both(model, val_loader, mod.DEVICE)
    p_mean = sum(plain) / 12 * 100
    t_mean = sum(tta) / 12 * 100

    print("=" * 50)
    print("%s 普通预测      : %.2f%%" % (exp, p_mean))
    print("%s TTA(原图+翻转): %.2f%%" % (exp, t_mean))
    print("TTA 净赚        : %+.2f" % (t_mean - p_mean))
    print("-" * 50)
    print("逐属性对比(普通 -> TTA):")
    for i in range(12):
        print("  属性%2d: %5.2f%% -> %5.2f%%  (%+.2f)"
              % (i + 1, plain[i] * 100, tta[i] * 100, (tta[i] - plain[i]) * 100))

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", exp + "_tta.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attr", "plain(%)", "tta(%)", "diff"])
        for i in range(12):
            w.writerow([i + 1, "%.2f" % (plain[i] * 100), "%.2f" % (tta[i] * 100),
                        "%+.2f" % ((tta[i] - plain[i]) * 100)])
        w.writerow(["mean", "%.2f" % p_mean, "%.2f" % t_mean, "%+.2f" % (t_mean - p_mean)])
    print("已保存 logs/%s_tta.csv" % exp)


if __name__ == "__main__":
    main()
