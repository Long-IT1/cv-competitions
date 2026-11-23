# -*- coding: utf-8 -*-
# probe_prior_correction.py —— 免费侦察(只读, 不训练、不改任何现有文件)
#   问题: 误差分析发现"稀有类被大流吞"(属7类2: 29张全错)。不训练能不能救?
#   方法: 推理端"先验修正" —— 把每个类的概率按稀有程度掰一下再取最大:
#           p'_c = p_c / prior_c^τ        (τ=0 就是原样; τ 越大越照顾稀有类)
#   prior 用"训练集"的类别频率(不用验证集, 防作弊)
#   得到两个数:
#     ① 全局最优 τ  → 老实版(一个 τ 管所有属性, 仍然是在验证集上挑的, 略乐观)
#     ② 每属性单独 τ → 上限版(明牌注明是上限)
#   如果连上限都没肉 → "训练版平衡损失"也不用打了; 有肉 → 再去训练
#   产出: logs/ens_probs.npz(概率存档, 以后分析复用), logs/prior_probe.csv
import os
import sys
import importlib
import numpy as np
import torch
import torch.nn.functional as F
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
NPZ = os.path.join(HERE, "logs", "ens_probs.npz")
TAUS = [round(t, 2) for t in np.arange(0, 1.05, 0.05)] + [1.25, 1.5]


def get_probs_and_labels():
    """有存档就加载, 没有就现场算(5 模型 x TTA)并存档"""
    _, val_items, index = ds.official_split()
    N = len(val_items)
    if os.path.exists(NPZ):
        z = np.load(NPZ)
        probs = [z["p%d" % i] for i in range(K)]
        yt = z["labels"]
        print("已加载概率存档 logs/ens_probs.npz")
        return val_items, probs, yt

    ds_val = ds.FFAttrDataset(val_items, index, exp.val_tf)
    loader = torch.utils.data.DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=3)
    prob_sum = [torch.zeros(N, c) for c in ATTR]
    yt_t = torch.zeros(N, K, dtype=torch.long)
    for tag in TAGS:
        model = exp.FFResNet(ATTR).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(HERE, tag + "_best.pt"), map_location="cpu"))
        model.eval()
        pos = 0
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="  " + tag, leave=False):
                imgs = imgs.to(DEVICE)
                l1 = model(imgs)
                l2 = model(torch.flip(imgs, dims=[3]))
                b = imgs.size(0)
                for i in range(K):
                    p = (F.softmax(l1[i], 1) + F.softmax(l2[i], 1)) / 2
                    prob_sum[i][pos:pos + b] += p.cpu()
                if tag == TAGS[0]:
                    yt_t[pos:pos + b] = labels
                pos += b
        del model
        torch.cuda.empty_cache()
        print("  %s 跑完" % tag)
    probs = [p.numpy() / len(TAGS) for p in prob_sum]
    yt = yt_t.numpy()
    np.savez_compressed(NPZ, labels=yt, **{"p%d" % i: probs[i] for i in range(K)})
    print("概率已存档: logs/ens_probs.npz")
    return val_items, probs, yt


def acc_of(probs, yt, priors, tau):
    """按 τ 修正后逐属性准确率"""
    acc = np.zeros(K)
    for i in range(K):
        pr = np.power(priors[i], tau)[None, :] + 1e-12
        pred = (probs[i] / pr).argmax(1)
        acc[i] = (pred == yt[:, i]).mean()
    return acc


def main():
    print("===== 先验修正侦察 (免费, 不训练) =====")
    train_items, val_items, index = ds.official_split()
    _, probs, yt = get_probs_and_labels()
    N = yt.shape[0]

    # 训练集类别频率(拉普拉斯平滑, 防 0)
    priors = []
    for i in range(K):
        cnt = np.bincount([lab[i] for _, lab in train_items], minlength=ATTR[i]).astype(np.float64)
        priors.append((cnt + 1) / (cnt.sum() + ATTR[i]))

    base = acc_of(probs, yt, priors, 0.0)
    print("\n基线(τ=0): %.2f%%   (存档 94.07%%, 对上就没算错)" % (base.mean() * 100))

    # ---- 全局 τ 扫描 ----
    print("\n---- 全局 τ 扫描 ----")
    print(" τ     总分      相对基线")
    best_tau, best_mean, rows = 0.0, base.mean(), []
    for tau in TAUS:
        a = acc_of(probs, yt, priors, tau)
        m = a.mean()
        rows.append((tau, m))
        if m > best_mean:
            best_tau, best_mean = tau, m
        print(" %.2f   %.2f%%   %+.2f" % (tau, m * 100, (m - base.mean()) * 100))

    a_best = acc_of(probs, yt, priors, best_tau)
    print("\n全局最优: τ=%.2f → %.2f%%  (比基线 %+.2f)" % (best_tau, best_mean * 100, (best_mean - base.mean()) * 100))
    print("它把每个属性从→变成:")
    for i in range(K):
        d = (a_best[i] - base[i]) * 100
        flag = " ★" if abs(d) >= 0.2 else ""
        print("  属性%-2d %5.2f%% → %5.2f%%  (%+.2f)%s" % (i + 1, base[i] * 100, a_best[i] * 100, d, flag))

    # ---- 每属性单独调 τ (上限版, 明牌乐观) ----
    print("\n---- 每属性单独挑 τ (上限版, 在验证集上挑的, 偏乐观) ----")
    per_best = np.zeros(K)
    per_tau = np.zeros(K)
    for i in range(K):
        for tau in TAUS:
            a = acc_of(probs, yt, priors, tau)[i]
            if a > per_best[i]:
                per_best[i], per_tau[i] = a, tau
    print("上限总分: %.2f%%  (比基线 %+.2f)" % (per_best.mean() * 100, (per_best.mean() - base.mean()) * 100))
    for i in range(K):
        if per_tau[i] > 0:
            print("  属性%-2d τ*=%.2f: %5.2f%% → %5.2f%%" % (i + 1, per_tau[i], base[i] * 100, per_best[i] * 100))

    # ---- 稀有类召回: 修正前后 ----
    print("\n---- 稀有类的召回变化 (全局最优 τ=%.2f) ----")
    for i in range(K):
        cnt = np.bincount([lab[i] for _, lab in train_items], minlength=ATTR[i])
        rare = [c for c in range(ATTR[i]) if cnt[c] < 100]
        if not rare:
            continue
        pr = np.power(priors[i], best_tau)[None, :] + 1e-12
        pred_adj = (probs[i] / pr).argmax(1)
        pred_base = probs[i].argmax(1)
        parts = []
        for c in rare:
            m = yt[:, i] == c
            if m.sum() == 0:
                continue
            parts.append("类%d(%d张): %.0f%%→%.0f%%" % (
                c, m.sum(), (pred_base[m] == c).mean() * 100, (pred_adj[m] == c).mean() * 100))
        print("  属性%-2d 稀有类<100张: %s" % (i + 1, "  ".join(parts) if parts else "无"))

    with open(os.path.join(HERE, "logs", "prior_probe.csv"), "w", encoding="utf-8") as f:
        f.write("tau,mean_acc(%%)\n")
        for tau, m in rows:
            f.write("%.2f,%.4f\n" % (tau, m * 100))
        f.write("per_attr_best,%.4f\n" % (per_best.mean() * 100))
    print("\n已存: logs/prior_probe.csv")
    print("===== 侦察完毕 =====")


if __name__ == "__main__":
    main()
