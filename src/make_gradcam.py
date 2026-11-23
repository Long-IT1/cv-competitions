# -*- coding: utf-8 -*-
# make_gradcam.py —— Grad-CAM 注意力热力图: 模型判断每个属性时"眼睛在看哪"
#   用本地最强的 EXP009 (ResNet-18 + CBAM, 640x320) 对 2 张示例图 x 4 个属性画热力图
#   原理: 目标类别的 logit 对 CBAM 输出特征图求梯度, 梯度做权重对特征图加权求和
#         -> 红色=对判断贡献大的区域
#   只读, 不改任何现有文件。输出: figures/gradcam.png
import os
import sys
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)

import dataset as ds
exp = importlib.import_module("EXP009")

DEVICE = exp.DEVICE
OUT = os.path.join(os.path.dirname(HERE), "figures")
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

SHOW_ATTRS = [5, 7, 8, 9]      # 0 基下标 -> 属性 6 / 8 / 9 / 10 (最需要解释的四个)


def load_model():
    model = exp.FFResNet(ds.ATTR_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(HERE, "EXP009_best.pt"), map_location="cpu"))
    model.eval()
    return model


def gradcam(model, img_t, attr):
    """对单个属性, 用预测出的那个类做 Grad-CAM"""
    x = img_t.unsqueeze(0).to(DEVICE)
    with torch.enable_grad():
        b = model.backbone
        f = b.relu(b.bn1(b.conv1(x)))
        f = b.maxpool(f)
        f = b.layer1(f); f = b.layer2(f); f = b.layer3(f); f = b.layer4(f)
        f = model.cbam(f)                 # 特征图 [1, 512, 20, 10]
        f.retain_grad()
        pooled = f.mean(dim=[2, 3])
        logit = model.heads[attr](pooled)          # [1, C]
        cls = int(logit.argmax(1).item())
        model.zero_grad()
        logit[0, cls].backward()
        w = f.grad.mean(dim=[2, 3], keepdim=True)          # 梯度的全局平均 = 通道权重
        cam = F.relu((w * f).sum(1, keepdim=True))          # 加权求和 + ReLU
        cam = F.interpolate(cam, size=(640, 320), mode="bilinear", align_corners=False)
        conf = float(F.softmax(logit, 1)[0, cls])
        cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, cls, conf


def show_img(img_t):
    a = img_t.cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return np.clip(a, 0, 1)


def pick_name(keyword, val_items):
    for name, _ in val_items:
        if keyword in name:
            return name
    return val_items[0][0]


def main():
    model = load_model()
    _, val_items, index = ds.official_split()
    picks = [pick_name("WOMEN-Dresses", val_items), pick_name("WOMEN-Blouses_Shirts", val_items)]
    print("示例图:", picks)

    cols = 1 + len(SHOW_ATTRS)
    fig, axes = plt.subplots(2, cols, figsize=(3.0 * cols, 7.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for r, name in enumerate(picks):
        img = Image.open(index[name]).convert("RGB")
        img_t = exp.val_tf(img)
        axes[r, 0].imshow(show_img(img_t))
        axes[r, 0].set_title(name[:28] + "\n原图", fontsize=8.5, color=INK)
        axes[r, 0].axis("off")

        for c0, attr in enumerate(SHOW_ATTRS):
            cam, cls, conf = gradcam(model, img_t, attr)
            ax = axes[r, c0 + 1]
            ax.imshow(show_img(img_t))
            ax.imshow(cam, cmap="Reds", alpha=0.55)      # 单色红: 越红=贡献越大
            chinese = attr + 1
            ax.set_title("属性%d\n预测: 类%d (%.0f%%)" % (chinese, cls, conf * 100),
                         fontsize=8.5, color=INK)
            ax.axis("off")

    fig.suptitle("Grad-CAM 注意力热力图(EXP009, 640x320)—— 红色=模型判断该属性时的关注区域",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "gradcam.png"), facecolor=SURFACE)
    print("已生成: figures/gradcam.png")


if __name__ == "__main__":
    main()
