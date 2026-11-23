# EXP015.py —— 实验15: 换躯干之"更大型号" (ResNet-18 -> ConvNeXt-Small)
#   和 EXP010 的区别: EXP010 用的是最小的 ConvNeXt-Tiny (28M), 本实验用 ConvNeXt-Small (50M)
#   和 EXP008 的区别: 只换主干网络 (输入尺寸 448x224 完全一致)
#   为什么: 同学的"ConvNet-small/base"很可能就是 ConvNeXt-Small/Base, 他拿到 93.5~93.8;
#           而我们的 Tiny 输了 —— 所以不能排除"型号选小了"这个原因
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from dataset import make_train_val_loaders, ATTR_CLASSES
from tqdm import tqdm

# ---------- ① 配置区: 超参数 ----------
EPOCHS = 100
PATIENCE = 15     # 早停: 连续多少轮没有新纪录就停止训练
# ★ 批大小: 本地 8GB 显卡只能开 8; 云端 5090(32GB) 可以开 32, 又快又更接近 EXP008 的配置
BATCH_SIZE = 32
LR = 0.01
# 种子: 只有第一个参数是数字时才当种子
_ARG = sys.argv[1] if len(sys.argv) > 1 else ""
SEED = int(_ARG) if _ARG.isdigit() else 42
TAG = ("_s%d" % SEED) if _ARG.isdigit() else ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- ② 图片预处理 (和 EXP008 完全一致) ----------
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.Resize((512, 256)),          # 高512 宽256 (1:2)
    T.RandomCrop((448, 224)),      # 高448 宽224 (1:2)
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

val_tf = T.Compose([
    T.Resize((448, 224)),          # 高448 宽224 (1:2)
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

# ---------- ③ 模型: ★ ConvNeXt-Small ----------
class FFResNet(nn.Module):
    def __init__(self, num_classes_per_attr):
        super().__init__()
        # ★ EXP015 唯一的改动: convnext_tiny -> convnext_small
        self.backbone = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        # ConvNeXt-Small 输出特征维度也是 768 (和 Tiny 一样)
        self.heads = nn.ModuleList(
            [nn.Linear(768, c) for c in num_classes_per_attr]
        )
        self.cbam = CBAM(768)
    def forward(self, x):
        x = self.backbone.features(x)           # 卷积主干 -> [B, 768, 14, 7]
        x = self.cbam(x)                        # CBAM
        x = x.mean(dim=[2, 3])                  # 全局平均池化 -> [B, 768]
        return [head(x) for head in self.heads]


# ---------- CBAM 注意力模块 ----------
class ChannelAttention(nn.Module):
    """通道注意力: 给每个通道打个 0~1 的分, 重要的放大, 不重要的压小"""
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """空间注意力: 给画面每个位置打分, 重要位置放大"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True)[0]
        scale = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """先通道注意力, 再空间注意力 (串行)"""
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ---------- ④ 损失函数 ----------
def compute_loss(logits_list, labels):
    total_loss=0.0
    for i,logits in enumerate(logits_list):
        total_loss = total_loss + F.cross_entropy(logits, labels[:,i], label_smoothing=0.1)
    return total_loss / len(logits_list)

def evaluate(model,loader):
    model.eval()
    correct = [0] * 12
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)
            for i in range(12):
                pred = logits[i].argmax(dim=1)
                correct[i] = correct[i] + (pred == labels[:, i]).sum().item()
            total = total + labels.size(0)
    per_attr = [c / total for c in correct]
    mean_acc = sum(per_attr)/12
    return mean_acc, per_attr


# ---------- ⑤ 训练一个 epoch ----------
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    correct = [0] * 12
    total = 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(imgs)
        loss = compute_loss(logits, labels)
        for i in range(12):
            correct[i] = correct[i] + (logits[i].argmax(dim=1) == labels[:, i]).sum().item()
        total = total + labels.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
    train_acc = sum(c / total for c in correct) / 12
    return total_loss / len(loader), train_acc


# ---------- ⑥ 主程序 ----------
if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")
    torch.manual_seed(SEED)
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_PATH = os.path.join(LOG_DIR, os.path.splitext(os.path.basename(__file__))[0] + TAG + "_log.csv")
    log_f = open(LOG_PATH, "w", encoding="utf-8")
    log_f.write("epoch,train_loss,val_acc(%),is_best,train_acc(%)\n")
    model = FFResNet(ATTR_CLASSES).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    train_loader, val_loader = make_train_val_loaders(train_tf, val_tf, batch_size=BATCH_SIZE)

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        scheduler.step()
        val_acc, _ = evaluate(model, val_loader)
        is_best = val_acc > best_acc
        log_f.write(f"{epoch},{train_loss:.4f},{val_acc * 100:.2f},{int(is_best)},{train_acc * 100:.2f}\n")
        log_f.flush()
        print(f"Epoch {epoch}/{EPOCHS}  训练损失 {train_loss:.4f}  训练正确率 {train_acc * 100:.2f}%  验证准确率 {val_acc * 100:.2f}%  当前lr {optimizer.param_groups[0]['lr']:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "EXP015" + TAG + "_best.pt"))
            print("  ↑ 新纪录, 最优模型已存档")
        else:
            no_improve = no_improve + 1
            print(f"  (连续 {no_improve} 轮无提升)")
            if no_improve >= PATIENCE:
                print(f"连续 {PATIENCE} 轮没有提升, 提前停止 (最佳 epoch {best_epoch})")
                break

    _, per_attr = evaluate(model, val_loader)
    print("\n各属性验证准确率:", [f"{a * 100:.2f}%" for a in per_attr])
    print(f"最佳验证准确率(12属性平均): {best_acc * 100:.2f}%  (最佳 epoch {best_epoch})")
    log_f.write("# per_attr_acc(%): " + ",".join(f"{a * 100:.2f}" for a in per_attr) + "\n")
    log_f.write(f"# best={best_acc * 100:.2f}% at epoch {best_epoch}\n")
    log_f.close()
