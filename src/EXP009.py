# EXP009.py —— 实验9: 两张牌叠起来 (分辨率 320 + 长宽比 1:2 同时拉满)
#   与 EXP008 的唯一差别: 输入从 448x224 提到 640x320 (像素再翻一倍)
#   其余一切照抄 EXP008 (温和增强+余弦退火+预训练+CBAM+LabelSmoothing)
#   目的: 分辨率和长宽比单独用都有效(分别 +1.63 / +0.47), 叠起来看能不能再有额外收益
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
# ★ 批大小必须减半: 640x320 的像素是 224x224 的 4 倍, 批 64 会爆 8GB 显存
BATCH_SIZE = 32
LR = 0.01
# 种子: 只有第一个参数是数字时才当种子 (如 python EXP009.py 3);
#       不带参数、或参数不是数字(比如被 eval_metrics.py 当模块导入)时, 一律用 42
_ARG = sys.argv[1] if len(sys.argv) > 1 else ""
SEED = int(_ARG) if _ARG.isdigit() else 42
TAG = ("_s%d" % SEED) if _ARG.isdigit() else ""   # 多种子跑时给文件名加后缀, 互不覆盖
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- ② 图片预处理 ----------
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ★ EXP009 唯一的改动: 448x224 -> 640x320 (长宽比还是 1:2, 但看得更清)
#   注意 torchvision 的 Resize/RandomCrop 收的是 (高, 宽)
train_tf = T.Compose([
    T.Resize((768, 384)),          # 高768 宽384 (1:2), 比目标大一圈以便随机裁剪
    T.RandomCrop((640, 320)),      # 高640 宽320 (1:2), 轻微随机平移
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

val_tf = T.Compose([
    T.Resize((640, 320)),          # 高640 宽320 (1:2)
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

# ---------- ③ 模型: ResNet-18 换头 ----------
class FFResNet(nn.Module):
    def __init__(self, num_classes_per_attr):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()   # 出厂头弃用: 我们手动取下层的特征图
        self.heads = nn.ModuleList(
            [nn.Linear(512, c) for c in num_classes_per_attr]
        )
        self.cbam = CBAM(512)
    def forward(self, x):
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x)))          # stem(第一层卷积)
        x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x)       # 卷积身体(逐级下采样)
        x = b.layer3(x); x = b.layer4(x)       # 最后输出特征图 [B, 512, 20, 10] (640x320输入)
        x = self.cbam(x)                        # CBAM 在特征图上划重点
        x = x.mean(dim=[2, 3])                  # 全局平均池化 -> [B, 512]
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
        scale = self.sigmoid(avg_out + max_out)     # [B, C] 每个通道一个分数
        return x * scale.view(b, c, 1, 1)           # 分数乘回原特征


class SpatialAttention(nn.Module):
    """空间注意力: 给画面每个位置打分, 重要位置放大"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)   # 7×7 大卷积看全局
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)        # [B,1,H,W] 通道方向取平均
        max_map = x.max(dim=1, keepdim=True)[0]      # [B,1,H,W] 通道方向取最大
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


# ---------- ④ 损失函数: 12 个属性各算一笔账 ----------
def compute_loss(logits_list, labels):
    total_loss=0.0
    for i,logits in enumerate(logits_list):
        # label_smoothing=0.1 (从 EXP006 继承过来)
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
    correct = [0] * 12      # 数一数训练集答对了多少个
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
    # ---- 训练日志: 每个 epoch 自动写入 logs/EXPxxx_log.csv ----
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
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "EXP009" + TAG + "_best.pt"))
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
