# EXP004.py —— 实验4: 温和增强 + 余弦退火 + ImageNet预训练 (与 EXP003 唯一差别: weights)
import os
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
BATCH_SIZE = 64
LR = 0.01
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  # ---------- ② 图片预处理 ----------
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

val_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

  # ---------- ③ 模型: ResNet-18 换头 ----------
class FFResNet(nn.Module):
    def __init__(self, num_classes_per_attr):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleList(
            [nn.Linear(512, c) for c in num_classes_per_attr]
        )
    def forward(self, x):
        feat = self.backbone(x)
        return [head(feat) for head in self.heads]
    
   # ---------- ④ 损失函数: 12 个属性各算一笔账 ----------
def compute_loss(logits_list, labels):
    total_loss=0.0
    for i,logits in enumerate(logits_list):
        total_loss = total_loss + F.cross_entropy(logits,labels[:,i])
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
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(imgs)
        loss = compute_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
    return total_loss / len(loader)


# ---------- ⑦ 主程序 ----------
if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")
    torch.manual_seed(SEED)
    # ---- 训练日志: 每个 epoch 自动写入 logs/EXPxxx_log.csv ----
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_PATH = os.path.join(LOG_DIR, os.path.splitext(os.path.basename(__file__))[0] + "_log.csv")
    log_f = open(LOG_PATH, "w", encoding="utf-8")
    log_f.write("epoch,train_loss,val_acc(%),is_best\n")
    model = FFResNet(ATTR_CLASSES).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    train_loader, val_loader = make_train_val_loaders(train_tf, val_tf, batch_size=BATCH_SIZE)

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        scheduler.step()
        val_acc, _ = evaluate(model, val_loader)
        is_best = val_acc > best_acc
        log_f.write(f"{epoch},{train_loss:.4f},{val_acc * 100:.2f},{int(is_best)}\n")
        log_f.flush()
        print(f"Epoch {epoch}/{EPOCHS}  训练损失 {train_loss:.4f}  验证准确率{val_acc * 100:.2f}%  当前lr {optimizer.param_groups[0]['lr']:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "EXP004_best.pt"))
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
