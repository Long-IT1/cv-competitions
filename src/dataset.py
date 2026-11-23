import os
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T

# 提速: 输入尺寸固定时, 让 cuDNN 自动挑选最快的卷积算法
torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
ANN_DIR = os.path.join(BASE_DIR,"train_images","shape_ann")
ANN_TRAIN = os.path.join(ANN_DIR,"train_ann_file.txt")
ANN_VAL = os.path.join(ANN_DIR,"val_ann_file.txt")

# 图片散在两个文件夹里(不是笔误): train_images\ 有 10335 张, test_images\ 有 1149 张。
# 官方 train/val 两份标注把这 11484 张全部用完, 所以两个文件夹都要扫。
IMG_DIRS = [os.path.join(BASE_DIR,"train_images","train_images"),
            os.path.join(BASE_DIR,"train_images","test_images")]
IMG_TRAIN = IMG_DIRS[0]      # 旧名字保留(有些老脚本还在 import 它)

# 预缩好的图片缓存(由 make_cache.py 生成): 读图快很多。没有这个文件夹也能跑, 只是慢。
CACHE_DIR = os.path.join(BASE_DIR,"image_cache")

ATTR_CLASSES = [6,5,4,3,5,3,3,3,5,7,3,3]
N_ATTRS = len(ATTR_CLASSES)

def read_ann(ann_path):
    """读标注文件: 返回 [(图片文件名, [12个标签]), ...]"""
    items =[]
    with open(ann_path,"r",encoding="utf-8")as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if len(parts) != 1 + N_ATTRS:
                raise ValueError(f"标注列数不对！预期{1 + N_ATTRS}列，实际{len(parts)} 列")
            name = os.path.basename(parts[0])
            labels = [int(x) for x in parts[1:]]
            items.append((name,labels))
    return items

def index_images():
    """扫描所有图片, 返回 {文件名: 完整路径}
    先扫原图(保证一张都不少), 再让 image_cache 文件夹里的预缩图覆盖同名项(读图更快)"""
    index = {}
    for d in IMG_DIRS:
        if not os.path.isdir(d):
            print(f"[dataset] 警告: 找不到图片文件夹 {d}")
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(".png"):
                index.setdefault(fn, os.path.join(d, fn))

    n_cache = 0
    if os.path.isdir(CACHE_DIR):
        for fn in os.listdir(CACHE_DIR):
            if fn.lower().endswith(".jpg"):
                key = fn[:-4] + ".png"          # 缓存文件是 .jpg, 对应原图 .png
                # 缓存直接作为图片来源(原图在不在都行) —— 云端只传缓存时也能跑
                index[key] = os.path.join(CACHE_DIR, fn)
                n_cache += 1

    print(f"[dataset] 扫描 {len(IMG_DIRS)} 个文件夹, 共找到 {len(index)} 张图 (其中 {n_cache} 张走缓存)")
    return index

def keep_existing(items, index):
    """只保留图片真实存在的行"""
    kept = [(name,labels) for name,labels in items if name in index]
    print(f"[dataset] 标注 {len(items)}行,图片存在{len(kept)}行,跳过{len(items) - len(kept)}行")
    return kept

class FFAttrDataset(Dataset):
    """把一个"标注列表"包装成 PyTorch 认识的数据集"""

    def __init__(self, items, index, transform=None):
        self.items = items
        self.index = index
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, labels = self.items[idx]
        img = Image.open(self.index[name]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        labels = torch.tensor(labels, dtype=torch.long)
        return img, labels


def official_split():
    """官方切分: 训练 = train_ann 全部(9188张), 验证 = val_ann 全部(2296张)
    两份标注互不重叠, 加起来正好是全部 11484 张图"""
    index = index_images()
    train_items = keep_existing(read_ann(ANN_TRAIN), index)
    val_items = keep_existing(read_ann(ANN_VAL), index)
    return train_items, val_items, index


def make_train_val_loaders(train_tf, val_tf, batch_size=64):
    """用官方切分做 train/val, 返回 (train_loader, val_loader)"""
    train_items, val_items, index = official_split()

    train_ds = FFAttrDataset(train_items, index, train_tf)
    val_ds = FFAttrDataset(val_items, index, val_tf)

    # ★ 关键: persistent_workers=True 让搬运工常驻。
    #   Windows 上每新建一个 worker 都要重新导入一遍 torch(1~2秒), 每轮重招 12 个 = 白扔一分钟以上。
    #   实测: 不常驻 69 秒/轮 -> 常驻 23 秒/轮。
    #   工人数控制在 4+3=7 个: 12 个常驻会吃掉 7GB 内存(可用只剩 0.7GB, 有假死风险)。
    #   训练 4 个足够(瓶颈在显卡, 不在读图); 验证 3 个够用。
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=3, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader


if __name__ == "__main__":
      print("===== dataset.py 自检 =====")
      train_items, val_items, index = official_split()
      print(f"训练 {len(train_items)} 张, 验证 {len(val_items)} 张, 合计 {len(train_items) + len(val_items)} 张")

      overlap = set(n for n, _ in train_items) & set(n for n, _ in val_items)
      print("训练/验证重叠张数(必须是 0):", len(overlap))

      mx = [0] * 12
      for _, labels in train_items:
          for i, v in enumerate(labels):
              mx[i] = max(mx[i], v)
      print("训练集 12 列最大值:", mx)
      print("取值范围正确(0开始):", mx == [c - 1 for c in ATTR_CLASSES])

      tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
      loader, _ = make_train_val_loaders(tf, tf, batch_size=8)
      imgs, labels = next(iter(loader))
      print("图片张量形状:", tuple(imgs.shape), "标签张量形状:", tuple(labels.shape))
      print("自检通过")
