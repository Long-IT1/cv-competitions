# 用"官方验证集"(val_ann_file.txt)评估某个实验的存档模型
# 只读脚本: 不改动任何现有文件, 结果只打印在屏幕上
# 用法(在 exp 目录下): python eval_official.py EXP006
import sys, os, re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import read_ann, ATTR_CLASSES

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANN_DIR = os.path.join(BASE, "train_images", "shape_ann")
ANN_TRAIN = os.path.join(ANN_DIR, "train_ann_file.txt")
ANN_VAL = os.path.join(ANN_DIR, "val_ann_file.txt")
# 图片散在两个文件夹里: 大部分在 train_images\, 另有 1149 张在 test_images\
IMG_DIRS = [os.path.join(BASE, "train_images", "train_images"),
            os.path.join(BASE, "train_images", "test_images")]


def index_images():
    """扫描所有图片文件夹, 返回 {文件名: 完整路径}"""
    idx, dup = {}, []
    for d in IMG_DIRS:
        if not os.path.isdir(d):
            print("!! 找不到图片文件夹:", d)
            continue
        for fn in os.listdir(d):
            if not fn.lower().endswith(".png"):
                continue
            if fn in idx:
                dup.append(fn)
            else:
                idx[fn] = os.path.join(d, fn)
    print("[索引] 扫描 %d 个文件夹, 共找到 %d 张图 (重名 %d)" % (len(IMG_DIRS), len(idx), len(dup)))
    return idx


def garment_id(name):
    m = re.search(r"id_(\d+)", name)
    return m.group(1) if m else name


class IndexedDataset(Dataset):
    """按"文件名->完整路径"的索引取图, 这样两个文件夹里的图都能用"""

    def __init__(self, items, index, transform):
        self.items, self.index, self.transform = items, index, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        name, labels = self.items[k]
        img = Image.open(self.index[name]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(labels, dtype=torch.long)


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "EXP006"
    mod = __import__(exp)

    idx = index_images()
    train_all = read_ann(ANN_TRAIN)
    val_all = read_ann(ANN_VAL)
    tr_items = [it for it in train_all if it[0] in idx]
    va_items = [it for it in val_all if it[0] in idx]
    print("[数据] 官方 train: 标注 %d 行, 图片能找到 %d 张" % (len(train_all), len(tr_items)))
    print("[数据] 官方 val  : 标注 %d 行, 图片能找到 %d 张" % (len(val_all), len(va_items)))

    tr_ids = set(garment_id(n) for n, _ in tr_items)
    va_ids = [garment_id(n) for n, _ in va_items]
    ov = sum(1 for g in va_ids if g in tr_ids)
    print("[体检] 官方 val 里与 train 撞'衣服ID'的: %d / %d = %.1f%%"
          % (ov, len(va_items), 100.0 * ov / max(1, len(va_items))))
    print("[体检] 唯一衣服ID: train %d 个, val %d 个" % (len(tr_ids), len(set(va_ids))))

    model = mod.FFResNet(ATTR_CLASSES).to(mod.DEVICE)
    model.load_state_dict(torch.load(exp + "_best.pt", map_location=mod.DEVICE))
    print("[模型] %s_best.pt 已加载" % exp)

    loader = DataLoader(IndexedDataset(va_items, idx, mod.val_tf),
                        batch_size=64, shuffle=False, num_workers=0)
    acc, per_attr = mod.evaluate(model, loader)
    print("=" * 60)
    print("%s 在【官方验证集】%d 张上的准确率: %.2f%%" % (exp, len(va_items), acc * 100))
    print("逐属性: " + ", ".join("%.2f" % (a * 100) for a in per_attr))
    print("(对照: 该模型在自切 825 张 val 上的成绩见 logs/)")
    print("=" * 60)


if __name__ == "__main__":
    main()
