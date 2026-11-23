# 临时测速脚本: 找出数据读取到底慢在哪
#   对比 0 个工人 / 6 个工人 / 6 个工人且常驻(persistent_workers) 三种情况
#   并单独看"第一个批次"(含工人启动开销) 和"后续批次"(纯吞吐) 的差别
# 用法(在 exp 目录下): python bench_loader.py
import os, time
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import official_split, FFAttrDataset

tf = T.Compose([T.Resize((256, 256)), T.RandomCrop(224), T.RandomHorizontalFlip(),
                T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def make_loader(train_items, idx, nw, persist):
    ds = FFAttrDataset(train_items, idx, tf)
    return DataLoader(ds, batch_size=64, shuffle=True, num_workers=nw,
                      pin_memory=True, persistent_workers=(persist and nw > 0))


def run(tag, dl, warm, n):
    it = iter(dl)
    t0 = time.time()
    for _ in range(warm):
        next(it)
    t1 = time.time()
    for _ in range(n):
        next(it)
    t2 = time.time()
    print("%s | 启动+%d批 %5.1fs | 之后%d批 %5.1fs | %5.0f img/s | 训练集一遍 %4.0f 秒"
          % (tag, warm, t1 - t0, n, t2 - t1, n * 64 / (t2 - t1), 9188 / (n * 64 / (t2 - t1))),
          flush=True)
    del it, dl


if __name__ == "__main__":
    train_items, val_items, index = official_split()
    print("train=%d val=%d" % (len(train_items), len(val_items)), flush=True)

    run("workers=0          ", make_loader(train_items, index, 0, False), 1, 10)
    run("workers=6          ", make_loader(train_items, index, 6, False), 1, 20)
    run("workers=6 常驻     ", make_loader(train_items, index, 6, True), 1, 20)
