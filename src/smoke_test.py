# 冒烟测试: 不训练, 只验证"原图 -> 预处理 -> 模型 -> 12个头"这条链走得通
# 新实验写好之后跑一下, 免得正式训练跑一半才发现尺寸对不上
# 用法(在 exp 目录下): python smoke_test.py EXP007 EXP008
import sys
import torch
from PIL import Image
from dataset import official_split, ATTR_CLASSES


def smoke(exp_name):
    mod = __import__(exp_name)
    print("=== %s ===" % exp_name)
    train_items, val_items, index = official_split()
    name = train_items[0][0]
    im = Image.open(index[name]).convert("RGB")
    print("  原图 size (宽,高):", im.size)

    t = mod.train_tf(im)
    v = mod.val_tf(im)
    print("  train_tf -> 张量形状", tuple(t.shape))
    print("  val_tf   -> 张量形状", tuple(v.shape))

    names = [nm for nm, _ in val_items[:4]]
    batch = torch.stack([mod.val_tf(Image.open(index[nm]).convert("RGB")) for nm in names]).to(mod.DEVICE)
    model = mod.FFResNet(ATTR_CLASSES).to(mod.DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    print("  输入 batch:", tuple(batch.shape))
    print("  12 个头输出:", [tuple(o.shape) for o in out])
    print("  OK - 这条链走得通\n")


if __name__ == "__main__":
    for e in (sys.argv[1:] or ["EXP007", "EXP008"]):
        smoke(e)
