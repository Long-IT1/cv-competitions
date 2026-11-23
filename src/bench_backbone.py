# 临时测速: 直接在显卡上量各"躯干"的真实训练速度(不管读图, 只看显卡算得多快)
# 用法(在 exp 目录下): python bench_backbone.py
import time, torch
from torchvision import models

DEVICE = "cuda"
BATCH = 16
WARM, N = 2, 5


def step(m, x, opt):
    out = m(x)
    if isinstance(out, (list, tuple)):
        loss = sum(t.float().mean() for t in out)
    else:
        loss = out.float().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()


def bench(name, build, size, batch=BATCH):
    try:
        m = build().to(DEVICE)
        m.train()
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        x = torch.randn(batch, 3, size[0], size[1], device=DEVICE)
        for _ in range(WARM):
            step(m, x, opt)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N):
            step(m, x, opt)
        torch.cuda.synchronize()
        el = (time.time() - t0) / N
        per_sec = batch / el
        print("  %-14s %-9s 每批 %5.2f 秒 => %6.0f 张/秒 => 9188张/轮 约 %5.1f 分钟"
              % (name, "%dx%d" % size, el, per_sec, 9188 / per_sec / 60))
        del m, x
        torch.cuda.empty_cache()
    except Exception as e:
        print("  %-14s %-9s 失败: %s" % (name, "%dx%d" % size, str(e)[:70]))


print("===== 躯干速度实测 (批大小 16, 只看显卡计算) =====")
print()
print("-- ResNet-18 (我们一直在用的) --")
for sz in [(224, 224), (448, 224), (640, 320)]:
    bench("resnet18", lambda: models.resnet18(weights=None), sz)
print()
print("-- ConvNeXt-Tiny (想换的) --")
for sz in [(224, 224), (448, 224)]:
    bench("convnext_tiny", lambda: models.convnext_tiny(weights=None), sz)
print()
print("注: 随机权重, 只看速度")
