# 临时脚本: 跑两轮训练验证最终配置(4+3 常驻工人)的速度和内存
# 用法(在 exp 目录下): python bench_epoch.py
import time, ctypes
import torch
import EXP006 as E
from dataset import make_train_val_loaders, ATTR_CLASSES


class _M(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def mem(tag):
    m = _M(); m.dwLength = ctypes.sizeof(_M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    print("%-22s RAM free %4.1fGB (load %d%%) | commit avail %4.1fGB"
          % (tag, m.ullAvailPhys / 2**30, m.dwMemoryLoad, m.ullAvailPageFile / 2**30), flush=True)


if __name__ == "__main__":
    tl, vl = make_train_val_loaders(E.train_tf, E.val_tf, batch_size=E.BATCH_SIZE)
    model = E.FFResNet(ATTR_CLASSES).to(E.DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=E.LR, momentum=0.9, weight_decay=5e-4)

    mem("启动前")
    t0 = time.time(); E.train_one_epoch(model, tl, opt); t1 = time.time()
    E.evaluate(model, vl); t2 = time.time()
    print("第 1 轮: 训练 %.1fs + 验证 %.1fs = %.1fs" % (t1 - t0, t2 - t1, t2 - t0), flush=True)
    mem("工人就位后")

    t3 = time.time(); E.train_one_epoch(model, tl, opt); t4 = time.time()
    E.evaluate(model, vl); t5 = time.time()
    print("第 2 轮: 训练 %.1fs + 验证 %.1fs = %.1fs   <-- 真实速度" % (t4 - t3, t5 - t4, t5 - t3), flush=True)
    print("=> 三个实验约 %.1f 小时 (290 轮)" % (290 * (t5 - t3) / 3600), flush=True)
    mem("第 2 轮结束")
