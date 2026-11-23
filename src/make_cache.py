# 把 11484 张原图预缩成小尺寸存一份, 训练时读图快很多
# 只读原图, 只往新文件夹 image_cache\ 里写, 不碰任何原文件
# 断点续跑: 已经缩好的会自动跳过, 中途停了重新运行即可
# 用法(在 exp 目录下): python make_cache.py
import os, time
from concurrent.futures import ProcessPoolExecutor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IMG_DIRS = [os.path.join(BASE_DIR, "train_images", "train_images"),
            os.path.join(BASE_DIR, "train_images", "test_images")]
CACHE_DIR = os.path.join(BASE_DIR, "image_cache")

SIZE = (384, 768)     # 按原始长宽比缩到这个框内 (原图 512x1024 -> 正好 384x768)
QUALITY = 95          # JPEG 质量
N_WORKERS = 6         # 用 6 个进程并行 (和训练一样, 别加太多)


def collect():
    names = []
    for d in IMG_DIRS:
        if not os.path.isdir(d):
            print("!! 找不到文件夹:", d)
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(".png"):
                names.append((fn, os.path.join(d, fn)))
    return names


def convert(job):
    fn, src = job
    dst = os.path.join(CACHE_DIR, os.path.splitext(fn)[0] + ".jpg")
    if os.path.exists(dst):
        return 0                       # 已经缩过了, 跳过
    from PIL import Image
    img = Image.open(src).convert("RGB")
    img.thumbnail(SIZE, Image.LANCZOS)      # 保持长宽比缩小
    img.save(dst, "JPEG", quality=QUALITY, subsampling=0)
    return 1


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    jobs = collect()
    print("原图 %d 张 -> 缩到 %s 以内 -> %s" % (len(jobs), SIZE, CACHE_DIR), flush=True)
    t0 = time.time()
    newly = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(convert, jobs, chunksize=16), 1):
            newly += r
            if i % 1500 == 0 or i == len(jobs):
                el = time.time() - t0
                print("  %d/%d  新缩 %d 张  已用 %.0fs  预计还要 %.0fs"
                      % (i, len(jobs), newly, el, el / i * (len(jobs) - i)), flush=True)
    total = len([f for f in os.listdir(CACHE_DIR) if f.lower().endswith(".jpg")])
    print("完成: 本次新缩 %d 张, 缓存目录共 %d 张 (原图 %d 张)" % (newly, total, len(jobs)), flush=True)
    print("OK" if total >= len(jobs) else "!! 数量对不上, 别急着用缓存", flush=True)


if __name__ == "__main__":
    main()
