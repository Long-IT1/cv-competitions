# 训练进度监视器: 在另一个终端运行, 每 5 秒刷新一次
# 用法(在 exp 目录下):  python watch_progress.py     (Ctrl+C 退出)
import os, time, glob

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")

while True:
    os.system("cls")
    print("===== 训练进度监视  " + time.strftime("%H:%M:%S") + "  (Ctrl+C 退出) =====")
    files = sorted(glob.glob(os.path.join(LOGS, "EXP*_log.csv")))
    for f in files:
        lines = open(f, encoding="utf-8", errors="ignore").read().strip().split(chr(10))
        data = [ln for ln in lines if ln and not ln.startswith("#")]
        best = [ln for ln in lines if ln.startswith("# best")]
        age = int(time.time() - os.path.getmtime(f))
        tag = "  <<< 正在跑" if age < 120 else ""
        print("-" * 46)
        print(os.path.basename(f) + "   共 " + str(max(0, len(data) - 1)) + " 轮" + tag)
        if len(data) > 1:
            print("   最新: " + data[-1])
        if best:
            print("   " + best[-1])
    print("-" * 46)
    print("(每 5 秒自动刷新)")
    time.sleep(5)
