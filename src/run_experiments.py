# 顺序运行多个实验脚本 (一个跑完再跑下一个, 不并行, 省内存)
# 断点续跑: 已经跑完的(日志里有 best= 行)会自动跳过 -> 中途挂了/关了窗口, 重新运行本脚本即可续上
# 用法: python run_experiments.py EXP006 EXP005 EXP004 EXP003 EXP002b EXP002 EXP001
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

names = sys.argv[1:]
if not names:
    print("用法: python run_experiments.py EXP006 EXP005 ...")
    sys.exit(1)


def is_done(name):
    """跑完的标志: logs/EXPxxx_log.csv 末尾有 '# best=' 那一行"""
    p = os.path.join(HERE, "logs", name + "_log.csv")
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8", errors="ignore") as f:
        return "# best=" in f.read()


env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"

for i, name in enumerate(names, 1):
    tag = "[%d/%d]" % (i, len(names))
    if is_done(name):
        print(tag + " %s 已完成, 跳过" % name, flush=True)
        continue
    print("=" * 60, flush=True)
    print(tag + " 开始运行: %s" % name, flush=True)
    print("=" * 60, flush=True)
    code = subprocess.run([sys.executable, "-u", name + ".py"], env=env).returncode
    print("-" * 60, flush=True)
    print("%s 结束, 退出码=%d" % (name, code), flush=True)
    if code != 0:
        print("有实验异常退出, 停止后续运行, 请检查上方报错", flush=True)
        print("修好后重新运行本脚本, 已经跑完的实验会自动跳过", flush=True)
        sys.exit(code)

print("全部实验运行完毕", flush=True)
