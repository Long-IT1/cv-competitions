# 多种子排队器: 依次运行"实验 x 种子"(一个跑完再跑下一个, 不并行)
# 断点续跑: 已经跑完的(日志里有 best= 行)会自动跳过 -> 中途挂了/关了窗口, 重新运行本脚本即可续上
# 用法(在 exp 目录下): python run_seeds.py
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# ============ 队列: 想跑什么, 改这里就行 ============
# ("实验名", [种子1, 种子2, ...])
PLAN = [
    ("EXP009", [1, 2, 3, 4]),          # 两张牌叠满(640x320); seed42 就是第 1 次跑, 再补 4 个凑够 5 个
]
# ===================================================


def log_path(exp, seed):
    return os.path.join(HERE, "logs", "%s_s%d_log.csv" % (exp, seed))


def is_done(exp, seed):
    """跑完的标志: 日志文件末尾有 '# best=' 那一行"""
    p = log_path(exp, seed)
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8", errors="ignore") as f:
        return "# best=" in f.read()


env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"

total = sum(len(seeds) for _, seeds in PLAN)
n = 0
for exp, seeds in PLAN:
    for seed in seeds:
        n += 1
        tag = "[%d/%d]" % (n, total)
        if is_done(exp, seed):
            print(tag + " %s seed=%d 已完成, 跳过" % (exp, seed), flush=True)
            continue
        print("=" * 60, flush=True)
        print(tag + " 开始运行: %s  seed=%d" % (exp, seed), flush=True)
        print("=" * 60, flush=True)
        code = subprocess.run([sys.executable, "-u", exp + ".py", str(seed)], env=env).returncode
        if code != 0:
            print(tag + " !! %s seed=%d 异常退出(退出码 %d)" % (exp, seed, code), flush=True)
            print("已停止队列。检查上方报错, 修好后重新运行本脚本会自动从这里续上。", flush=True)
            sys.exit(code)
        if not is_done(exp, seed):
            print(tag + " !! 跑完了但没找到日志 " + log_path(exp, seed), flush=True)
            print("可能是这个实验文件还不支持种子参数(会把结果写进不带后缀的文件)。", flush=True)
            print("已停止队列, 请先确认该文件改好了。", flush=True)
            sys.exit(1)

print("=" * 60, flush=True)
print("队列全部跑完, 共 %d 个任务。下一步: python seed_stats.py" % total, flush=True)
