import sys, re
txt=sys.stdin.read()
def g(pat):
    m=re.search(pat,txt)
    return m.group(0) if m else "?"
print(" | ".join([g(r"Stable 帧数: \d+/\d+"), g(r"接收更新 \d+，被拒 \d+"),
                  g(r"中位 [\d.]+ m"), g(r"p90 [\d.]+ m"), g(r"p99 [\d.]+ m")]))
