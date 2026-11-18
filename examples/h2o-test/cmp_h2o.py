"""比两次运行:python3 cmp_h2o.py <甲前缀> <乙前缀> [词表大小]。logits 行逐行比;保留表按 (轮, 层, 头) 比重合率。"""
import json, os, sys
import numpy as np
A, Bp = sys.argv[1], sys.argv[2]
V = int(sys.argv[3]) if len(sys.argv) > 3 else 248320
a = np.fromfile(A + ".logits.bin", dtype=np.float32).reshape(-1, V)
b = np.fromfile(Bp + ".logits.bin", dtype=np.float32).reshape(-1, V)
print(f"logits 行数 {a.shape[0]} / {b.shape[0]};逐位相同 {a.tobytes() == b.tobytes()}")
n = min(len(a), len(b)); a, b = a[:n], b[:n]
d = np.abs(a - b)
top = a.argmax(1) == b.argmax(1)
def lsm(x):
    x = x.astype(np.float64); x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))
la, lb = lsm(a), lsm(b)
kl = (np.exp(la) * (la - lb)).sum(1)
nH = int(os.environ.get("NH", 14))
for name, sl in (("历史轮末位", slice(0, nH)), ("探题各位", slice(nH, n)), ("探题末 32 位", slice(max(nH, n - 32), n))):
    if d[sl].size == 0: continue
    print(f"  {name}(n {d[sl].shape[0]}):|Δlogit| 最大 {d[sl].max():.5f} 均值 {d[sl].mean():.6f};top1 一致 {top[sl].mean():.3f};KL 均值 {kl[sl].mean():.2e} 最大 {kl[sl].max():.2e}")
print("  逐行 |Δ| 最大:", " ".join(f"{x:.4f}" for x in d.max(1)))
ka = {(x["turn"], x["layer"], x["head"]): set(x["keep"]) for x in map(json.loads, open(A + ".keep.jsonl"))} if os.path.exists(A + ".keep.jsonl") else {}
kb = {(x["turn"], x["layer"], x["head"]): set(x["keep"]) for x in map(json.loads, open(Bp + ".keep.jsonl"))} if os.path.exists(Bp + ".keep.jsonl") else {}
if ka and kb:
    assert ka.keys() == kb.keys(), (len(ka), len(kb))
    ov = {k: len(ka[k] & kb[k]) / max(1, len(ka[k])) for k in ka}
    same_n = all(len(ka[k]) == len(kb[k]) for k in ka)
    turns = sorted({k[0] for k in ka})
    print(f"保留表 {len(ka)} 张(轮 × 层 × 头),两边大小都相同 {same_n},大小 {sorted({len(v) for v in ka.values()})}")
    print(f"  重合率:均值 {np.mean(list(ov.values())):.5f} 最小 {min(ov.values()):.5f};完全相同的张数 {sum(v == 1 for v in ov.values())}/{len(ov)}")
    for t in turns:
        v = [ov[k] for k in ov if k[0] == t]
        print(f"  轮 {t}:均值 {np.mean(v):.5f} 最小 {min(v):.5f}")
    last = max(turns)
    diff = sum(len(ka[k] ^ kb[k]) // 2 for k in ka if k[0] == last)
    print(f"  末轮不同的位置数合计 {diff}(共 {sum(len(ka[k]) for k in ka if k[0] == last)})")
