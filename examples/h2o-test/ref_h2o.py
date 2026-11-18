"""H2O M1 参考实现(HF,0.8B,CPU,float32)。规则逐行照 /w/s478.py(probe / evict 的 H 支)与 /w/q35.py(_attn、prefill),
只把写死的 4 个键值头换成读配置(0.8B 是 2 个),并多记一张「原始位置」表好和 llama.cpp 的保留表对。
    python3 ref_h2o.py build              → turns.txt(每行 H/P + token id)
    python3 ref_h2o.py run <B> <out 前缀>  → <out>.logits.bin / <out>.keep.jsonl(B = 0:全留)
"""
import json, os, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface, DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

D = "/w/concept/llama-h2o"
PATH = "/w/models/Qwen3.5-0.8B"
SINK = 4
tk = AutoTokenizer.from_pretrained(PATH)
enc = lambda s: tk(s, add_special_tokens=False)["input_ids"]
def q_text(um): return f"<|im_start|>user\n{um}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
def h_text(u, a): return f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"

if sys.argv[1] == "build":
    NT, NU, NA = int(os.environ.get("NT", 14)), 90, 160
    txt = "".join(open(f"{D}/src/{p}", encoding="utf-8").read() for p in ("docs/build.md", "docs/docker.md", "docs/install.md"))
    ids = enc(txt)
    assert len(ids) >= NT * (NU + NA), len(ids)
    out = []
    for i in range(NT):
        a = i * (NU + NA)
        u, s = tk.decode(ids[a:a + NU]), tk.decode(ids[a + NU:a + NU + NA])
        out.append(("H", enc(h_text(u.strip(), s.strip()))))
    q = q_text("Based on our conversation so far, which CMake option enables CUDA, and how do I build with it?")
    cont = "To build with CUDA support, pass the option -DGGML_CUDA=ON to CMake when configuring the project, then build in Release mode. You also need the CUDA toolkit installed."
    out.append(("P", enc(q + cont)))
    with open(f"{D}/turns.txt", "w") as f:
        for k, x in out: f.write(k + " " + " ".join(map(str, x)) + "\n")
    print("轮数", len(out), "各轮长", [len(x) for _, x in out], "合计", sum(len(x) for _, x in out), "探题里问句长", len(enc(q)))
    raise SystemExit

B, OUT = int(sys.argv[2]), sys.argv[3]
torch.set_num_threads(6)
_PROBE = {"fn": None}

def _attn(module, q, k, v, attention_mask, **kw):                          # = q35._attn
    if _PROBE["fn"] is not None:
        kk = k.repeat_interleave(q.shape[1] // k.shape[1], 1)[0]
        sc = (q[0] @ kk.transpose(-1, -2)).float() * module.scaling
        ql, kl = q.shape[2], kk.shape[1]
        sc = sc.masked_fill(torch.arange(kl, device=q.device)[None, None, :] > torch.arange(kl - ql, kl, device=q.device)[None, :, None], float("-inf"))
        _PROBE["fn"](module.layer_idx, torch.softmax(sc, -1))
    return sdpa_attention_forward(module, q, k, v, attention_mask, **kw)

AttentionInterface.register("q35", _attn)
ALL_MASK_ATTENTION_FUNCTIONS.register("q35", sdpa_mask)
lm = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32).eval()
lm.config._attn_implementation = "q35"
cfg = getattr(lm.config, "text_config", lm.config)
FULL = [l for l, t in enumerate(cfg.layer_types) if t != "linear_attention"]
NKV = cfg.num_key_value_heads
print("全注意力层", FULL, "键值头", NKV, "查询头", cfg.num_attention_heads, flush=True)

ACC = {}
def probe(l, w):                                                           # = s478.probe(4 → NKV)
    g = w.view(NKV, w.shape[0] // NKV, w.shape[1], w.shape[2]).sum((1, 2))
    a = ACC.get(l)
    ACC[l] = g if a is None else torch.nn.functional.pad(a, (0, g.shape[-1] - a.shape[-1])) + g

cache = DynamicCache(config=lm.config); mask = []; pos = 0
POS = {l: torch.zeros((NKV, 0), dtype=torch.long) for l in FULL}

@torch.inference_mode()
def prefill(ids, all_logits=False):                                        # = q35.prefill(一轮 < CH,不分块;分块数学等价)
    global pos, mask
    am = torch.tensor([mask + [1] * len(ids)])
    o = lm(input_ids=torch.tensor([ids]), attention_mask=am, position_ids=torch.arange(pos, pos + len(ids))[None],
           past_key_values=cache, use_cache=True, **({} if all_logits else {"logits_to_keep": 1}))
    mask += [1] * len(ids); pos += len(ids)
    return o.logits[0].float()

@torch.inference_mode()
def evict():                                                               # = s478.evict 的 H 支(4 → NKV;TID → POS)
    global mask
    for l in FULL:
        lay = cache.layers[l]; n = lay.keys.shape[-2]
        if n <= B: continue
        idx = torch.arange(n)
        nr = (B - SINK) // 2; nh = B - SINK - nr
        mid = idx[SINK:n - nr]
        top = mid[ACC[l][:, SINK:n - nr].topk(nh, -1).indices]
        keep = torch.cat([idx[:SINK][None].expand(NKV, -1), top, idx[n - nr:][None].expand(NKV, -1)], 1).sort(-1).values
        g = keep[None, :, :, None].expand(1, NKV, -1, lay.keys.shape[-1])
        lay.keys, lay.values = lay.keys.gather(2, g).contiguous(), lay.values.gather(2, g).contiguous()
        if l in ACC: ACC[l] = ACC[l].gather(1, keep)
        POS[l] = POS[l].gather(1, keep)
    mask = [1] * cache.layers[FULL[0]].keys.shape[-2]

turns = [(l[0], list(map(int, l.split()[1:]))) for l in open(f"{D}/turns.txt") if l.strip()]
fl, fk = open(OUT + ".logits.bin", "wb"), open(OUT + ".keep.jsonl", "w")
for it, (kind, ids) in enumerate(turns):
    t0 = time.time()
    if kind == "H":
        if B: _PROBE["fn"] = probe
        lg = prefill(ids)
        _PROBE["fn"] = None
        for l in FULL: POS[l] = torch.cat([POS[l], torch.arange(pos - len(ids), pos)[None].expand(NKV, -1)], 1)
        if B:
            evict()
            for l in FULL:
                for h in range(NKV): fk.write(json.dumps({"turn": it, "layer": l, "head": h, "keep": POS[l][h].tolist()}) + "\n")
    else:                                                                  # 探题:不累计、不淘汰(参考实现里在快照上问;这里放在最后,不用快照)
        lg = prefill(ids, all_logits=True)
    lg.numpy().astype(np.float32).tofile(fl)
    print(f"turn {it} {kind} n={len(ids)} 缓存长 {cache.layers[FULL[0]].keys.shape[-2]} {time.time() - t0:.1f}s", flush=True)
fl.close(); fk.close()
