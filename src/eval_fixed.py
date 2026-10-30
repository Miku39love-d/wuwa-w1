# -*- coding: utf-8 -*-
"""用训练时的正确输入格式 [BOS][USER] q [EOS][ASSIST] 重新评估各 ckpt"""
import torch, torch.nn.functional as F, sys, time, os
import tokenizer as T
from model import V4Config
from model_mono import WuwaMono

T_EOS = T.EOS_ID
TK = T.WuwaTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ckpt', 'tokenizer'))
DEMO_QS = {
    "你好": ["你好", "wuwa", "您好"],
    "请问，你是谁": ["wuwa", "我是", "模型"],
    "1+1等于几": ["2", "等于二", "=2"],
    "你能做什么": ["对话", "聊天", "问题", "回答"],
    "今天天气怎么样": ["天气", "不联网", "本地", "无法", "查不到", "不知道"],
    "什么是人工智能？": ["人工智能", "AI", "机器", "智能"],
    "谢谢": ["不客气", "不用", "嗯", "好的"],
    "再见": ["再见", "拜拜", "回见"],
    "100 块钱和 50 块钱，你选哪个？": ["100"],
    "100 和 50 哪个大？": ["100"],
    "8 加 5 等于几？": ["13"],
}

@torch.no_grad()
def gen(model, q, max_new=32, temp=0.5, top_p=0.9, top_k=30, rep=1.2):
    ids = [T.BOS_ID, T.USER_ID] + TK.encode(q) + [T.EOS_ID, T.ASSIST_ID]
    out = torch.tensor([ids])
    for _ in range(max_new):
        logits = model(out)[:, -1].float()
        logits = logits / max(temp, 1e-5)
        uniq = torch.unique(out[0])
        logits[0, uniq] /= rep
        probs = F.softmax(logits, dim=-1)
        k = min(top_k, probs.size(-1))
        v, idx = probs.topk(k, dim=-1)
        v = v / v.sum(-1, keepdim=True)
        nxt = idx.gather(-1, torch.multinomial(v, 1))
        out = torch.cat([out, nxt], dim=1)
        if nxt[0, 0].item() == T_EOS:
            break
    # 只解码 ASSIST 之后的部分
    full = out[0].tolist()
    cut = full.index(T.ASSIST_ID) + 1 if T.ASSIST_ID in full else 4
    return TK.decode(full[cut:])

def judge(q, txt, keys):
    t = txt.strip()
    if not t: return False, "EMPTY"
    if len(t) > 60: return False, "LONG"
    # 循环检测：任何 2-gram 重复 >=4 次
    for i in range(len(t) - 1):
        if t.count(t[i:i+2]) >= 4: return False, "LOOP"
    if not any(k in t for k in keys): return False, "OFF"
    return True, "OK"

def eval_ckpt(path, n=3):
    ck = torch.load(path, map_location="cpu")
    m = WuwaMono(V4Config(**ck["cfg"]))
    m.load_state_dict(ck["model"]); m.eval()
    ok_total = 0
    print(f"\n===== {path} =====")
    for q, keys in DEMO_QS.items():
        votes = []
        for _ in range(n):
            txt = gen(m, q)
            good, tag = judge(q, txt, keys)
            votes.append(good)
            if _ == 0:
                print(f"  [{q}] -> '{txt[:50]}' {tag}")
        ok_total += sum(votes)
    print(f"  对题合格 {ok_total}/{len(DEMO_QS)*n}")
    return ok_total

if __name__ == "__main__":
    paths = sys.argv[1:] or ["ckpt/mono_best.pt", "ckpt/sft_v3.pt", "ckpt/sft_v4.pt"]
    for p in paths:
        try: eval_ckpt(p)
        except Exception as e: print(f"{p} FAILED: {e}")
