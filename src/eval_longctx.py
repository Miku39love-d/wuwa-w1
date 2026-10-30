# -*- coding: utf-8 -*-
"""长上下文实测：5K 元数据窗口下模型的真实行为。
场景：模拟 PocketPal 长对话历史（~1500 token）+ 提问，看回答是否连贯。
注意：PyTorch 带 SWA=128（训练行为，乐观情形）；GGUF/llama.cpp 无 SWA（全注意力）。
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokenizer as T
from model import V4Config
from model_mono import WuwaMono

HERE = os.path.dirname(os.path.abspath(__file__))
T_EOS, T_BOS, T_USER, T_ASSIST = T.EOS_ID, T.BOS_ID, T.USER_ID, T.ASSIST_ID


@torch.no_grad()
def gen_long(model, prefix_ids, q, max_new=24, temp=0.5, top_k=30, rep=1.2):
    """prefix_ids: 长历史上文（多轮对话）；q: 当前问题"""
    ids = prefix_ids + [T_BOS, T_USER] + TK.encode(q) + [T_EOS, T_ASSIST]
    out = torch.tensor([ids])
    t0 = time.time()
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
    full = out[0].tolist()
    cut = full.index(T_ASSIST, len(prefix_ids)) + 1
    return TK.decode(full[cut:]), time.time() - t0


def build_history(n_tokens_target):
    """构造 ~n_tokens_target 的多轮历史（对齐训练时的单轮结构重复）"""
    rounds = [
        ("你好", "你好，我是 wuwa。有什么可以帮你？"),
        ("什么是张量", "张量就是多维数组。标量是零维，向量是一维，矩阵是二维。"),
        ("你能做什么", "我能对话、看图、推理，也能按描述画图。"),
        ("谢谢", "不客气。"),
        ("再见", "再见，随时找我。"),
        ("今天天气怎么样", "我是本地模型，查不了实时天气。"),
        ("什么是机器学习", "让机器从数据里自己学规律，不用手写规则。"),
        ("1+1等于几", "等于 2。"),
    ]
    ids, n = [], 0
    i = 0
    while n < n_tokens_target:
        q, a = rounds[i % len(rounds)]
        seg = [T_BOS, T_USER] + TK.encode(q) + [T_EOS, T_ASSIST] + TK.encode(a) + [T_EOS]
        ids.extend(seg)
        n += len(seg)
        i += 1
    return ids, i


if __name__ == "__main__":
    global TK
    ck = torch.load(os.path.join(HERE, "ckpt", "mono_best.pt"), map_location="cpu")
    model = WuwaMono(V4Config(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    TK = T.WuwaTokenizer(os.path.join(HERE, "ckpt", "tokenizer"))

    for target in [200, 800, 1500, 3000]:
        prefix, rounds = build_history(target)
        for q in ["你好", "你能做什么"]:
            txt, dt = gen_long(model, prefix, q)
            print(f"[上文{len(prefix)}tok/{rounds}轮] {q} -> '{txt[:60]}' ({dt:.1f}s)")
        print()
