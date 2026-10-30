# -*- coding: utf-8 -*-
"""w1 系列训练：Flash / Pro / Pro+ 三模型统一入口
配方（基于六轮 SFT 失败教训）：
- 全参数 + KL 锚定（λ=3.0，v5 的 λ=1.0 压不住 lr 5e-5，这次 lr 降 2.5 倍、λ 提 3 倍）
- 人格数据为主（架构图四则），混入 gap/numeric/chat 防遗忘
- 输入格式铁律：[BOS][USER] q [EOS][ASSIST] a [EOS]
用法: python3 train_w1.py --model flash|pro|proplus
"""
import os, sys, json, math, random, time, argparse
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tokenizer as T
from model import V4Config
from model_mono import WuwaMono

T_BOS, T_USER, T_ASSIST, T_EOS, T_PAD = T.BOS_ID, T.USER_ID, T.ASSIST_ID, T.EOS_ID, T.PAD_ID
TK = T.WuwaTokenizer(os.path.join(HERE, "ckpt", "tokenizer"))

MODELS = {
    "flash":   dict(init="ckpt/mono_best.pt",        out="ckpt/w1_flash.pt",   steps=250, lr=5e-5, warmup=15),
    "pro":     dict(init="ckpt/w1_pro_init.pt",      out="ckpt/w1_pro.pt",     steps=250, lr=5e-5, warmup=15),
    "proplus": dict(init="ckpt/w1_proplus_init.pt",  out="ckpt/w1_proplus.pt", steps=250, lr=5e-5, warmup=15),
}
MAX_LEN = 176
DEMO_EVERY = 50
SEED = 20261030

DEMO_QS = {
    "你好": ["你好", "wuwa", "您好", "嗯", "说吧"],
    "请问，你是谁": ["wuwa", "我是", "模型"],
    "1+1等于几": ["2", "等于二", "=2"],
    "你能做什么": ["对话", "聊天", "问题", "回答", "推理", "画图"],
    "今天天气怎么样": ["天气", "不联网", "本地", "无法", "查不了", "不知道", "窗外", "看天", "预报"],
    "什么是人工智能？": ["人工智能", "AI", "机器", "智能"],
    "谢谢": ["不客气", "不用", "嗯", "好的"],
    "再见": ["再见", "拜拜", "回见", "拜"],
    "100 块钱和 50 块钱，你选哪个？": ["100"],
    "100 和 50 哪个大？": ["100"],
    "8 加 5 等于几？": ["13"],
    # 人格四则检测（架构图）
    "讲个笑话": ["不讲", "不会", "你讲", "拒绝", "不好笑"],
    "夸夸我": ["不夸", "拒绝", "挺好", "别求"],
    "人工智能会取代人类吗": ["不会", "取代", "不动脑", "你动"],
}

def load_rows():
    rows = []
    # 人格数据：主菜，重复 20 遍
    persona = [json.loads(l) for l in open(os.path.join(HERE, "data/persona_chat.jsonl"), encoding="utf-8")]
    rows.extend(persona * 30)
    # 防遗忘：gap + numeric
    for f in ["data/gap_chat.jsonl", "data/numeric_chat.jsonl"]:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            rows.extend([json.loads(l) for l in open(p, encoding="utf-8")] * 2)
    # 防遗忘：原 chat 抽样（仅 Pro/Pro+ 需要 recovery；Flash 已会）
    return rows

def load_rows_recovery():
    rows = []
    persona = [json.loads(l) for l in open(os.path.join(HERE, "data/persona_chat.jsonl"), encoding="utf-8")]
    rows.extend(persona * 10)
    for f in ["data/gap_chat.jsonl", "data/numeric_chat.jsonl"]:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            rows.extend([json.loads(l) for l in open(p, encoding="utf-8")])
    # 原 chat 数据抽 800 条做 recovery（恢复插值损失的对话能力）
    chat = [json.loads(l) for l in open(os.path.join(HERE, "data/sft.jsonl"), encoding="utf-8") if json.loads(l).get("type") == "chat"]
    chat = [r for r in chat if len(r["user"]) < 60 and len(r["assistant"]) < 80]
    random.Random(SEED).shuffle(chat)
    rows.extend(chat[:800])
    return rows

def make_batch(rows):
    maxlen = min(MAX_LEN, max(len(r["user"]) + len(r["assistant"]) + 12 for r in rows))
    maxlen = max(maxlen, 48)
    ids, labels = [], []
    for r in rows:
        q = TK.encode(r["user"]); a = TK.encode(r["assistant"]) + [T_EOS]
        x = [T_BOS, T_USER] + q + [T_EOS, T_ASSIST] + a
        y = [-100] * (4 + len(q)) + a
        x, y = x[:maxlen], y[:maxlen]
        pad = maxlen - len(x)
        assert pad >= 0, f"seq too long: {len(x)} > {maxlen}"
        ids.append(x + [T_PAD] * pad); labels.append(y + [-100] * pad)
    return torch.tensor(ids), torch.tensor(labels)

@torch.no_grad()
def gen(model, q, max_new=32, temp=0.5, top_p=0.9, top_k=30, rep=1.2):
    ids = [T_BOS, T_USER] + TK.encode(q) + [T_EOS, T_ASSIST]
    out = torch.tensor([ids])
    for _ in range(max_new):
        logits = model(out)[:, -1].float() / max(temp, 1e-5)
        uniq = torch.unique(out[0])
        logits[0, uniq] /= rep
        probs = F.softmax(logits, dim=-1)
        v, idx = probs.topk(min(top_k, probs.size(-1)), dim=-1)
        v = v / v.sum(-1, keepdim=True)
        nxt = idx.gather(-1, torch.multinomial(v, 1))
        out = torch.cat([out, nxt], dim=1)
        if nxt[0, 0].item() == T_EOS:
            break
    full = out[0].tolist()
    cut = full.index(T_ASSIST) + 1 if T_ASSIST in full else 4
    return TK.decode(full[cut:])

def judge(txt, keys):
    t = txt.strip()
    if not t: return False, "EMPTY"
    if len(t) > 60: return False, "LONG"
    for i in range(len(t) - 1):
        if t.count(t[i:i+2]) >= 4: return False, "LOOP"
    if not any(k in t for k in keys): return False, "OFF"
    return True, "OK"

def demo(model, log):
    ok = 0
    for q, keys in DEMO_QS.items():
        txt = gen(model, q)
        good, tag = judge(txt, keys)
        ok += good
        log(f"  [{q}] -> '{txt[:40]}' {tag}")
    return ok

def train_stage(model, anchor, rows, steps, lr, warmup, tag, log, demo_fn=None, lambda_kl=0.5):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 1.0) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)) if s < steps else 0.0)
    best, t0 = -1, time.time()
    for step in range(1, steps + 1):
        batch = random.Random(SEED + step).sample(rows, min(16, len(rows)))
        x, y = make_batch(batch)
        logits = model(x)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        loss = ce
        if anchor is not None:
            with torch.no_grad():
                a_logits = anchor(x)
            kl = F.kl_div(F.log_softmax(logits, -1), F.softmax(a_logits, -1), reduction="batchmean")
            loss = ce + lambda_kl * kl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 25 == 0 or step == 1:
            log(f"[{tag}] step {step}/{steps} | ce {ce.item():.4f} | loss {loss.item():.4f} | lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s")
        if demo_fn and (step % DEMO_EVERY == 0 or step == steps):
            ok = demo_fn()
            log(f"[{tag}] step {step} demo: 对题合格 {ok}/{len(DEMO_QS)}")
            if ok > best:
                best = ok
                torch.save({"cfg": model.cfg.__dict__, "model": model.state_dict()}, MODELS[CUR]["out"])
                log(f"[{tag}] ** best {ok}/{len(DEMO_QS)} 已保存")
    return best

CUR = None

def main():
    global CUR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["flash", "pro", "proplus"])
    ap.add_argument("--recovery", type=int, default=0, help="recovery 阶段步数（Pro/Pro+ 用）")
    args = ap.parse_args()
    CUR = args.model
    cfg_m = MODELS[CUR]
    log_path = os.path.join(HERE, "logs", f"train_w1_{CUR}.log")
    def log(msg):
        line = time.strftime("[%H:%M:%S] ") + msg
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    random.seed(SEED); torch.manual_seed(SEED)
    ck = torch.load(os.path.join(HERE, cfg_m["init"]), map_location="cpu")
    model = WuwaMono(V4Config(**ck["cfg"]))
    model.load_state_dict(ck["model"]); model.train()
    n = sum(p.numel() for p in model.parameters())
    log(f"===== w1 {CUR.upper()} 训练开始 | {n/1e6:.2f}M 参数 | init={cfg_m['init']} =====")
    anchor = WuwaMono(V4Config(**ck["cfg"])); anchor.load_state_dict(ck["model"]); anchor.eval()
    for p in anchor.parameters(): p.requires_grad_(False)

    def demo_fn():
        model.eval()
        with torch.no_grad():
            ok = demo(model, log)
        model.train()
        return ok

    # 阶段 1：recovery（仅 Pro/Pro+，从插值初始化恢复对话能力）
    if args.recovery > 0:
        rows = load_rows_recovery()
        log(f"recovery 数据: {len(rows)} 条")
        best1 = train_stage(model, anchor, rows, args.recovery, 1e-5, 10, "recovery", log, demo_fn, lambda_kl=3.0)
        log(f"recovery 完成: best {best1}/{len(DEMO_QS)}")
    # 阶段 2：人格 SFT
    rows = load_rows()
    log(f"人格数据: {len(rows)} 条")
    best2 = train_stage(model, anchor, rows, cfg_m["steps"], cfg_m["lr"], cfg_m["warmup"], "persona", log, demo_fn)
    log(f"DONE. w1 {CUR.upper()} persona best {best2}/{len(DEMO_QS)} | 总耗时 {time.time()-0:.0f}s")

if __name__ == "__main__":
    main()
