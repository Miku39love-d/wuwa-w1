# -*- coding: utf-8 -*-
"""权重迁移（upcycling）：mono(384/8层) → Pro(544/12层) / Pro+(448/11层)
策略：
- 宽度扩展：所有权重按 hidden/moe_inter 维度 bicubic 插值到目标大小（保留原始信息，新维度平滑过渡）
- 深度扩展：均匀复制中间层（复制的层加小噪声打破对称）
- 不动：MLA 的 q_b_proj/kv_b_proj/kv_a_proj 输出维（qk/v/kv_lora 不随 hidden 变）、norm 类 1D 按 hidden 插值
"""
import torch, torch.nn.functional as F, copy, os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def interp2d(t, out_size, in_size):
    """把 2D 权重 [out, in] 插值到 [out_size, in_size]"""
    if tuple(t.shape) == (out_size, in_size):
        return t.clone()
    x = t[None, None].float()  # [1,1,out,in]
    y = F.interpolate(x, size=(out_size, in_size), mode="bicubic", align_corners=False)
    return y[0, 0].to(t.dtype)

def interp1d(t, size):
    if t.shape[0] == size:
        return t.clone()
    y = F.interpolate(t[None, None].float(), size=(size,), mode="linear", align_corners=False)
    return y[0, 0].to(t.dtype)

def upcycle(src_ckpt, out_path, new_hidden, new_layers, new_moe_inter, name):
    ck = torch.load(src_ckpt, map_location="cpu")
    sd = ck["model"]
    cfg = dict(ck["cfg"])
    old_hidden = cfg["hidden_size"]; old_layers = cfg["num_hidden_layers"]
    old_moe = cfg["moe_inter"]
    print(f"[{name}] {old_hidden}h/{old_layers}L/moe{old_moe} -> {new_hidden}h/{new_layers}L/moe{new_moe_inter}")

    # ---- 1. 深度扩展的层索引映射：均匀复制中间层 ----
    if new_layers == old_layers:
        layer_map = list(range(old_layers))
    else:
        extra = new_layers - old_layers
        # 在中间区域均匀选 extra 个层位置插入副本
        lo, hi = 1, old_layers - 2  # 不复制首尾
        span = hi - lo + 1
        inserts = []
        for i in range(extra):
            pos = lo + (i * span) // extra
            inserts.append(pos)
        layer_map = []
        for i in range(old_layers):
            layer_map.append(i)
            n_dup = inserts.count(i)
            layer_map.extend([i] * n_dup)
        assert len(layer_map) == new_layers, f"层映射错误 {len(layer_map)} != {new_layers}"
    print(f"  层映射: {layer_map}")

    new_sd = {}
    # ---- 2. 非层参数 ----
    for key in ["tok_emb.weight", "lm_head.weight"]:
        new_sd[key] = interp2d(sd[key], sd[key].shape[0], new_hidden)
    if "norm.weight" in sd:
        new_sd["norm.weight"] = interp1d(sd["norm.weight"], new_hidden)

    # ---- 3. 逐层迁移 ----
    H, M = new_hidden, new_moe_inter
    for new_idx, old_idx in enumerate(layer_map):
        p = f"blocks.{old_idx}."
        n = f"blocks.{new_idx}."
        def cp(src_key, dst_key, fn):
            new_sd[dst_key] = fn(sd[src_key])
        # 注意力：扩 hidden 维的矩阵
        cp(p+"attn.kv_a_proj.weight", n+"attn.kv_a_proj.weight", lambda t: interp2d(t, t.shape[0], H))   # [112, hidden]
        cp(p+"attn.q_a_proj.weight", n+"attn.q_a_proj.weight", lambda t: interp2d(t, t.shape[0], H))   # [192, hidden]
        cp(p+"attn.o_up.weight",      n+"attn.o_up.weight",      lambda t: interp2d(t, H, t.shape[1]))   # [hidden, 192]
        # 不动的：kv_b_proj[512,96], q_b_proj[256,192], kv_a_norm, q_a_norm
        # o_down: nn.Linear(nh*vd, o_lora_rank) -> [192, nh*vd]，与 hidden 无关，不动！
        for k in ["attn.kv_b_proj.weight", "attn.q_b_proj.weight", "attn.kv_a_norm.weight", "attn.q_a_norm.weight", "attn.o_down.weight"]:
            new_sd[n+k] = sd[p+k].clone()
        # 1D norm
        new_sd[n+"attn_norm.weight"] = interp1d(sd[p+"attn_norm.weight"], H)
        new_sd[n+"ffn_norm.weight"] = interp1d(sd[p+"ffn_norm.weight"], H)
        # MoE：w1/w3 = [moe_inter, hidden]，w2 = [hidden, moe_inter]
        for grp in ["experts", "shared"]:
            if grp == "experts":
                n_exp = cfg["n_experts"]
                for e in range(n_exp):
                    for w in ["w1", "w3"]:
                        new_sd[f"{n}ffn.experts.{e}.{w}.weight"] = interp2d(sd[f"{p}ffn.experts.{e}.{w}.weight"], M, H)
                    new_sd[f"{n}ffn.experts.{e}.w2.weight"] = interp2d(sd[f"{p}ffn.experts.{e}.w2.weight"], H, M)
            else:
                for w in ["w1", "w3"]:
                    new_sd[f"{n}ffn.shared.{w}.weight"] = interp2d(sd[f"{p}ffn.shared.{w}.weight"], M, H)
                new_sd[f"{n}ffn.shared.w2.weight"] = interp2d(sd[f"{p}ffn.shared.w2.weight"], H, M)
        # gate [n_experts, hidden]
        new_sd[n+"ffn.gate.weight"] = interp2d(sd[p+"ffn.gate.weight"], sd[p+"ffn.gate.weight"].shape[0], H)

    # ---- 4. 复制层加噪声打破对称 ----
    seen_src = {}
    for new_idx, old_idx in enumerate(layer_map):
        seen_src.setdefault(old_idx, []).append(new_idx)
    for old_idx, idxs in seen_src.items():
        if len(idxs) > 1:
            for dup in idxs[1:]:
                for k in list(new_sd.keys()):
                    if k.startswith(f"blocks.{dup}."):
                        t = new_sd[k]
                        new_sd[k] = t + torch.randn_like(t) * (t.std() * 0.01 + 1e-8)
            print(f"  层 {old_idx} 复制为 {idxs}（已加噪声）")

    # ---- 5. 新 cfg 与保存 ----
    cfg["hidden_size"] = new_hidden
    cfg["num_hidden_layers"] = new_layers
    cfg["moe_inter"] = new_moe_inter
    cfg["shared_inter"] = new_moe_inter
    # 上下文目标：Pro 16K / Pro+ 15K
    total = sum(v.numel() for v in new_sd.values())
    torch.save({"cfg": cfg, "model": new_sd}, out_path)
    print(f"  -> {out_path}  参数量 {total/1e6:.2f}M")
    return total

if __name__ == "__main__":
    src = os.path.join(HERE, "ckpt", "mono_best.pt")
    upcycle(src, os.path.join(HERE, "ckpt", "w1_pro_init.pt"), 544, 12, 512, "w1 Pro")
    print()
    upcycle(src, os.path.join(HERE, "ckpt", "w1_proplus_init.pt"), 448, 11, 448, "w1 Pro+")
