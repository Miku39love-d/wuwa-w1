"""WuwaMono：单残差学生版模型（DeepSeek 兼容架构）
- MLA 注意力：原样复用（与 DeepSeek 同构）
- MoE top-2 + shared expert：原样复用
- 残差流：标准单路 h = h + attn(...); h = h + ffn(...)  （mHC 折叠）
- 无 Engram / 无 MTP / 无 ViT
用途：从双路教师模型蒸馏，产出可被 llama.cpp(deepseek 架构)加载的权重
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import V4Config, MLAttention, MoE, RMSNorm, build_rope, build_causal_mask
from tokenizer import EOS_ID as T_EOS


class Block(nn.Module):
    def __init__(self, cfg: V4Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.attn = MLAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.ffn = MoE(cfg)

    def forward(self, h, ids, cos, sin, attn_mask=None):
        a, _ = self.attn(self.attn_norm(h), cos, sin, attn_mask=attn_mask)
        h = h + a
        m, _, _ = self.ffn(self.ffn_norm(h))
        h = h + m
        return h


class WuwaMono(nn.Module):
    def __init__(self, cfg: V4Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, ids, attn_mask=None):
        cfg = self.cfg
        B, S = ids.shape
        h = self.tok_emb(ids)
        cos, sin = build_rope(S + 64, cfg.qk_rope_head_dim, cfg.rope_theta, ids.device)
        if attn_mask is None:
            attn_mask = build_causal_mask(S, S, cfg, ids.device, h.dtype)
        for blk in self.blocks:
            h = blk(h, ids, cos, sin, attn_mask=attn_mask)
        h = self.norm(h)
        return self.lm_head(h)

    @torch.no_grad()
    def generate(self, ids, max_new=64, temperature=0.7, top_p=0.9, top_k=40,
                 stop_ids=(), repetition_penalty=1.1):
        """无 KV cache 简化生成（与教师模型接口对齐）。
        注意：每步 forward 必须喂完整序列（cur = out），只喂上一 token 会让模型丢失上下文。"""
        cfg = self.cfg
        out = ids.clone()
        for _ in range(max_new):
            logits = self.forward(out)[:, -1].float()
            logits = logits / max(temperature, 1e-5)
            for b in range(ids.size(0)):
                uniq = torch.unique(out[b])
                logits[b, uniq] /= repetition_penalty
            probs = F.softmax(logits, dim=-1)
            k = min(top_k, probs.size(-1))
            v, idx = probs.topk(k, dim=-1)
            v = v / v.sum(-1, keepdim=True)
            nxt = idx.gather(-1, torch.multinomial(v, 1))
            out = torch.cat([out, nxt], dim=1)
            if nxt[0, 0].item() in stop_ids or nxt[0, 0].item() == T_EOS:
                break
        return out


def init_from_teacher(student: WuwaMono, teacher_state: dict):
    """从教师 state_dict 拷贝权重（MLA/MoE/norm/emb/head 原样迁移）"""
    sd = {}
    for k, v in teacher_state.items():
        if k.startswith("blocks."):
            parts = k.split(".")
            i = parts[1]
            leaf = ".".join(parts[2:])
            if leaf.startswith("attn") or leaf.startswith("ffn"):
                sd[f"blocks.{i}.{leaf}"] = v
        elif k in ("tok_emb.weight", "norm.weight", "lm_head.weight"):
            sd[k] = v
        # hc.*, engram.*, mtp.*, vision.*, image_emb, hc_head 全部丢弃
    missing, unexpected = student.load_state_dict(sd, strict=False)
    # 确认缺失的只有预期之外的
    expect_missing = {"tok_emb.weight"}  # 教师里叫 tok_emb.weight，学生也是
    real_missing = [m for m in missing if m not in expect_missing]
    assert not real_missing, f"意外缺失: {real_missing}"
    assert not unexpected, f"意外多余: {unexpected}"
    return student


def count_params(m):
    return sum(p.numel() for p in m.parameters())
