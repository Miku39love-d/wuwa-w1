"""
wuwa-v4 语言骨干：DeepSeek-V4.1-Flash 架构的紧凑复刻。

参考 deepinfer/model_executor/models/v4_modules/*：
  - MLA 注意力（q_lora / kv_lora / 分离 rope，o_lora 分组输出）
  - MoE（top-k 路由 + 共享专家，SwiGLU）
  - mHC 多路超连接残差（hc_mult 路残差流，pre/post 混合）
  - Engram n-gram 条件记忆（哈希查表 -> 投影 -> 门控注入）
  - 块稀疏注意力（按块打分取 top-k，SWA 兜底）
  - MTP / DFlash 投机头（含 confidence 头，用于 draft-verify）
  - ViT 视觉塔 + MLP 投影（等价 vl_aligner）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- config
@dataclass
class V4Config:
    vocab_size: int = 512
    hidden_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    q_lora_rank: int = 128
    kv_lora_rank: int = 64
    o_lora_rank: int = 128
    o_groups: int = 2
    # MoE
    n_experts: int = 8
    top_k: int = 2
    moe_inter: int = 256
    shared_inter: int = 256
    # mHC
    hc_mult: int = 2
    # Engram
    engram_layers: tuple = (2, 5)
    engram_table: int = 8192
    engram_dim: int = 32
    engram_ngram: int = 2
    # attention sparsity
    sparse_attn: bool = True
    block_size: int = 16
    topk_blocks: int = 6
    swa_window: int = 128
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    mtp: bool = True
    image_token_id: int = 258       # tokenizer 初始化时覆盖
    image_tokens: int = 64          # 每张图的 patch 数（8x8）
    vision_dim: int = 192
    vision_depth: int = 4
    vision_heads: int = 4
    patch_size: int = 8
    image_size: int = 64

    @property
    def qk_head_dim(self):
        return self.qk_nope_head_dim + self.qk_rope_head_dim


# ---------------------------------------------------------------- norms / rope
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope(seq_len, dim, theta, device, dtype=torch.float32):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)          # [S, dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin, pos_offset=0):
    # x: [B, H, S, Dh]
    S = x.size(2)
    cos = cos[pos_offset:pos_offset + S].unsqueeze(0).unsqueeze(0)
    sin = sin[pos_offset:pos_offset + S].unsqueeze(0).unsqueeze(0)
    c1, c2 = cos.chunk(2, dim=-1)
    s1, s2 = sin.chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c1 - x2 * s1, x2 * c2 + x1 * s2], dim=-1)


# ---------------------------------------------------------------- MLA
class MLAttention(nn.Module):
    """DeepSeek MLA：低秩 Q/KV 压缩 + 解耦 RoPE，输出按组低秩投影。"""

    def __init__(self, cfg: V4Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        self.nh = cfg.num_attention_heads
        self.qk_nope = cfg.qk_nope_head_dim
        self.qk_rope = cfg.qk_rope_head_dim
        self.vd = cfg.v_head_dim
        self.qk = self.qk_nope + self.qk_rope

        self.q_a_proj = nn.Linear(d, cfg.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_eps)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, self.nh * self.qk, bias=False)

        self.kv_a_proj = nn.Linear(d, cfg.kv_lora_rank + self.qk_rope, bias=False)
        self.kv_a_norm = RMSNorm(cfg.kv_lora_rank, cfg.rms_eps)
        self.kv_b_proj = nn.Linear(cfg.kv_lora_rank, self.nh * (self.qk_nope + self.vd), bias=False)

        # 分组低秩输出（o_groups 组各自降维再拼）
        assert cfg.o_lora_rank % cfg.o_groups == 0
        self.o_groups = cfg.o_groups
        self.o_down = nn.Linear(self.nh * self.vd, cfg.o_lora_rank, bias=False)
        self.o_up = nn.Linear(cfg.o_lora_rank, d, bias=False)
        self.softmax_scale = self.qk ** -0.5

    def forward(self, x, cos, sin, past=None, attn_mask=None, use_cache=False):
        B, S, _ = x.shape
        cfg = self.cfg
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x))).view(B, S, self.nh, self.qk).transpose(1, 2)
        kv = self.kv_a_proj(x)
        kv_c, k_pe = kv.split([cfg.kv_lora_rank, self.qk_rope], dim=-1)
        kv_c = self.kv_a_norm(kv_c)
        kv = self.kv_b_proj(kv_c).view(B, S, self.nh, self.qk_nope + self.vd).transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope, self.vd], dim=-1)
        k_pe = k_pe.view(B, S, 1, self.qk_rope).transpose(1, 2).expand(-1, self.nh, -1, -1)
        k = torch.cat([k_nope, k_pe], dim=-1)
        q_rope, q_nope = q[..., self.qk_nope:], q[..., :self.qk_nope]

        offset = past[0].size(2) if past is not None else 0
        q_rope = apply_rope(q_rope, cos, sin, offset)
        k_rope = apply_rope(k[..., self.qk_nope:], cos, sin, offset)
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k[..., :self.qk_nope], k_rope], dim=-1)

        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        new_past = (k, v) if use_cache else None

        attn_mask = attn_mask if attn_mask is not None else build_causal_mask(
            S, k.size(2), cfg, x.device, q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             scale=self.softmax_scale)
        out = out.transpose(1, 2).reshape(B, S, self.nh * self.vd)
        out = self.o_up(self.o_down(out))
        return out, new_past


def build_causal_mask(q_len, kv_len, cfg: V4Config, device, dtype, block_sparse=None):
    """因果掩码 + SWA 窗口 + 可选块稀疏 top-k。"""
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    mask = (k_pos <= q_pos)
    if cfg.swa_window and kv_len > cfg.swa_window:
        mask &= (k_pos >= q_pos - cfg.swa_window)
    if block_sparse and cfg.sparse_attn:
        allow = block_sparse  # [q_blocks, k_blocks] bool
        bs = cfg.block_size
        qb = q_pos // bs
        kb = k_pos // bs
        nb_q, nb_k = allow.shape
        # 仅当块索引覆盖当前位置时应用
        if nb_q == math.ceil(q_len / bs) and nb_k == math.ceil(kv_len / bs):
            qb = qb - (math.ceil(kv_len / bs) - nb_q)
            ok = torch.zeros_like(mask)
            for i in range(nb_q):
                ok[qb.squeeze(1) == i] = allow[i][kb[0][None, :]]
            mask &= ok
    m = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
    m.masked_fill_(~mask, float("-inf"))
    return m.view(1, 1, q_len, kv_len)


def block_index_scores(q, k, cfg: V4Config):
    """按块均值打分的 top-k 块选择（训练时生成 sparse mask 用）。"""
    bs = cfg.block_size
    B, H, S, _ = q.shape
    nb = math.ceil(S / bs)
    pad = nb * bs - S
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
    qb = q.view(B, H, nb, bs, -1).mean(3).mean(1)   # [B, nb, d]
    kb = k.view(B, H, nb, bs, -1).mean(3).mean(1)
    score = torch.matmul(qb, kb.transpose(1, 2)) / math.sqrt(qb.size(-1))
    causal = torch.tril(torch.ones(nb, nb, device=q.device, dtype=torch.bool))
    score = score.masked_fill(~causal, float("-inf"))
    allow = torch.zeros_like(causal)
    kk = min(cfg.topk_blocks, nb)
    idx = score.topk(kk, dim=-1).indices          # [B, nb, kk]
    allow.scatter_(1, idx, True)
    allow |= torch.eye(nb, device=q.device, dtype=torch.bool)
    return allow[0]                                # 取 batch0 代表（训练同构块模式）


# ---------------------------------------------------------------- MoE
class Expert(nn.Module):
    def __init__(self, d, inter):
        super().__init__()
        self.w1 = nn.Linear(d, inter, bias=False)
        self.w3 = nn.Linear(d, inter, bias=False)
        self.w2 = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """top-k 路由 + 共享专家。"""

    def __init__(self, cfg: V4Config):
        super().__init__()
        d = cfg.hidden_size
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.gate = nn.Linear(d, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d, cfg.moe_inter) for _ in range(cfg.n_experts)])
        self.shared = Expert(d, cfg.shared_inter)

    def forward(self, x):
        B, S, D = x.shape
        logits = self.gate(x)
        probs = F.softmax(logits.float(), dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)
        flat = x.reshape(B * S, D)
        out = torch.zeros_like(flat)
        fi = topi.reshape(B * S, self.top_k)
        fw = topv.reshape(B * S, self.top_k).to(x.dtype)
        for e in range(self.n_experts):
            mask = (fi == e)
            rows = mask.any(-1)
            if not rows.any():
                continue
            w = (mask.to(x.dtype) * fw).sum(-1, keepdim=True)[rows]
            out[rows] += self.experts[e](flat[rows]) * w
        out = out.reshape(B, S, D) + self.shared(x)
        return out, topi, topv


# ---------------------------------------------------------------- mHC
class HyperConnection(nn.Module):
    """mHC：hc_mult 路残差流，子层前后做学习到的混合。"""

    def __init__(self, d, mult=2):
        super().__init__()
        self.mult = mult
        self.pre = nn.Linear(d, mult, bias=False)
        self.post = nn.Linear(d, mult, bias=False)
        nn.init.zeros_(self.post.weight)
        self.post.weight.data[:, 0] = 1.0

    def pre_mix(self, streams):
        # streams: [B,S,mult,d] -> 子层输入 [B,S,d]
        w = F.softmax(self.pre(streams.mean(2)), dim=-1)      # [B,S,mult]
        return (streams * w.unsqueeze(-1)).sum(2)

    def post_mix(self, streams, out):
        w = self.post(out)                                     # [B,S,mult]
        return streams + w.unsqueeze(-1) * out.unsqueeze(2)

    def fold(self, streams):
        return streams.mean(2)


# ---------------------------------------------------------------- Engram
class Engram(nn.Module):
    """n-gram 哈希条件记忆：查表 -> 投影 -> 门控注入。"""

    def __init__(self, cfg: V4Config):
        super().__init__()
        self.table = nn.Embedding(cfg.engram_table, cfg.engram_dim)
        nn.init.normal_(self.table.weight, std=0.02)
        self.proj = nn.Linear(cfg.engram_dim, cfg.hidden_size, bias=False)
        self.gate = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.ngram = cfg.engram_ngram
        self.table_size = cfg.engram_table

    def forward(self, x, ids, prev_ids=None):
        B, S, _ = x.shape
        if prev_ids is not None and S == 1:
            h = (prev_ids[:, None] * 1000003 + ids) % self.table_size
        else:
            prev = torch.zeros_like(ids)
            prev[:, 1:] = ids[:, :-1]
            h = (prev * 1000003 + ids) % self.table_size
        h = h.clamp(0, self.table_size - 1)
        emb = self.table(h)                                   # [B,S,dim]
        g = torch.sigmoid(self.gate(x))
        return x + g * self.proj(emb)


# ---------------------------------------------------------------- block
class Block(nn.Module):
    def __init__(self, cfg: V4Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.attn = MLAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.ffn = MoE(cfg)
        self.hc = HyperConnection(cfg.hidden_size, cfg.hc_mult) if cfg.hc_mult > 1 else None
        self.engram = Engram(cfg) if layer_id in cfg.engram_layers else None

    def forward(self, streams, ids, cos, sin, past=None, attn_mask=None, use_cache=False, prev_ids=None):
        # streams [B,S,hc,d]
        if self.hc is not None:
            inp = self.hc.pre_mix(streams)
        else:
            inp = streams[:, :, 0]
        if self.engram is not None:
            inp = self.engram(inp, ids, prev_ids)
        h = self.attn_norm(inp)
        a, new_past = self.attn(h, cos, sin, past=past, attn_mask=attn_mask, use_cache=use_cache)
        if self.hc is not None:
            streams = self.hc.post_mix(streams, a)
            inp2 = self.hc.pre_mix(streams)
        else:
            streams = streams + a.unsqueeze(2)
            inp2 = streams[:, :, 0]
        m, _, _ = self.ffn(self.ffn_norm(inp2))
        if self.hc is not None:
            streams = self.hc.post_mix(streams, m)
            out = streams
        else:
            streams = streams + m.unsqueeze(2)
            out = streams
        return out, new_past


# ---------------------------------------------------------------- MTP
class MTPBlock(nn.Module):
    """DFlash 风格 MTP：融合 h_t 与 emb(x_{t+1})，预测 x_{t+2}，附 confidence 头。"""

    def __init__(self, cfg: V4Config):
        super().__init__()
        d = cfg.hidden_size
        self.hn = RMSNorm(d, cfg.rms_eps)
        self.en = RMSNorm(d, cfg.rms_eps)
        self.fuse = nn.Linear(2 * d, d, bias=False)
        self.norm = RMSNorm(d, cfg.rms_eps)
        self.attn = MLAttention(cfg)
        self.ffn_norm = RMSNorm(d, cfg.rms_eps)
        self.ffn = Expert(d, cfg.moe_inter)
        self.conf = nn.Linear(d, 1)

    def forward(self, h, emb_next, cos, sin, attn_mask=None):
        x = self.fuse(torch.cat([self.hn(h), self.en(emb_next)], dim=-1))
        x = x + self.attn(self.norm(x), cos, sin, attn_mask=attn_mask)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x, torch.sigmoid(self.conf(x))


# ---------------------------------------------------------------- vision
class ViT(nn.Module):
    def __init__(self, cfg: V4Config):
        super().__init__()
        p, d = cfg.patch_size, cfg.vision_dim
        self.patch = nn.Conv2d(3, d, p, p)
        n = (cfg.image_size // p) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, cfg.vision_heads, d * 4, dropout=0.0,
                                       batch_first=True, norm_first=True,
                                       activation="gelu")
            for _ in range(cfg.vision_depth)
        ])
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Sequential(nn.Linear(d, cfg.hidden_size), nn.GELU(),
                                  nn.Linear(cfg.hidden_size, cfg.hidden_size))

    def forward(self, images):
        x = self.patch(images).flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.proj(self.norm(x))          # [B, 64, hidden]


# ---------------------------------------------------------------- model
class WuwaV4(nn.Module):
    def __init__(self, cfg: V4Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.hc_mult > 1:
            self.hc_head = nn.Linear(cfg.hidden_size, cfg.hc_mult, bias=False)
            nn.init.zeros_(self.hc_head.weight)
            self.hc_head.weight.data[:, 0] = 1.0
        self.mtp = MTPBlock(cfg) if cfg.mtp else None
        self.vision = ViT(cfg)
        self.image_emb = nn.Parameter(torch.zeros(cfg.image_tokens, cfg.hidden_size))
        nn.init.normal_(self.image_emb, std=0.02)

    def _embed(self, ids, image_feats=None):
        h = self.tok_emb(ids)
        if image_feats is not None:
            # image_feats: [B, 64, d]，替换 <image> 后连续 64 个位置
            for b in range(ids.size(0)):
                pos = (ids[b] == self.cfg.image_token_id).nonzero(as_tuple=True)[0]
                if len(pos) == 0:
                    continue
                start = int(pos[0]) + 1
                n = min(self.cfg.image_tokens, ids.size(1) - start)
                h[b, start:start + n] = image_feats[b, :n]
        return h

    def forward(self, ids, images=None, labels=None, mtp_labels=None, use_cache=False, past=None):
        cfg = self.cfg
        B, S = ids.shape
        h = self._embed(ids, self.vision(images) if images is not None else None)
        streams = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        cos, sin = build_rope(max(S, 1) + 64, cfg.qk_rope_head_dim, cfg.rope_theta, ids.device)

        n_past = past[0][0][0].size(2) if past is not None else 0
        new_past = []
        for i, blk in enumerate(self.blocks):
            p = past[i] if past is not None else None
            attn_mask = None
            if not use_cache:
                attn_mask = build_causal_mask(S, S, cfg, ids.device, streams.dtype)
            streams, np_ = blk(streams, ids, cos, sin, past=p, attn_mask=attn_mask, use_cache=use_cache)
            new_past.append(np_)

        if cfg.hc_mult > 1:
            w = F.softmax(self.hc_head(streams.mean(2)), dim=-1)
            h = (streams * w.unsqueeze(-1)).sum(2)
        else:
            h = streams[:, :, 0]
        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        mtp_logits = None
        if labels is not None:
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(shift_logits.reshape(-1, cfg.vocab_size),
                                   shift_labels.reshape(-1), ignore_index=-100)
            if self.mtp is not None and mtp_labels is not None:
                emb_next = self.tok_emb(ids[:, 1:])
                mh, conf = self.mtp(h[:, :-1], emb_next,
                                    cos[:S - 1], sin[:S - 1],
                                    attn_mask=build_causal_mask(S - 1, S - 1, cfg, ids.device, h.dtype))
                mtp_logits = self.lm_head(mh)
                m_loss = F.cross_entropy(mtp_logits.reshape(-1, cfg.vocab_size),
                                         mtp_labels[:, :-1].reshape(-1), ignore_index=-100)
                valid = mtp_labels[:, :-1] != -100
                if valid.any():
                    with torch.no_grad():
                        tgt = (mtp_logits.argmax(-1) == mtp_labels[:, :-1])[valid]
                    c_loss = F.binary_cross_entropy(conf.squeeze(-1)[valid], tgt.float())
                else:
                    c_loss = torch.zeros((), device=h.device)
                loss = loss + 0.2 * m_loss + 0.05 * c_loss
        return {"logits": logits, "loss": loss, "past": new_past if use_cache else None,
                "mtp_logits": mtp_logits}

    # ------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, ids, images=None, max_new=64, temperature=0.8, top_p=0.9,
                 top_k=40, stop_ids=(), use_cache=True, repetition_penalty=1.1,
                 think_log=None):
        cfg = self.cfg
        B, S = ids.shape
        img_feats = self.vision(images) if images is not None else None
        h = self._embed(ids, img_feats)
        streams = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        cos, sin = build_rope(S + max_new + 16, cfg.qk_rope_head_dim, cfg.rope_theta, ids.device)
        past = None
        out = ids.clone()
        cur = ids
        prev_id = None
        for step in range(max_new):
            seq_len = cur.size(1)
            new_past = []
            x = streams
            for i, blk in enumerate(self.blocks):
                p = past[i] if past is not None else None
                mask = None
                if p is None:
                    mask = build_causal_mask(seq_len, seq_len, cfg, ids.device, x.dtype)
                x, np_ = blk(x, cur, cos, sin, past=p, attn_mask=mask, use_cache=True, prev_ids=prev_id)
                new_past.append(np_)
            past = new_past
            prev_id = cur[:, -1]
            if cfg.hc_mult > 1:
                w = F.softmax(self.hc_head(x.mean(2)), dim=-1)
                hh = (x * w.unsqueeze(-1)).sum(2)
            else:
                hh = x[:, :, 0]
            hh = self.norm(hh)
            logits = self.lm_head(hh)[:, -1].float()
            logits = logits / max(temperature, 1e-5)
            # repetition penalty
            for b in range(B):
                uniq = torch.unique(out[b])
                logits[b, uniq] /= repetition_penalty
            probs = F.softmax(logits, dim=-1)
            # top-k / top-p
            k = min(top_k, probs.size(-1))
            v, idx = probs.topk(k, dim=-1)
            v = v / v.sum(-1, keepdim=True)
            nxt = idx.gather(-1, torch.multinomial(v, 1))
            out = torch.cat([out, nxt], dim=1)
            cur = nxt
            streams = self.tok_emb(nxt).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
            if think_log is not None:
                think_log.append(int(nxt[0, 0]))
            if nxt[0, 0].item() in stop_ids:
                break
        return out


def count_params(m):
    return sum(p.numel() for p in m.parameters())
