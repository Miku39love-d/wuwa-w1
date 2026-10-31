"""WuwaMono(学生) -> deepseek2 架构 GGUF 转换
依据 llama.cpp conversion/deepseek.py + gguf-py constants.py 的 DEEPSEEK2 规范:
- MLA: kv_a_proj -> attn_kv_a_mqa; kv_b_proj 拆分为 attn_k_b(转置) + attn_v_b
- MoE: 8 专家 stack 成 3D (ffn_gate_exps/ffn_up_exps/ffn_down_exps); shared -> ffn_*_shexp
- o_down/o_up 折叠为 attn_output
"""
import json
import os

import torch
from gguf import GGUFWriter

HERE = os.path.dirname(os.path.abspath(__file__))
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt", "mono_best.pt"))
_ap.add_argument("--out", default=os.path.join(HERE, "release", "wuwa-mono-deepseek2.gguf"))
_ap.add_argument("--ctx", type=int, default=5120)
_args = _ap.parse_args()
CKPT = _args.ckpt
TOK_DIR = os.path.join(HERE, "ckpt", "tokenizer")
OUT = _args.out

# ---------------------------------------------------------------- load
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
sd = ck["model"]

V = len(json.load(open(os.path.join(TOK_DIR, "vocab.json"), encoding="utf-8")))
SPECIALS = json.load(open(os.path.join(TOK_DIR, "specials.json"), encoding="utf-8"))
N_SPECIAL = len(SPECIALS)
IMAGE_CODE_N = 256
VOCAB_SIZE = V + N_SPECIAL + IMAGE_CODE_N
SPECIAL_BASE = V
IMAGE_CODE_BASE = V + N_SPECIAL

NH = cfg["num_attention_heads"]  # 8
QK_NOPE = cfg["qk_nope_head_dim"]  # 16
QK_ROPE = cfg["qk_rope_head_dim"]  # 16
VD = cfg["v_head_dim"]  # 48
KV_LORA = cfg["kv_lora_rank"]  # 96
N_EXP = cfg["n_experts"]  # 8
D = cfg["hidden_size"]  # 384

# ---------------------------------------------------------------- writer
os.makedirs(os.path.dirname(OUT), exist_ok=True)
w = GGUFWriter(OUT, arch="deepseek2")

# --- general
w.add_name("wuwa-mono")
w.add_description(
    "WuwaMono: distilled from WuwaV4 (DeepSeek-V4.1-Flash-style architecture). "
    "MLA attention + MoE(top-2) + shared expert, deepseek2-compatible GGUF."
)
w.add_file_type(1)  # MOSTLY_F16

# --- deepseek2 metadata（键名对齐 gguf-py add_* 方法）
w.add_context_length(_args.ctx)   # 5K 上下文窗口（用户要求；模型训练序列≤176、SWA=128，长程能力见交付说明）
w.add_embedding_length(D)
w.add_block_count(cfg["num_hidden_layers"])
w.add_feed_forward_length(cfg["moe_inter"])
w.add_expert_feed_forward_length(cfg["moe_inter"])
w.add_head_count(NH)
w.add_head_count_kv(1)   # 官方 conversion/deepseek.py: MLA 转为 MQA（1 组 KV），必须为 1
w.add_key_length(KV_LORA + QK_ROPE)          # 112（官方 conversion/deepseek.py: kv_lora_rank + qk_rope_head_dim）
w.add_value_length(KV_LORA)                   # 96（官方 conversion/deepseek.py: kv_lora_rank）
w.add_key_length_mla(QK_NOPE + QK_ROPE)      # 32
w.add_value_length_mla(VD)                    # 48
w.add_q_lora_rank(cfg["q_lora_rank"])         # 192
w.add_kv_lora_rank(KV_LORA)                   # 96
w.add_rope_dimension_count(QK_ROPE)           # 16
w.add_rope_freq_base(cfg["rope_theta"])
w.add_layer_norm_rms_eps(cfg["rms_eps"])
w.add_vocab_size(VOCAB_SIZE)
w.add_expert_count(N_EXP)
w.add_expert_used_count(cfg["top_k"])
w.add_expert_shared_count(1)
w.add_expert_weights_scale(1.0)
w.add_expert_weights_norm(True)               # topv/topv.sum 归一化
w.add_leading_dense_block_count(0)            # 全 MoE
w.add_sliding_window(cfg["swa_window"])       # 128

# --- tokenizer
tokens = []
with open(os.path.join(TOK_DIR, "vocab.json"), encoding="utf-8") as f:
    vocab = json.load(f)
for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1]):
    tokens.append(tok)
assert len(tokens) == V
tokens.extend(SPECIALS)
tokens.extend(f"<img{i}>" for i in range(IMAGE_CODE_N))
token_type = [1] * V + [3] * N_SPECIAL + [3] * IMAGE_CODE_N

merges = []
with open(os.path.join(TOK_DIR, "merges.txt"), encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        a, _, b = line.partition(" ")
        if a and b:
            merges.append(a + " " + b)

# llama.cpp 的 vocab 加载只认枚举值：none/llama/bert/gpt2/t5/rwkv/plamo2
# 字节级 BPE 必须写 "gpt2"（写 "bpe" 会 throw unknown tokenizer: 'bpe'）
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("default")   # GPT-2 字节级 BPE 的标准 pre-tokenizer
w.add_token_list(tokens)
w.add_token_types(token_type)
if merges:
    w.add_token_merges(merges)
w.add_bos_token_id(SPECIAL_BASE + SPECIALS.index("<bos>"))
w.add_eos_token_id(SPECIAL_BASE + SPECIALS.index("<eos>"))
w.add_pad_token_id(SPECIAL_BASE + SPECIALS.index("<pad>"))
w.add_add_bos_token(False)
w.add_add_eos_token(False)

# --- chat template：严格匹配训练格式 [BOS][USER] user [EOS][ASSIST] assistant [EOS]
# 训练时每条样本都是单轮 [BOS][USER]Q[EOS][ASSIST]A[EOS]，
# 多轮对话时每轮 user 自带 <bos>，让模型每轮都落在训练分布内。
CHAT_TEMPLATE = (
    "{%- for message in messages -%}\n"
    "{%- if message.role == 'system' -%}\n"
    "{%- elif message.role == 'user' -%}\n"
    "{{- '<bos><user>' + message.content + '<eos><assistant>' -}}\n"
    "{%- elif message.role == 'assistant' -%}\n"
    "{{- message.content + '<eos>' -}}\n"
    "{%- endif -%}\n"
    "{%- endfor -%}"
)
w.add_chat_template(CHAT_TEMPLATE)

# ---------------------------------------------------------------- tensors
def to_f16(t):
    return t.to(torch.float16).numpy()

def to_f32(t):
    return t.to(torch.float32).numpy()

written = []

def add(name, t, f32=False):
    w.add_tensor(name, to_f32(t) if f32 else to_f16(t))
    written.append(name)

# 全局
add("token_embd.weight", sd["tok_emb.weight"])
add("output_norm.weight", sd["norm.weight"], f32=True)   # 必须 F32！build_norm 里 ggml_mul 不支持 f32×f16
add("output.weight", sd["lm_head.weight"])

for i in range(cfg["num_hidden_layers"]):
    p = f"blocks.{i}."
    b = f"blk.{i}."
    # norms
    add(b + "attn_norm.weight", sd[p + "attn_norm.weight"], f32=True)
    add(b + "ffn_norm.weight", sd[p + "ffn_norm.weight"], f32=True)
    # MLA
    add(b + "attn_q_a.weight", sd[p + "attn.q_a_proj.weight"])
    add(b + "attn_q_a_norm.weight", sd[p + "attn.q_a_norm.weight"], f32=True)
    add(b + "attn_q_b.weight", sd[p + "attn.q_b_proj.weight"])
    add(b + "attn_kv_a_mqa.weight", sd[p + "attn.kv_a_proj.weight"])
    add(b + "attn_kv_a_norm.weight", sd[p + "attn.kv_a_norm.weight"], f32=True)
    # kv_b 按 head 拆分：每 head 64 维 = [qk_nope(16) | v(48)]（与官方 conversion 一致）
    # attn_k_b: PyTorch [n_head, kv_lora, qk_nope]（GGUF ne=[qk_nope, kv_lora, n_head]）
    # attn_v_b: PyTorch [n_head, v, kv_lora]（GGUF ne=[kv_lora, v, n_head]）
    kv_b = sd[p + "attn.kv_b_proj.weight"]  # [NH*(QK_NOPE+VD), KV_LORA] = [512, 96]
    kv_b = kv_b.view(NH, QK_NOPE + VD, KV_LORA)
    k_b, v_b = torch.split(kv_b, [QK_NOPE, VD], dim=1)  # [8,16,96] / [8,48,96]
    add(b + "attn_k_b.weight", k_b.transpose(1, 2))   # [8, 96, 16]
    add(b + "attn_v_b.weight", v_b)                    # [8, 48, 96]
    # o_down/o_up 折叠为 attn_output: o_up @ o_down
    o_proj = sd[p + "attn.o_up.weight"] @ sd[p + "attn.o_down.weight"]  # [D, NH*VD]
    add(b + "attn_output.weight", o_proj)
    # MoE router
    add(b + "ffn_gate_inp.weight", sd[p + "ffn.gate.weight"])
    # experts -> 3D stack（w1=gate 进 silu, w3=up 乘数，与 HF/llama.cpp 一致）
    gates = torch.stack([sd[p + f"ffn.experts.{m}.w1.weight"] for m in range(N_EXP)], dim=0)
    ups = torch.stack([sd[p + f"ffn.experts.{m}.w3.weight"] for m in range(N_EXP)], dim=0)
    downs = torch.stack([sd[p + f"ffn.experts.{m}.w2.weight"] for m in range(N_EXP)], dim=0)
    add(b + "ffn_up_exps.weight", ups)      # [8, 384, 384]
    add(b + "ffn_gate_exps.weight", gates)
    add(b + "ffn_down_exps.weight", downs)
    # shared expert（w1=gate, w3=up）
    add(b + "ffn_gate_shexp.weight", sd[p + "ffn.shared.w1.weight"])
    add(b + "ffn_up_shexp.weight", sd[p + "ffn.shared.w3.weight"])
    add(b + "ffn_down_shexp.weight", sd[p + "ffn.shared.w2.weight"])

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()

size = os.path.getsize(OUT) / 1024 / 1024
print(f"OK -> {OUT} ({size:.1f} MB), {len(written)} tensors")
