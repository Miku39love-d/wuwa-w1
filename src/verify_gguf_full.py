"""GGUF 全量校验：tokenizer 枚举 / dtype 约束 / 张量形状 / 数值健康度
用法: python3 verify_gguf_full.py <file.gguf>
"""
import sys
import numpy as np
from gguf import GGUFReader

ALLOWED_TOK_MODEL = {"no_vocab", "none", "llama", "bert", "gpt2", "t5", "rwkv", "plamo2"}


def main(path):
    r = GGUFReader(path)
    fails, warns = [], []
    print(f"=== {path} ===")

    # ---------- 1. tokenizer ----------
    f = r.fields.get("tokenizer.ggml.model")
    tok_model = str(f.contents()) if f else None
    print(f"[tokenizer] model = {tok_model!r}")
    if tok_model not in ALLOWED_TOK_MODEL:
        fails.append(f"tokenizer.ggml.model={tok_model!r} 不在 llama.cpp 白名单 {sorted(ALLOWED_TOK_MODEL)}")

    pre = r.fields.get("tokenizer.ggml.pre")
    print(f"[tokenizer] pre   = {str(pre.contents()) if pre else 'MISSING(默认 default)'}")

    tk = r.fields.get("tokenizer.ggml.tokens")
    tt = r.fields.get("tokenizer.ggml.token_type")
    mg = r.fields.get("tokenizer.ggml.merges")
    vs = r.fields.get("deepseek2.vocab_size")
    n_tok = len(tk.contents()) if tk else -1
    print(f"[tokenizer] tokens={n_tok}  types={len(tt.contents()) if tt else -1}  merges={len(mg.contents()) if mg else 0}  vocab_size={int(vs.contents()) if vs else -1}")
    if vs and n_tok != int(vs.contents()):
        fails.append(f"tokens 长度 {n_tok} != vocab_size {int(vs.contents())}")
    if tt and len(tt.contents()) != n_tok:
        fails.append("token_type 长度与 tokens 不一致")

    # ---------- 2. dtype 约束 ----------
    # llama.cpp: norm 权重必须 F32（build_norm -> ggml_mul，不支持 f32*f16）
    # 其它矩阵权重 F16 或量化
    norm_f16, weight_bad = [], []
    dt_count = {}
    for t in r.tensors:
        dt = t.tensor_type.name
        dt_count[dt] = dt_count.get(dt, 0) + 1
        n = t.name
        is_norm = ("norm" in n) or n.endswith("_norm.weight")
        if is_norm:
            if dt != "F32":
                norm_f16.append((n, dt))
        else:
            if dt not in ("F16", "F32", "Q8_0", "Q4_K", "Q6_K"):
                weight_bad.append((n, dt))
    print(f"[dtype] {dt_count}")
    if norm_f16:
        fails.append(f"{len(norm_f16)} 个 norm 张量不是 F32: {norm_f16[:5]}")
    else:
        print(f"[dtype] ✅ 全部 {sum(1 for t in r.tensors if 'norm' in t.name)} 个 norm 张量为 F32")
    if weight_bad:
        fails.append(f"异常 dtype 张量: {weight_bad[:5]}")

    # ---------- 3. 形状 / 元数据一致性 ----------
    NUM_KEYS = [
        "deepseek2.embedding_length", "deepseek2.attention.head_count",
        "deepseek2.attention.head_count_kv", "deepseek2.attention.q_lora_rank",
        "deepseek2.attention.kv_lora_rank", "deepseek2.attention.key_length_mla",
        "deepseek2.attention.value_length_mla", "deepseek2.feed_forward_length",
        "deepseek2.expert_count", "deepseek2.block_count", "deepseek2.vocab_size",
    ]
    md = {}
    for k in NUM_KEYS:
        fld = r.fields.get(k)
        md[k] = int(fld.contents()) if fld is not None else None
    missing_meta = [k for k, v in md.items() if v is None]
    if missing_meta:
        fails.append(f"缺失元数据: {missing_meta}")
    D = md.get("deepseek2.embedding_length")
    NH = md.get("deepseek2.attention.head_count")
    NKV = md.get("deepseek2.attention.head_count_kv")
    QB = md.get("deepseek2.attention.q_lora_rank")
    KVL = md.get("deepseek2.attention.kv_lora_rank")
    KLMLA = md.get("deepseek2.attention.key_length_mla")
    VMLA = md.get("deepseek2.attention.value_length_mla")
    NF = md.get("deepseek2.feed_forward_length")
    NEXP = md.get("deepseek2.expert_count")
    NL = md.get("deepseek2.block_count")
    print(f"[meta] D={D} NH={NH} NKV={NKV} q_lora={QB} kv_lora={KVL} k_mla={KLMLA} v_mla={VMLA} ff={NF} exp={NEXP} layers={NL}")

    expect = {
        "token_embd.weight": [int(md["deepseek2.vocab_size"]), D],
        "output.weight": [int(md["deepseek2.vocab_size"]), D],
        "output_norm.weight": [D],
    }
    for i in range(NL):
        p = f"blk.{i}."
        expect.update({
            p + "attn_norm.weight": [D],
            p + "ffn_norm.weight": [D],
            p + "attn_q_a.weight": [QB, D],
            p + "attn_q_a_norm.weight": [QB],
            p + "attn_q_b.weight": [NH * KLMLA, QB],
            p + "attn_kv_a_mqa.weight": [KVL + KLMLA // 2, D],
            p + "attn_kv_a_norm.weight": [KVL],
            p + "attn_k_b.weight": [NH, KVL, KLMLA - KLMLA // 2],
            p + "attn_v_b.weight": [NH, VMLA, KVL],
            p + "attn_output.weight": [D, NH * VMLA],
            p + "ffn_gate_inp.weight": [NEXP, D],
            p + "ffn_up_exps.weight": [NEXP, NF, D],
            p + "ffn_gate_exps.weight": [NEXP, NF, D],
            p + "ffn_down_exps.weight": [NEXP, D, NF],
            p + "ffn_gate_shexp.weight": [NF, D],
            p + "ffn_up_shexp.weight": [NF, D],
            p + "ffn_down_shexp.weight": [D, NF],
        })

    tensors = {t.name: t for t in r.tensors}
    shape_bad = []
    for name, exp in expect.items():
        t = tensors.get(name)
        if t is None:
            shape_bad.append((name, "MISSING"))
            continue
        got = [int(x) for x in t.shape][::-1]  # GGUF ne0 在前(=in)，反转成常规 [out, in]
        if got != exp:
            shape_bad.append((name, got, exp))
    if shape_bad:
        fails.append(f"{len(shape_bad)} 个形状不符: {shape_bad[:4]}")
    else:
        print(f"[shape] ✅ {len(expect)}/{len(expect)} 张量形状全部符合 llama.cpp deepseek2 定义")

    extra = set(tensors) - set(expect)
    if extra:
        warns.append(f"多余张量: {sorted(extra)[:6]}")

    # ---------- 4. 数值健康 ----------
    nan_inf = []
    for t in r.tensors:
        if t.tensor_type.name not in ("F16", "F32"):
            continue
        a = t.data
        if np.isnan(a).any() or np.isinf(a).any():
            nan_inf.append(t.name)
    if nan_inf:
        fails.append(f"含 NaN/Inf: {nan_inf[:5]}")
    else:
        print("[num] ✅ 无 NaN / 无 Inf")

    # ---------- 5. 结论 ----------
    print()
    if fails:
        print("RESULT: FAIL ❌")
        for x in fails:
            print("  ✗", x)
    else:
        print("RESULT: PASS ✅")
    for x in warns:
        print("  ⚠", x)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
