"""真机（真 llama.cpp 内核）验证：加载 + 分词 + 生成
用法: python3 test_load_llamacpp.py <gguf> [n_ctx]
"""
import sys
import time
import llama_cpp

path = sys.argv[1]
n_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 512

print(f"=== 加载 {path} (n_ctx={n_ctx}) ===")
t0 = time.time()
try:
    llm = llama_cpp.Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_threads=8,
        n_gpu_layers=0,
        verbose=True,
        logits_all=False,
    )
except Exception as e:
    print(f"\n>>> 加载失败: {type(e).__name__}: {e}")
    raise SystemExit(1)
print(f">>> 加载成功，耗时 {time.time()-t0:.1f}s")

# 分词验证：中文必须能正常往返
for s in ["你好", "今天天气不错", "你是谁"]:
    ids = llm.tokenize(s.encode(), add_bos=False, special=True)
    back = llm.detokenize(ids).decode("utf-8", errors="replace")
    flag = "OK" if back == s else "MISMATCH"
    print(f"[tok] {s!r} -> {ids} -> {back!r}  {flag}")

# 生成验证：正确格式 [BOS][USER] q [EOS][ASSIST]
for q in ["你好", "你是谁"]:
    prompt = f"<bos><user>{q}<eos><assistant>"
    ids = llm.tokenize(prompt.encode(), add_bos=False, special=True)
    print(f"\n[gen] prompt={prompt!r} tokens={len(ids)}")
    out = llm.create_completion(
        prompt, max_tokens=24, temperature=0.7, top_p=0.9, repeat_penalty=1.1,
        stop=["<eos>", "<bos>", "<user>"], echo=False,
    )
    print(f"[gen] -> {out['choices'][0]['text']!r}")

print("\nRESULT: RUNTIME OK ✅")
