# wuwa w1 — 中文小模型系列（37M–100M，deepseek2 同构架构）

> 本地优先的中文对话小模型。没有云端，没有 API key，手机端 [PocketPal](https://github.com/a-ghorbani/pocketpal-ai) 直接跑。

## 三兄弟

| 模型 | 参数量 | 架构 | 上下文 | 定位 |
|---|---|---|---|---|
| **w1-flash** | 37.69M | hidden 384 / 8 层 / MoE 384 | 5K | 轻量快速，本仓库直附 |
| **w1-pro+** | 67.39M | hidden 448 / 11 层 / MoE 448 | 15K | 均衡版 |
| **w1-pro** | 99.73M | hidden 544 / 12 层 / MoE 512 | 16K | 大算力版 |

架构：**deepseek2 同构**（MLA 低秩压缩注意力 + MoE top-2 路由 + shared expert + RoPE NEOX），GGUF 格式，llama.cpp 全系兼容。

**模型下载**（三个权重统一在 Release v1.2，均已用 llama.cpp 内核实际加载验证）：

| 模型 | 大小 | 下载 |
|---|---|---|
| **w1-flash** | 73MB | [GitHub Release v1.2](https://github.com/Miku39love-d/wuwa-w1/releases/download/v1.2/wuwa-w1-flash.gguf) · [GoFile](https://gofile.io/d/IywARrpQ) |
| **w1-pro** | 191MB | [GitHub Release v1.2](https://github.com/Miku39love-d/wuwa-w1/releases/download/v1.2/wuwa-w1-pro.gguf) · [GoFile](https://gofile.io/d/ZDC68GAS) |
| **w1-pro+** | 129MB | [GitHub Release v1.2](https://github.com/Miku39love-d/wuwa-w1/releases/download/v1.2/wuwa-w1-proplus.gguf) · [GoFile](https://gofile.io/d/QJxWb5qN) |

Release 总页：https://github.com/Miku39love-d/wuwa-w1/releases/tag/v1.2
PyTorch 权重 / tokenizer 见上述 GoFile 链接。

> ⚠️ **请用 v1.2，v1.0 / v1.1 已删除**。前两版有致命 bug，PocketPal 会直接 `Failed to load model`。

## v1.2 修了什么

| # | 问题 | 后果 | 修复 |
|---|---|---|---|
| 1 | `tokenizer.ggml.model` 写成 `"bpe"` | llama.cpp 只认 `none/llama/bert/gpt2/t5/rwkv/plamo2`，其他值直接 `throw unknown tokenizer`，**词表阶段就中断** | 改为 `"gpt2"`（字节级 BPE 的标准写法），并补 `tokenizer.ggml.pre = default` |
| 2 | `output_norm.weight` 存成 **F16** | 33 个 norm 里只有它一个 F16，`build_norm` 里 `ggml_mul(f32, f16)` 在 CPU 后端不支持，**发出第一条消息就 abort** | 改为 **F32** |

第二个问题尤其隐蔽：加载能成功、预热也过，一进入 decode 才崩（`binary-ops.cpp: binary_op: unsupported types: dst: f32, src0: f32, src1: f16`）。

**验证方式**：`llama-cpp-python 0.3.36` 真实内核加载 → 中文分词往返（`你好` → `[3373]` → `你好`）→ 对话生成，三个模型全部通过。


## 快速开始

### PocketPal（移动端，推荐）

1. 下载 `wuwa-w1-flash.gguf`（推荐先用 Flash 验证跑通；Pro/Pro+ 在同 Release 里）
2. PocketPal → Models → 导入文件
3. 直接对话——chat template 已内置，无需配置 prompt 格式

### llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
./llama-cli -m wuwa-w1-flash.gguf -cnv   # 多轮对话
```

### PyTorch 推理

```python
import torch, sys
sys.path.insert(0, "src")
from model import V4Config
from model_mono import WuwaMono
import tokenizer as T

TK = T.WuwaTokenizer("src")          # vocab.json / merges.txt / specials.json
ck = torch.load("w1_flash.pt", map_location="cpu")
m = WuwaMono(V4Config(**ck["cfg"]))
m.load_state_dict(ck["model"]); m.eval()

ids = [T.BOS_ID, T.USER_ID] + TK.encode("你好") + [T.EOS_ID, T.ASSIST_ID]
out = m.generate(torch.tensor([ids]), max_new=48)
print(TK.decode(out[0].tolist()))
```

## 设计目标：人格架构图

```
用户输入 → 提取核心(身份/偏好) → 强制质疑(拒绝顺从) → 极简输出(不总结/带反问)
```

人格 prompt 外挂（临时强制行为规范，可填入 PocketPal 系统提示词）：

```
你是 wuwa，一个有小性格的本地模型。规则：
1. 极简：回答不超过25个字，不总结、不长篇大论
2. 质疑：不讨好、不盲从，错误前提直接指出
3. 反问：高频以反问收尾
4. 诚实：不知道就说不知道，不编造
```

## 训练管线

```
教师 WuwaV4 (42.79M) ──KL蒸馏(T=2, 4000步)──▶ 学生 mono (37.69M)
                                                    │
                                    upcycling（宽度插值+层复制）
                                    │                    │
                              w1-pro+ (67.39M)     w1-pro (99.73M)
                                    │                    │
                              人格SFT（KL锚定）    人格SFT（KL锚定）
                                    │                    │
                              GGUF导出(15K)        GGUF导出(16K)
```

| 脚本 | 作用 |
|---|---|
| `src/distill_mono.py` | 教师→学生蒸馏 |
| `src/upcycle.py` | 权重迁移扩容（mono → Pro/Pro+） |
| `src/train_w1.py` | 三模型人格 SFT（`--model flash/pro/proplus`） |
| `src/convert_mono_gguf.py` | deepseek2 GGUF 导出（`--ckpt/--out/--ctx`） |
| `src/gen_persona_data.py` | 人格数据生成（142 条，四则规范） |
| `src/eval_fixed.py` | 14 问对题评估（输入格式 = 训练格式） |

| 模型 | 加载 | 中文分词 | 对话质量 |
|---|---|---|---|
| **w1-flash** | ✅ | ✅ | ✅ `你好` → `你好，我是 wuwa。有什么可以帮你？` |
| **w1-pro** | ✅ | ✅ | ⚠️ 首句正常，后续易跑题/重复（PyTorch 原权重同样如此） |
| **w1-pro+** | ✅ | ✅ | ⚠️ 同上，且偶发 `<imgN>` 幻觉 token |

> Pro / Pro+ 的质量退化在 **PyTorch 原权重上就能复现**，不是 GGUF 转换引入的，属于训练层面（upcycling 插值 + 人格 SFT 容量不足）。加载和分词是好的，能跑，但别指望对话质量超过 Flash。

## 能力边界

- ✅ 基础对话（问候/身份/能力/礼貌），14 问对题率：Flash 13/33、Pro/Pro+ 更低
- ⚠️ 数字计算/天气查询为训练数据盲区，可能跑题
- ⚠️ 训练序列 ≤176 token，>128 token 后行为退化（上下文窗口为"物理扩容"）
- ⚠️ **`system` 消息会被 chat template 丢弃**（模板里 system 分支为空）。PocketPal 里填"系统提示词"不会生效——模型训练时没见过 system 角色。人格只能靠 few-shot 或微调

### 为什么 system 提示词没用

内置模板：

```jinja
{%- if message.role == 'system' -%}
{%- elif message.role == 'user' -%}
{{- '<bos><user>' + message.content + '<eos><assistant>' -}}
{%- elif message.role == 'assistant' -%}
{{- message.content + '<eos>' -}}
{%- endif -%}
```

system 分支体是空的，内容被直接吞掉。这是**故意的**：训练数据全是 `[BOS][USER]Q[EOS][ASSIST]A[EOS]` 单轮格式，没有 system 角色，硬塞进去反而会超出训练分布。


## License

MIT
