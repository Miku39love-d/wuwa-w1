# wuwa w1 — 中文小模型系列（37M–100M，deepseek2 同构架构）

> 本地优先的中文对话小模型。没有云端，没有 API key，手机端 [PocketPal](https://github.com/a-ghorbani/pocketpal-ai) 直接跑。

## 三兄弟

| 模型 | 参数量 | 架构 | 上下文 | 定位 |
|---|---|---|---|---|
| **w1-flash** | 37.69M | hidden 384 / 8 层 / MoE 384 | 5K | 轻量快速，本仓库直附 |
| **w1-pro+** | 67.39M | hidden 448 / 11 层 / MoE 448 | 15K | 均衡版 |
| **w1-pro** | 99.73M | hidden 544 / 12 层 / MoE 512 | 16K | 大算力版 |

架构：**deepseek2 同构**（MLA 低秩压缩注意力 + MoE top-2 路由 + shared expert + RoPE NEOX），GGUF 格式，llama.cpp 全系兼容。

**模型下载**（Flash 已内置本仓库，Pro/Pro+ 走 GoFile）：

| 模型 | 大小 | 下载 |
|---|---|---|
| **w1-flash** | 73MB | 本仓库 `wuwa-w1-flash.gguf`（直下）· [GoFile](https://gofile.io/d/IywARrpQ) |
| **w1-pro** | 191MB | [GoFile](https://gofile.io/d/ZDC68GAS) |
| **w1-pro+** | 129MB | [GoFile](https://gofile.io/d/QJxWb5qN) |

PyTorch 权重 / tokenizer 同见上述 GoFile 链接。

## 快速开始

### PocketPal（移动端，推荐）

1. 下载 `wuwa-w1-flash.gguf`（或 HF 上的 Pro/Pro+）
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

## 能力边界

- ✅ 基础对话（问候/身份/能力/礼貌），14 问对题率 4–5/14
- ⚠️ 数字计算/天气查询为训练数据盲区，可能跑题
- ⚠️ 训练序列 ≤176 token，>128 token 后行为退化（上下文窗口为"物理扩容"）

## License

MIT
