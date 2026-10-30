"""wuwa-v4 分词器（动态 ID 布局）。

布局（由 vocab.json 的实际大小决定）：
  0 .. V-1               BPE 文本
  V .. V+11              特殊标记（按 SPECIALS 顺序追加）
  V+12 .. V+267          图像码（256）
"""
from __future__ import annotations

import json
import os

from tokenizers import ByteLevelBPETokenizer

SPECIALS = [
    "<pad>", "<bos>", "<eos>", "<user>", "<assistant>",
    "<think>", "</think>", "<image>", "<patch>", "<draw>",
    "<img_start>", "<img_end>",
]
N_SPECIAL = len(SPECIALS)
IMAGE_CODE_N = 256

_HERE = os.path.dirname(os.path.abspath(__file__))
_VOCAB_FILE = os.path.join(_HERE, "ckpt", "tokenizer", "vocab.json")
if os.path.exists(_VOCAB_FILE):
    _V = len(json.load(open(_VOCAB_FILE, encoding="utf-8")))
else:
    _V = 3518

SPECIAL_BASE = _V
PAD_ID = SPECIAL_BASE + 0
BOS_ID = SPECIAL_BASE + 1
EOS_ID = SPECIAL_BASE + 2
USER_ID = SPECIAL_BASE + 3
ASSIST_ID = SPECIAL_BASE + 4
THINK_ID = SPECIAL_BASE + 5
NTHINK_ID = SPECIAL_BASE + 6
IMAGE_ID = SPECIAL_BASE + 7
PATCH_ID = SPECIAL_BASE + 8
DRAW_ID = SPECIAL_BASE + 9
IMG_START_ID = SPECIAL_BASE + 10
IMG_END_ID = SPECIAL_BASE + 11
IMAGE_CODE_BASE = SPECIAL_BASE + N_SPECIAL
VOCAB_SIZE = IMAGE_CODE_BASE + IMAGE_CODE_N


class WuwaTokenizer:
    def __init__(self, model_dir: str | None = None):
        self.dir = model_dir or os.path.join(_HERE, "ckpt", "tokenizer")
        self.tok = ByteLevelBPETokenizer(
            os.path.join(self.dir, "vocab.json"),
            os.path.join(self.dir, "merges.txt"),
        )
        self.tok.add_special_tokens(SPECIALS)
        self.v = self.tok.get_vocab_size() - N_SPECIAL
        self.specials = {i: s for i, s in zip(range(SPECIAL_BASE, SPECIAL_BASE + N_SPECIAL), SPECIALS)}

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    @staticmethod
    def train(files: list[str], out_dir: str, vocab_size: int = 4000, min_freq: int = 2):
        os.makedirs(out_dir, exist_ok=True)
        t = ByteLevelBPETokenizer()
        t.train(files=files, vocab_size=vocab_size, min_frequency=min_freq, show_progress=False)
        t.save_model(out_dir)
        with open(os.path.join(out_dir, "specials.json"), "w") as f:
            json.dump(SPECIALS, f, ensure_ascii=False)
        return out_dir

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        ids = list(self.tok.encode(text, add_special_tokens=False).ids)
        return ([BOS_ID] if add_bos else []) + ids

    def decode(self, ids: list[int]) -> str:
        out, buf = [], []
        for i in ids:
            if i in self.specials:
                if buf:
                    out.append(self.tok.decode(buf)); buf = []
                out.append(self.specials[i])
            elif IMAGE_CODE_BASE <= i < IMAGE_CODE_BASE + IMAGE_CODE_N:
                if buf:
                    out.append(self.tok.decode(buf)); buf = []
                out.append(f"<img{i - IMAGE_CODE_BASE}>")
            else:
                buf.append(i)
        if buf:
            out.append(self.tok.decode(buf))
        return "".join(out)

    def image_ids(self, codes) -> list[int]:
        return [IMAGE_CODE_BASE + int(c) for c in codes]

    def is_image_code(self, t: int) -> bool:
        return IMAGE_CODE_BASE <= t < IMAGE_CODE_BASE + IMAGE_CODE_N


if __name__ == "__main__":
    tk = WuwaTokenizer()
    print("vocab:", tk.vocab_size, "| special_base:", SPECIAL_BASE)
    s = "你好，我是 wuwa。<think>30 × 8 = 240</think>答案 240。"
    ids = tk.encode(s)
    print(len(ids), ids[:10])
    print(repr(tk.decode(ids)))
