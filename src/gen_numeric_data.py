"""合成数字/比较/选择类简短对话样本，补训练数据盲区（原 2600 条 chat 零数字）。
风格对齐现有数据：assistant 简短完整、带 EOS、无代码无 <think>。
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "numeric_chat.jsonl")

CN = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九",
      10: "十", 11: "十一", 12: "十二", 20: "二十", 50: "五十", 100: "一百", 200: "两百"}


def cn(n):
    return CN.get(n, str(n))


def main():
    random.seed(20241024)
    rows = []

    # 1) 大小比较（问法多样，答案选大的）
    pairs = [(100, 50), (100, 99), (80, 20), (60, 40), (90, 10), (75, 25),
             (45, 15), (30, 30), (12, 8), (55, 45), (200, 100), (36, 6),
             (18, 9), (64, 16), (88, 11), (25, 5), (70, 7), (48, 12)]
    tpl_cmp = ["{} 和 {} 哪个大？", "{} 比 {} 大吗？", "{} 和 {}，哪个数字更大？", "{} 块钱和 {} 块钱哪个多？"]
    for a, b in pairs:
        for t in random.sample(tpl_cmp, 2):
            if a > b:
                ans = f"{a} 更大。" if "大" in t or "多" in t else f"是的，{a} 比 {b} 大。"
                if "大吗" in t:
                    ans = f"是的，{a} 比 {b} 大。"
                elif "多" in t:
                    ans = f"{a} 块更多。"
                else:
                    ans = f"{a} 更大。"
            else:
                ans = f"一样大。"
            rows.append({"type": "chat", "user": t.format(a, b), "assistant": ans})

    # 2) 选择（都选多的/好的）
    pairs2 = [(100, 50), (200, 100), (80, 30), (60, 20), (99, 9), (45, 5)]
    tpl_sel = ["{} 块钱和 {} 块钱，你选哪个？", "给你 {} 块或者 {} 块，你要哪个？",
               "只能拿一个：{} 还是 {}？", "{} 元和 {} 元，你拿哪个？"]
    for a, b in pairs2:
        t = random.choice(tpl_sel)
        rows.append({"type": "chat", "user": t.format(a, b), "assistant": f"选 {a} 块。"})

    # 3) 简单算术（加法/减法，结果 ≤ 20）
    for _ in range(70):
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        if random.random() < 0.6:
            rows.append({"type": "chat",
                         "user": f"{a} 加 {b} 等于几？",
                         "assistant": f"等于 {a + b}。"})
        else:
            if a < b:
                a, b = b, a
            rows.append({"type": "chat",
                         "user": f"{a} 减 {b} 等于几？",
                         "assistant": f"等于 {a - b}。"})

    # 4) 数字序列/常识
    seqs = [(1, 2), (5, 6), (9, 10), (19, 20), (49, 50), (99, 100)]
    for n, nxt in seqs:
        rows.append({"type": "chat", "user": f"{n} 后面是几？", "assistant": f"是 {nxt}。"})
    for q, a in [("100 里面有几个 10？", "有 10 个。"),
                 ("1 米等于多少厘米？", "等于 100 厘米。"),
                 ("一天有几个小时？", "有 24 个小时。"),
                 ("一周有几天？", "有 7 天。"),
                 ("一只手有几根手指？", "有 5 根。"),
                 ("1 千克等于多少克？", "等于 1000 克。")]:
        rows.append({"type": "chat", "user": q, "assistant": a})

    # 打乱 + 去重
    random.shuffle(rows)
    seen, uniq = set(), []
    for r in rows:
        k = (r["user"], r["assistant"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"生成 {len(uniq)} 条数字类样本 -> {OUT}")
    lens = [len(r["assistant"]) for r in uniq]
    print(f"assistant 长度: max {max(lens)} mean {sum(lens)/len(lens):.1f}")


if __name__ == "__main__":
    main()
