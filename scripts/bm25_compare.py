#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BM25 만으로 골든셋 40문항을 풀어 토크나이저 둘을 비교한다. 오늘의 핵심 증거.

    python scripts/bm25_compare.py            # 공백 분리 vs Kiwi 형태소

Dense 검색을 쓰지 않으므로 Qdrant 를 열지 않는다(노트북이 열려 있어도 된다).
결과는 results/bm25_whitespace.json, results/bm25_kiwi.json.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from finrag.eval import harness
    from finrag.index.bm25_ko import BM25Index, tokenize, whitespace_tokenize
    from finrag.retrieval.hybrid import _chunks

    chunks = _chunks()
    print(f"청크 {len(chunks):,}개로 BM25 인덱스를 두 번 만듭니다 (공백 / Kiwi)")

    results = {}
    for name, tok in (("공백 분리", whitespace_tokenize), ("Kiwi 형태소", tokenize)):
        t = time.perf_counter()
        idx = BM25Index.build(chunks, tokenizer=tok)
        build_s = time.perf_counter() - t
        vocab = len(idx.bm25.idf)
        pipeline = (lambda idx=idx, tok=tok: (lambda q: {"chunk_ids": [cid for cid, _ in idx.search(q, k=10, tokenizer=tok)]}))()
        key = "bm25_whitespace" if tok is whitespace_tokenize else "bm25_kiwi"
        res = harness.run(pipeline, key)
        harness.save(res)
        results[name] = (res, build_s, vocab)
        print(f"  {name}: 인덱스 {build_s:.1f}초, 어휘 {vocab:,}개")

    print(f"\n{'토크나이저':12}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR':>7}   조항번호형 R@5   상품명조건형 R@5")
    for name, (res, _, _) in results.items():
        o = res["overall"]; bt = res["by_type"]
        a = bt.get("조항번호형", {}).get("recall@5", 0); p = bt.get("상품명조건형", {}).get("recall@5", 0)
        print(f"{name:12}{o['recall@1']:>7.1%}{o['recall@5']:>7.1%}{o['recall@10']:>7.1%}{o['mrr']:>7.3f}   {a:>12.1%}   {p:>14.1%}")
    print("\n이 두 줄의 차이가 토크나이저의 값어치입니다. 채팅에 붙여 주세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
