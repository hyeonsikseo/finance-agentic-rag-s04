#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""검색을 손으로 한 번 해 본다. 같은 질문을 Dense 만 · Hybrid(BM25+RRF) · 리랭커까지로 비교한다.

    python scripts/search_demo.py "제31조 리볼빙"                       # Dense 만 (3회차)
    python scripts/search_demo.py "제31조 리볼빙" --hybrid              # Dense + Kiwi BM25 → RRF
    python scripts/search_demo.py "제31조 리볼빙" --hybrid --rerank     # + Cross-Encoder 리랭커
    python scripts/search_demo.py "연체이자율" --hybrid --latest       # 같은 상품은 최신 판본만
    python scripts/search_demo.py "중도해지이율" --hybrid --issuer 카카오뱅크

지금은 필터를 사람이 넣는다. 5회차에 LLM 이 질문에서 필터를 뽑아낸다(Self-Query).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--hybrid", action="store_true", help="Dense + Kiwi BM25 를 RRF 로 합친다")
    ap.add_argument("--rerank", action="store_true", help="후보를 Cross-Encoder 로 다시 줄 세운다 (--hybrid 와 같이)")
    ap.add_argument("--candidates", type=int, default=10, help="리랭커에 넣을 후보 수 (기본 10)")
    ap.add_argument("--latest", action="store_true", help="같은 발행사·상품·종류는 최신 판본만 남긴다")
    ap.add_argument("--issuer", help="발행사로 거른다 (예: 카카오뱅크)")
    ap.add_argument("--doc-type", help="문서 종류로 거른다 (예: 상품설명서, 약관, 특약)")
    args = ap.parse_args()

    from qdrant_client import models
    from finrag.index import qdrant
    from finrag.retrieval import dense, hybrid, rerank
    from finrag.retrieval.filters import pick_latest

    must = []
    if args.issuer:
        must.append(models.FieldCondition(key="issuer", match=models.MatchValue(value=args.issuer)))
    if args.doc_type:
        must.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=args.doc_type)))
    flt = models.Filter(must=must) if must else None

    # 모델 로드와 인덱스 읽기는 한 번만 드는 비용이라 시간에서 뺀다. 먼저 한 번 돌려 데운다.
    if args.hybrid:
        hybrid.search("준비", k=1, candidates=10)
    else:
        dense.search("준비", k=1)
    if args.rerank:
        warm = hybrid.search("준비", k=10, candidates=10)
        rerank.rerank("준비", warm, top_k=1)     # 첫 호출은 모델 준비 시간이 섞인다. 같은 후보 수로 한 번 돌려 둔다

    t0 = time.perf_counter()
    if args.hybrid:
        n = args.candidates if args.rerank else args.k
        hits = hybrid.search(args.query, k=n, candidates=max(n, 10), flt=flt)
        mode = "Hybrid(Dense + Kiwi BM25 → RRF)"
    else:
        hits = dense.search(args.query, k=args.k, flt=flt)
        mode = "Dense"
    search_ms = (time.perf_counter() - t0) * 1000

    stats = {}
    if args.rerank:
        hits, stats = rerank.rerank(args.query, hits, top_k=args.k)
        mode += " → 리랭커"
    if args.latest:
        hits = pick_latest(hits)
        mode += " → 최신 판본만"

    label = " · ".join(f"{k}={v}" for k, v in (("issuer", args.issuer), ("doc_type", args.doc_type)) if v) or "필터 없음"
    print(f"\n[{mode} · {label}] {args.query}")
    if stats.get("reranked"):
        print(f"  검색 {search_ms:.0f}ms, 리랭커 후보 {stats['candidates']}개 {stats['ms']}ms (후보당 {stats['ms_per_candidate']}ms)")
    elif args.rerank:
        print(f"  검색 {search_ms:.0f}ms  (리랭커 모델이 없어 건너뜀. make models-rerank)")
    else:
        print(f"  검색 {search_ms:.0f}ms")
    print(f"  {'점수':>6}  {'문서':28} {'종류':6} {'시행일':10}  {'조항':8}  본문")
    for h in hits[:args.k]:
        score = h.get("rerank_score", h["score"])
        text = " ".join(h["text"].split())[:28]
        print(f"  {score:6.3f}  {h['doc_id']:28} {h['doc_type']:6} {h['effective_from'] or '-':10}  {h['article'] or '-':8}  {text}")
    if not hits:
        print("  (결과 없음. 필터 값이 payload 와 정확히 같아야 합니다. data/documents.csv 의 issuer, doc_type 열을 보세요)")
    qdrant.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
