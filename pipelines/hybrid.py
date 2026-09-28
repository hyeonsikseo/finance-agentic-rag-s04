#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hybrid: Dense + Kiwi BM25 → RRF → Cross-Encoder 리랭킹. 4회차 결과."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finrag.retrieval import hybrid, rerank   # noqa: E402


def build(k: int = 10, *, candidates: int | None = None, use_rerank: bool = True,
          use_bm25: bool = True):
    from finrag.settings import get_settings
    n = candidates or get_settings().rerank_candidates

    def run(question: str) -> dict:
        hits = hybrid.search(question, k=n, candidates=n, use_bm25=use_bm25)
        stats = {}
        if use_rerank:
            hits, stats = rerank.rerank(question, hits, top_k=k)
        else:
            hits = hits[:k]
        return {"chunk_ids": [h["chunk_id"] for h in hits], "hits": hits, "rerank": stats}
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    from finrag.eval import harness
    name = a.name or ("hybrid" if not a.no_rerank else "hybrid_norerank")
    res = harness.run(build(use_rerank=not a.no_rerank, use_bm25=not a.no_bm25), name)
    harness.print_summary(res)
    print("→", harness.save(res))
