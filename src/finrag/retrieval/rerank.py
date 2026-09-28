# -*- coding: utf-8 -*-
"""Cross-Encoder 리랭커.

Bi-encoder(임베딩)는 질문과 문서를 따로 벡터로 만들어 비교한다. 빠르지만 둘 사이의
상호작용을 못 본다. Cross-Encoder 는 질문과 문서를 함께 넣어 점수를 낸다. 느려서
전체에는 못 쓰고, 앞 단계가 추린 20~50개에만 쓴다.

모델이 없으면 조용히 건너뛴다. 리랭커 때문에 파이프라인 전체가 죽으면 안 된다.
"""
from __future__ import annotations

import time
from functools import lru_cache

from ..settings import get_settings


@lru_cache
def get_reranker():
    import os
    s = get_settings()
    os.environ.setdefault("HF_HOME", s.hf_home)
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(s.reranker_model, max_length=512)
    except Exception:
        # 리랭커가 없다고 파이프라인 전체가 죽으면 안 된다. 없으면 없는 대로 간다.
        return None


def rerank(query: str, hits: list[dict], top_k: int | None = None) -> tuple[list[dict], dict]:
    """(재정렬된 결과, 측정치). 측정치는 4회차에서 지연 비용을 보여주는 데 쓴다."""
    s = get_settings()
    top_k = top_k or s.rerank_top_k
    model = get_reranker()
    if model is None or not hits:
        return hits[:top_k], {"reranked": False, "candidates": len(hits), "ms": 0.0}

    t0 = time.perf_counter()
    scores = model.predict([(query, h["text"][:2000]) for h in hits],
                           show_progress_bar=False)
    ms = (time.perf_counter() - t0) * 1000
    for h, sc in zip(hits, scores):
        h["rerank_score"] = float(sc)
    out = sorted(hits, key=lambda h: -h["rerank_score"])[:top_k]
    return out, {"reranked": True, "candidates": len(hits), "ms": round(ms, 1),
                 "ms_per_candidate": round(ms / max(len(hits), 1), 2)}
