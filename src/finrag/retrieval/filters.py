# -*- coding: utf-8 -*-
"""Self-Query 필터 스키마와 Qdrant Filter 변환.

LLM 은 이 Pydantic 스키마를 채우기만 한다. Qdrant Filter 로 옮기는 일은 여기서 한다.
"필터가 0건이면 완화한다"는 규칙도 여기 있다 — 필터를 잘못 걸어 0건이 나오는 것이
필터를 안 거는 것보다 나쁘기 때문이다.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field
from qdrant_client import models


class QueryFilters(BaseModel):
    """질문에서 뽑아낼 수 있는 조건. 모르면 비워 둔다(추측 금지)."""
    doc_type: str | None = Field(None, description="약관·표준약관·특약·상품설명서·법령·분쟁조정사례 등")
    issuer: str | None = Field(None, description="발행 기관명. 예: 하나은행, 카카오뱅크, KB국민카드")
    product: str | None = Field(None, description="상품명. 예: 정기예금, 주택담보대출, 실손의료보험")
    generation: str | None = Field(None, description="실손보험 세대. 예: 4세대, 5세대")
    effective_on: str | None = Field(None, description="기준일(YYYY-MM-DD) 또는 'latest'")
    exclude_synthetic: bool = Field(True, description="열화 재현본 제외 여부")


def to_qdrant(f: QueryFilters) -> models.Filter | None:
    must: list[models.Condition] = []
    if f.doc_type:
        must.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=f.doc_type)))
    if f.issuer:
        must.append(models.FieldCondition(key="issuer", match=models.MatchValue(value=f.issuer)))
    if f.product:
        must.append(models.FieldCondition(key="product", match=models.MatchValue(value=f.product)))
    if f.generation:
        must.append(models.FieldCondition(key="generation", match=models.MatchValue(value=f.generation)))
    if f.effective_on and f.effective_on != "latest":
        # 시행일이 기준일보다 앞선 판본만. 종료일은 비어 있거나 기준일보다 뒤.
        must.append(models.FieldCondition(key="effective_from",
                                          range=models.DatetimeRange(lte=f.effective_on)))
    must_not: list[models.Condition] = []
    if f.exclude_synthetic:
        must_not.append(models.FieldCondition(key="synthetic_degraded",
                                              match=models.MatchValue(value=True)))
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def relax(f: QueryFilters) -> QueryFilters | None:
    """0건일 때 한 단계씩 푼다. 가장 좁은 조건부터 버린다."""
    for field in ("generation", "product", "effective_on", "issuer", "doc_type"):
        if getattr(f, field):
            d = f.model_dump()
            d[field] = None
            return QueryFilters(**d)
    return None


def pick_latest(hits: list[dict], as_of: str | None = None) -> list[dict]:
    """effective_on='latest' 는 필터가 아니라 정렬 문제다.

    같은 (issuer, product, doc_type) 안에서 기준일 이전의 가장 최근 판본만 남긴다.
    이걸 안 하면 2013년 약관과 2024년 약관이 같이 올라와 답이 섞인다.
    """
    as_of = as_of or date.today().isoformat()
    best: dict[tuple, str] = {}
    for h in hits:
        key = (h.get("issuer", ""), h.get("product", ""), h.get("doc_type", ""))
        eff = h.get("effective_from") or ""
        if eff and eff > as_of:
            continue
        if key not in best or eff > best[key]:
            best[key] = eff
    out = []
    for h in hits:
        key = (h.get("issuer", ""), h.get("product", ""), h.get("doc_type", ""))
        eff = h.get("effective_from") or ""
        if key not in best or eff == best[key]:
            out.append(h)
    return out
