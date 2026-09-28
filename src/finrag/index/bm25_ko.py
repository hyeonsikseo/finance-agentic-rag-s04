# -*- coding: utf-8 -*-
"""한국어 BM25.

공백으로 자르면 "제15조의"와 "제15조"가 다른 토큰이 되어 조항번호 검색이 실패한다.
rank_bm25·fastembed·Qdrant 내장 BM25 모두 한국어 형태소 토크나이저가 없다
(fastembed 지원 언어 18개에 한·중·일이 없다). 그래서 kiwipiepy 로 직접 자른다.

토큰화 규칙 세 가지.
  1. 조항번호는 정규형 하나로 모은다: "제 15 조", "제15조의", "제15조" → "제15조"
  2. 숫자·비율·금액은 원형을 남긴다: "1.4%", "466,666"
  3. 나머지는 명사·동사 어간 등 내용어만 남기고 조사·어미는 버린다
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from ..parsing.metadata import ARTICLE_PAT, expand_query_terms

# 내용어 품사만 남긴다. 조사(J*)·어미(E*)·접미사(X*)는 검색에 방해만 된다.
KEEP_TAGS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "MAG", "SL", "SN", "SH"}
NUMERIC = re.compile(r"\d[\d,.]*%?")
ARTICLE_TOKEN = re.compile(r"제\s*\d+\s*조(?:\s*의\s*\d+)?")


@lru_cache
def _kiwi():
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    # 금융 도메인 낱말은 형태소 분석기가 쪼개 버린다. 통째로 하나의 토큰이 되게 넣는다.
    for w in ["중도해지이율", "중도해지이자율", "중도상환수수료", "중도상환해약금",
              "연체이자율", "지연배상금", "자기부담금", "공제금액", "해약환급금",
              "해지환급금", "일부결제금액이월약정", "전신환매도율", "만기후이율",
              "기한의이익", "보장개시일", "납입면제", "심의필", "예금자보호",
              "약정이자율", "기본이자율", "경과월수", "계약월수", "비과세종합저축"]:
        try:
            kiwi.add_user_word(w, "NNG")
        except Exception:
            pass
    return kiwi


def normalize_articles(text: str) -> str:
    """조항 표기를 하나로 모은다. 토큰화 전에 해야 형태소 분석기가 쪼개지 않는다."""
    def rep(m: re.Match) -> str:
        g = ARTICLE_PAT.search(m.group(0))
        if not g:
            return m.group(0)
        return f" 제{int(g.group(1))}조" + (f"의{int(g.group(2))}" if g.group(2) else "") + " "
    return ARTICLE_TOKEN.sub(rep, text)


def tokenize(text: str, *, expand: bool = False) -> list[str]:
    """글을 BM25 토큰 목록으로 만든다. 문서를 인덱싱할 때와 질의를 검색할 때 둘 다 이 함수를 쓴다.

    expand=True 는 질의에만 쓴다. 별칭 사전(ALIASES)의 낱말을 더해 "자기부담금"으로 물어도
    "공제금액"이라고 적힌 조항이 걸리게 한다. 문서 쪽은 확장하지 않는다.
    """
    # ── TODO: 여기를 채우세요 ──────────────────────────────
    # 요구사항은 tests/test_hybrid.py 의 tokenize 테스트 6개가 정한다. 재료는 이 파일 위에 다 있다.
    #   normalize_articles · ARTICLE_TOKEN · NUMERIC · _kiwi() (사용자 사전) · KEEP_TAGS · expand_query_terms
    # Kiwi 토큰은 .form(글자)과 .tag(품사)를 갖는다. 어떤 재료를 어떤 순서로 쓸지는 직접 정한다.
    # 순서가 틀리면 어느 테스트가 빨개지는지가 힌트다. expand 는 질의에만 쓴다(문서 쪽은 확장하지 않는다).
    raise NotImplementedError("TODO: tokenize 를 구현하세요")


def whitespace_tokenize(text: str) -> list[str]:
    """비교용 기준선. 4회차에서 이것과 Kiwi 의 Recall 차이를 직접 잰다."""
    return text.split()


class BM25Index:
    """rank_bm25 위에 얇게 얹은 인덱스. 토큰화 방식을 갈아 끼울 수 있게 해 둔다."""

    def __init__(self, chunk_ids: list[str], corpus_tokens: list[list[str]]):
        from rank_bm25 import BM25Okapi
        self.chunk_ids = chunk_ids
        self.bm25 = BM25Okapi(corpus_tokens)

    @classmethod
    def build(cls, chunks: list[dict], tokenizer=tokenize) -> "BM25Index":
        ids = [c["chunk_id"] for c in chunks]
        toks = [tokenizer(c["text"]) or ["_"] for c in chunks]
        return cls(ids, toks)

    def search(self, query: str, k: int = 20, tokenizer=tokenize) -> list[tuple[str, float]]:
        q = tokenizer(query, expand=True) if tokenizer is tokenize else tokenizer(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order if scores[i] > 0]
