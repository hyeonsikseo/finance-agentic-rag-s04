# -*- coding: utf-8 -*-
"""4회차 자기 확인.

    make test        (Windows: .venv\\Scripts\\python -m pytest -q)

네 묶음입니다. 앞의 셋(tokenize, rrf, search 의 합치기)은 문서와 Qdrant 없이 돌고, 마지막
묶음은 3회차에 만든 청크 파일과 `make hybrid` 결과가 있어야 돕니다(없으면 건너뜁니다).
채우기 전에는 앞의 세 묶음이 NotImplementedError 로 실패합니다. 그게 정상입니다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ── 실습 1 · bm25_ko.tokenize ──────────────────────────────────────────
def test_tokenize_merges_article_number_into_one_token():
    from finrag.index.bm25_ko import tokenize
    toks = tokenize("제 15 조의 2에 따른 중도해지이율은 연 1.4%입니다")
    assert "제15조의2" in toks, "조항 표기는 '제15조의2' 한 토큰으로 모여야 합니다 (normalize_articles 를 먼저)"
    assert "제" not in toks and "조의" not in toks, "조항이 '제', '15', '조의' 로 쪼개지면 안 됩니다"


def test_tokenize_keeps_domain_word_whole():
    from finrag.index.bm25_ko import tokenize
    toks = tokenize("제 15 조의 2에 따른 중도해지이율은 연 1.4%입니다")
    assert "중도해지이율" in toks, "사용자 사전에 넣은 낱말은 통째로 한 토큰입니다"


def test_tokenize_keeps_numbers_ratios_and_amounts():
    from finrag.index.bm25_ko import tokenize
    toks = tokenize("연 1.4%입니다. 해약금은 466,666원입니다")
    assert "1.4%" in toks, "비율은 원형 그대로 남겨야 합니다 (NUMERIC)"
    assert "466,666" in toks, "금액은 쉼표까지 원형 그대로 남겨야 합니다"


def test_tokenize_drops_particles_and_endings():
    from finrag.index.bm25_ko import tokenize
    toks = tokenize("카카오뱅크의 정기예금은 만기 후 이율이 어떻게 되나요")
    assert "이율" in toks and "만기" in toks
    for bad in ("의", "은", "이", "되나요", "입니다"):
        assert bad not in toks, f"조사·어미('{bad}')는 빠져야 합니다 (KEEP_TAGS 만 남깁니다)"


def test_tokenize_empty_text_returns_empty_list():
    from finrag.index.bm25_ko import tokenize
    assert tokenize("") == []


def test_tokenize_expand_adds_aliases_only_when_asked():
    from finrag.index.bm25_ko import tokenize
    assert "일부결제금액이월약정" in tokenize("리볼빙 수수료율", expand=True), \
        "expand=True 면 별칭 사전(ALIASES)의 낱말이 더해집니다"
    assert "일부결제금액이월약정" not in tokenize("리볼빙 수수료율"), \
        "expand 가 False 면 더하지 않습니다. 문서 쪽은 확장하지 않습니다"


def test_whitespace_tokenizer_is_the_control():
    from finrag.index.bm25_ko import whitespace_tokenize
    # 비교용 기준선. 조항 하나가 네 토막으로 쪼개진다. 오늘 실측에서 이것과 Kiwi 를 비교합니다.
    assert whitespace_tokenize("제 15 조의 2") == ["제", "15", "조의", "2"]


# ── 실습 2 · hybrid.rrf ────────────────────────────────────────────────
def test_rrf_scores_are_reciprocal_ranks():
    from finrag.retrieval.hybrid import rrf
    fused = rrf([["a", "b", "c"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61), "순위 1 은 1/(60+1)"
    assert fused["b"] == pytest.approx(1 / 62) and fused["c"] == pytest.approx(1 / 63)


def test_rrf_adds_up_across_rankings():
    from finrag.retrieval.hybrid import rrf
    fused = rrf([["a", "b"], ["b", "a"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62), "목록마다 나온 순위의 몫을 더합니다"
    assert fused["a"] == pytest.approx(fused["b"])


def test_rrf_document_in_both_lists_beats_one_list():
    from finrag.retrieval.hybrid import rrf
    fused = rrf([["a", "b"], ["a", "c"]])
    assert fused["a"] > fused["b"] and fused["a"] > fused["c"], "두 목록에 다 나온 문서가 앞입니다"


def test_rrf_uses_rank_not_list_length():
    from finrag.retrieval.hybrid import rrf
    assert rrf([["a"]])["a"] == pytest.approx(rrf([["a", "b", "c", "d"]])["a"]), \
        "같은 순위면 목록 길이와 상관없이 같은 점수입니다. 점수가 아니라 순위만 씁니다"


def test_rrf_k_controls_how_much_rank_one_dominates():
    from finrag.retrieval.hybrid import rrf
    fused = rrf([["a", "b"]], k=0)
    assert fused["a"] == pytest.approx(1.0) and fused["b"] == pytest.approx(0.5)


# ── 실습 2 · hybrid.search 의 합치기 (Qdrant·청크 없이 가짜 재료로 돕니다) ──────
def _fake_search_env(monkeypatch, dense: list[str], bm25: list[str]):
    """dense.search, _bm25, _by_id 를 가짜로 바꾼다. 청크 a1·a2 는 문서 A, b1 은 B, c1 은 C."""
    from finrag.retrieval import hybrid
    docs = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}
    monkeypatch.setattr(hybrid.dense, "search",
                        lambda query, k, flt=None, collection=None:
                        [{"chunk_id": c, "doc_id": docs[c], "score": 0.9} for c in dense])

    class FakeBM25:
        def search(self, query, k=20, tokenizer=None):
            return [(c, 10.0) for c in bm25]

    monkeypatch.setattr(hybrid, "_bm25", lambda: FakeBM25())
    monkeypatch.setattr(hybrid, "_by_id", lambda: {c: {"chunk_id": c, "doc_id": d, "text": c} for c, d in docs.items()})
    return hybrid


def test_search_fuses_dense_and_bm25_by_rank(monkeypatch):
    hybrid = _fake_search_env(monkeypatch, dense=["a1", "b1", "c1"], bm25=["c1", "a1"])
    out = hybrid.search("질문", k=3)
    ids = [h["chunk_id"] for h in out]
    assert ids[0] == "a1", "두 목록에 다 나온 문서가 1등입니다 (Dense 1위 + BM25 2위)"
    assert ids[1] == "c1", "Dense 3위 + BM25 1위가 2등. 한 목록에만 있는 b1 은 그 뒤입니다"
    assert set(out[0]) >= {"chunk_id", "score", "doc_id", "text"}, "_hydrate 가 만든 dict 를 돌려줍니다"
    assert out[0]["score"] == pytest.approx(1 / 61 + 1 / 62)


def test_search_restricts_bm25_to_filtered_docs(monkeypatch):
    from qdrant_client import models
    # 발행사 필터로 Dense 는 문서 A 의 청크만 돌려줬다. BM25 는 필터를 모르고 문서 B 의 청크를 올렸다.
    hybrid = _fake_search_env(monkeypatch, dense=["a1", "a2"], bm25=["b1", "a2"])
    flt = models.Filter(must=[models.FieldCondition(key="issuer", match=models.MatchValue(value="A은행"))])
    ids = [h["chunk_id"] for h in hybrid.search("질문", k=5, flt=flt)]
    assert "b1" not in ids, ("필터가 있으면 BM25 결과를 Dense 가 걸러 낸 문서(doc_id) 안으로 제한해야 합니다. "
                             "BM25 인덱스는 Qdrant 밖이라 필터를 모릅니다")
    assert ids[0] == "a2", "필터 안의 문서는 두 목록의 몫이 더해져 앞으로 옵니다"
    assert "b1" in [h["chunk_id"] for h in hybrid.search("질문", k=5)], "필터가 없으면 제한하지 않습니다"


# ── 코퍼스가 있어야 도는 것 (3회차 make ingest 결과) ─────────────────────
def _chunks_of(doc_id: str) -> list[dict]:
    p = ROOT / "data" / "chunks" / "chunks.jsonl"
    if not p.exists():
        pytest.skip("청크 없음 (3회차 make ingest, 또는 install.md 의 복사 단계)")
    return [c for c in (json.loads(l) for l in p.open(encoding="utf-8") if l.strip())
            if c["doc_id"] == doc_id]


def test_kiwi_bm25_finds_the_article_by_its_number():
    from finrag.index.bm25_ko import BM25Index, tokenize
    chunks = _chunks_of("hanacard_std_2017")
    assert chunks, "하나카드 표준약관 청크가 없습니다"
    idx = BM25Index.build(chunks, tokenizer=tokenize)
    top = idx.search("제31조 리볼빙", k=3, tokenizer=tokenize)
    assert top, "검색 결과가 비었습니다"
    by_id = {c["chunk_id"]: c for c in chunks}
    assert by_id[top[0][0]]["article"] == "제31조", "조항번호로 물으면 그 조항이 1등이어야 합니다"


# ── make hybrid 뒤에 도는 것 ───────────────────────────────────────────
def test_hybrid_report_has_all_questions_and_latency():
    p = ROOT / "results" / "hybrid.json"
    if not p.exists():
        pytest.skip("결과 없음 (make hybrid)")
    rep = json.loads(p.read_text(encoding="utf-8"))
    assert rep["n"] == 40, "골든셋 40문항 전부를 돌려야 합니다"
    assert all(r.get("latency_ms") for r in rep["rows"]), "문항마다 지연 시간이 기록되어야 합니다"
