# 약관부터 답변까지 — 4회차

금융 문서 특화 Agentic RAG 수업의 4회차 레포지토리입니다. 3회차 레포지토리(finance-agentic-rag-s03)와는 별개이고 새로 받습니다. 1~3회차 내용은 그대로 들어 있고, 3회차에 채운 함수 두 개(`gate`, `build_ingest_graph`)는 **강사 정답으로 채워져** 있습니다. 자기 코드로 바꿔 써도 되지만, 그러면 청크 수와 검색 성능이 강사 값과 조금 달라질 수 있습니다.

## 4회차에 할 일

1. 설치합니다. 3회차 폴더에서 문서·청크·모델을 복사하고, 리랭커 모델 2.2GB 를 **수업 전에** 받아 둡니다. → [docs/install.md](docs/install.md)
2. 수업 앞부분은 노트북으로 3회차 baseline 이 놓친 문항과 한국어 토크나이저를 봅니다. → `make notebook`
3. 수업 중반은 채울 자리 셋(`tokenize`, `rrf`, `search` 의 합치기)을 채우고, BM25 만으로 공백 토크나이저와 Kiwi 를 비교 측정합니다. → [docs/session4/lab.md](docs/session4/lab.md)
4. 수업 뒷부분은 리랭커까지 붙여 40문항을 다시 측정하고(`results/hybrid.json`), 후보 수를 바꿔 가며 품질과 지연을 잽니다.
5. 과제는 필수 둘(전체 실행, ADR-003)에 준비 하나(OpenAI 키)와 선택 하나이고, `make check-s04` 가 6/6 이면 끝입니다. → [docs/session4/homework.md](docs/session4/homework.md)

## 폴더

| 폴더 | 무엇 |
|---|---|
| `docs/install.md` | 설치. 3회차 폴더에서 복사하는 단계와 리랭커 받는 단계가 새로 있습니다 |
| `docs/session4/` | 실습 순서(lab.md)와 과제(homework.md) |
| `docs/templates/adr.md` | 과제 ADR-003 에 쓰는 템플릿. 완성 예시는 `docs/session3/examples/01_ADR-002_인제스천_그래프.md` |
| `notebooks/s04_hybrid.ipynb` | Hybrid 검색 20분. 수업 1~2부. Qdrant 를 열지 않습니다 |
| `src/finrag/index/bm25_ko.py` | 한국어 BM25. 채울 함수 `tokenize`(실습 1)가 여기 있습니다 |
| `src/finrag/retrieval/hybrid.py` | Dense + BM25 → RRF. 채울 것 `rrf` 와 `search` 의 합치기(실습 2)가 여기 있습니다 |
| `src/finrag/retrieval/rerank.py`, `filters.py` | Cross-Encoder 리랭커, 필터와 최신 판본 고르기. 읽기만 합니다 |
| `pipelines/hybrid.py` | 골든셋 40문항을 Hybrid + 리랭커로 풀어 `results/hybrid.json` 을 만듭니다 |
| `scripts/bm25_compare.py`, `rerank_sweep.py`, `search_demo.py` | 실측 1, 실습 3, 질문 하나로 검색 비교 |
| `tests/test_hybrid.py` | 채운 것이 맞는지 보는 테스트 16개. 2·3회차 테스트도 그대로 돕니다 |
| `scripts/check_session.py` | 4회차 확인. 결과 파일이 제출물입니다 |

## 명령

```bash
make setup           # 파이썬 3.12 와 패키지 설치. 3회차 폴더가 있으면 2~3분
make models-rerank   # 리랭커 bge-reranker-v2-m3 받기. 2.2GB, 10~20분. 수업 전에
make index           # 3회차 청크를 Qdrant 에 넣는다 (스냅샷 재사용). 10초
make baseline        # 3회차 baseline 을 이 폴더에서 다시 만든다. 30초
make notebook        # Hybrid 검색 노트북 열기
make test            # 자동 채점. 2·3회차 35개 + 4회차 16개. 채우기 전에는 14개 실패가 정상입니다
make bm25-compare    # BM25 만으로 40문항: 공백 vs Kiwi. 처음 한 번은 토큰 캐시를 만드느라 약 1분 걸립니다
make hybrid-norerank # Dense + BM25 → RRF 로 40문항. 리랭커 없이. 30초
make hybrid          # + 리랭커 → results/hybrid.json (오늘 결과물). 약 8~10분
make rerank-sweep    # 리랭커 후보 10·20 별 품질과 지연, 문항 10개. 약 6분
make check-s04       # 결과물이 조건 6개를 만족하는지 확인 → results/check_s04.json
```

Windows 에서는 `make` 가 없으니 [docs/install.md](docs/install.md) 의 PowerShell 명령을 씁니다.

## 원본 문서가 레포지토리에 없는 이유

약관과 상품설명서는 각 금융회사의 저작물이라 다시 나눠 줄 수 없습니다. 레포지토리에는 "무엇을 어디서 받는지"만 있고, 각자 원 출처에서 받거나 3회차 폴더에서 복사합니다. 자세한 것은 [data/README.md](data/README.md) 에 있습니다. 같은 이유로 문서 본문이 들어 있는 `data/chunks/`, `data/ocr_cache/`, 그리고 청크를 토큰으로 바꿔 둔 `data/bm25_cache/` 도 레포지토리에 올리지 않습니다.
