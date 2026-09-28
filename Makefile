.PHONY: help setup download models models-rerank embeddings notebook test ingest-sample ingest index baseline bm25-compare hybrid-norerank hybrid rerank-sweep chunk chunk-sample profile
PY := .venv/bin/python
export HF_HOME := ./models
EMB_URL := https://github.com/hyeonsikseo/finance-agentic-rag-s03/releases/download/s03/BAAI_bge-m3.npz
SAMPLE := --only hana_credit_terms_2009 --only fss_deposit_terms_2024_pdf --only knia_3500_scan --only kakao_deposit_terms_2024 --only hanacard_std_2017 --only law_silson_std_hwp

help:
	@echo "make setup           파이썬 3.12 와 패키지 설치 (3회차 폴더가 있으면 2~3분)"
	@echo "make download        문서 다운로드 (3회차 폴더에서 복사했으면 몇 초)"
	@echo "make models          임베딩 모델 BGE-M3 (3회차 폴더에서 models 를 복사했으면 건너뜁니다)"
	@echo "make models-rerank   리랭커 bge-reranker-v2-m3 다운로드 (2.2GB, 10~20분. 수업 전에)"
	@echo "make embeddings      강사가 만든 임베딩 스냅샷 다운로드 (13MB. 3회차 폴더에서 복사했으면 건너뜁니다)"
	@echo "make notebook        Hybrid 검색 노트북 열기 (수업 1~2부)"
	@echo "make test            자동 채점. 2·3회차 35개 + 4회차 검사 (채우기 전에는 실패가 정상)"
	@echo "make bm25-compare    BM25 만으로 40문항: 공백 토크나이저 vs Kiwi (실측 1, 1~2분)"
	@echo "make hybrid-norerank Dense + BM25 를 RRF 로 합쳐 40문항 (리랭커 없이, 30초)"
	@echo "make hybrid          + 리랭커까지 → results/hybrid.json (오늘 결과물, 1~2분)"
	@echo "make rerank-sweep    리랭커 후보 10·20 별 품질과 지연, 문항 10개 (실습 3, 약 6분)"
	@echo "make check-s04       4회차 확인 → results/check_s04.json (이 파일을 제출)"
	@echo "--- 3회차 명령. 그대로 남겨 둡니다 ---"
	@echo "make ingest          문서 51건 → data/chunks/chunks.jsonl + data/ingest_report.json (약 6분)"
	@echo "make index           청크를 임베딩(스냅샷 재사용)해서 Qdrant 에 넣는다 (10초)"
	@echo "make baseline        골든셋 40문항 Dense 검색 → results/baseline.json (30초)"

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt

download:
	$(PY) data/download_corpus.py

models:
	$(PY) scripts/pull_models.py --only embed

models-rerank:
	$(PY) scripts/pull_models.py --only rerank

embeddings:
	mkdir -p data/embeddings
	curl -L --fail -o data/embeddings/BAAI_bge-m3.npz $(EMB_URL)
	@ls -la data/embeddings/BAAI_bge-m3.npz

notebook:
	$(PY) -m jupyter lab notebooks/s04_hybrid.ipynb

test:
	$(PY) -m pytest -q

bm25-compare:
	$(PY) scripts/bm25_compare.py

hybrid-norerank:
	$(PY) pipelines/hybrid.py --no-rerank

hybrid:
	$(PY) pipelines/hybrid.py

rerank-sweep:
	$(PY) scripts/rerank_sweep.py

check-s%:
	$(PY) scripts/check_session.py $*

# ── 3회차 명령. 그대로 남겨 둡니다 ──
ingest-sample:
	$(PY) pipelines/ingest.py --no-checkpoint $(SAMPLE)

ingest:
	$(PY) pipelines/ingest.py

index:
	$(PY) scripts/build_index.py

baseline:
	$(PY) pipelines/baseline.py

# ── 2회차 명령 ──
chunk-sample:
	$(PY) pipelines/chunk_only.py --verbose $(SAMPLE)

chunk:
	$(PY) pipelines/chunk_only.py --verbose

profile:
	$(PY) scripts/profile_docs.py
