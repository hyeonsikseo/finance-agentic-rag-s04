# 설치와 준비물 받기 (4회차)

3회차와 같은 컴퓨터에서, 3회차 폴더(`finance-agentic-rag-s03`) 옆에 새 레포지토리를 받습니다. 새로 받는 것은 **리랭커 모델 2.2GB** 하나이고, 나머지는 3회차 폴더에서 복사합니다. **5번이 10~20분 걸립니다.** 수업 전날까지 끝내 두고, 마지막 8번의 확인 줄을 디스코드에 남겨 주세요.

## 1. 준비물 두 가지, git 과 uv

1~3회차에 했으면 건너뜁니다. 확인은 `git --version`, `uv --version`.

## 2. 레포지토리 받기

지난 회차처럼 포크를 먼저 합니다.

1. 브라우저에서 https://github.com/hyeonsikseo/finance-agentic-rag-s04 를 열고 오른쪽 위 **Fork** 를 누릅니다.
2. 그 레포지토리를 받습니다. `<계정>` 자리에 자기 GitHub 계정을 넣습니다.

```bash
git clone https://github.com/<계정>/finance-agentic-rag-s04.git
cd finance-agentic-rag-s04
git remote add upstream https://github.com/hyeonsikseo/finance-agentic-rag-s04.git
```

포크가 막히면 원본을 그대로 받고 과제는 zip 으로 냅니다.

## 3. 3회차 폴더에서 가져오기

문서, OCR 캐시, 청크, 임베딩 스냅샷, 그리고 임베딩 모델(2.2GB)을 복사합니다. 모델을 복사하지 않으면 `make models` 로 다시 받아야 합니다. 폴더가 다른 곳에 있으면 경로만 바꿉니다.

**Mac**

```bash
cp -R ../finance-agentic-rag-s03/data/raw data/
cp -R ../finance-agentic-rag-s03/data/ocr_cache data/
cp -R ../finance-agentic-rag-s03/data/chunks data/
cp ../finance-agentic-rag-s03/data/ingest_report.json data/
cp -R ../finance-agentic-rag-s03/data/embeddings data/
cp -R ../finance-agentic-rag-s03/models .
```

**Windows (PowerShell)**

```powershell
Copy-Item -Recurse ..\finance-agentic-rag-s03\data\raw data\
Copy-Item -Recurse ..\finance-agentic-rag-s03\data\ocr_cache data\
Copy-Item -Recurse ..\finance-agentic-rag-s03\data\chunks data\
Copy-Item ..\finance-agentic-rag-s03\data\ingest_report.json data\
Copy-Item -Recurse ..\finance-agentic-rag-s03\data\embeddings data\
Copy-Item -Recurse ..\finance-agentic-rag-s03\models .
```

`models` 폴더는 4GB 가 넘어서 복사에 1~2분 걸립니다. 디스크가 빠듯하면 복사 대신 옮겨도 됩니다(`mv`, `Move-Item`). 3회차 폴더는 이제 쓰지 않습니다. Qdrant 인덱스(`data/qdrant_local`)는 복사하지 않고 6번에서 다시 만듭니다.

3회차 폴더가 없거나 `make ingest` 를 끝내지 못했으면, 문서는 `make download`, 모델은 `make models`, 스냅샷은 `make embeddings` 로 받고, 6번에서 `make ingest` 부터 합니다.

## 4. 파이썬과 패키지 설치

**Mac**

```bash
make setup
```

**Windows (PowerShell)**

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

3회차와 같은 패키지에 한국어 형태소 분석기(kiwipiepy)와 BM25(rank-bm25)가 더해집니다. 3회차 때 받아 둔 패키지는 uv 가 다시 받지 않아서 2~3분이면 끝납니다.

## 5. 리랭커 모델 받기 — 2.2GB, 10~20분

검색 결과 후보를 질문과 함께 다시 읽고 순서를 바로잡는 모델(bge-reranker-v2-m3)입니다. 임베딩 모델과 같은 `models` 폴더에 받고, 받은 뒤 실제로 한 번 돌려 봅니다.

**Mac**

```bash
make models-rerank
```

**Windows (PowerShell)**

```powershell
$env:HF_HOME = ".\models"
.venv\Scripts\python scripts\pull_models.py --only rerank
```

마지막에 `검증: O 관련 조항 … > 무관 조항 …` 처럼 O 가 나오면 된 것입니다. 관련 조항의 점수가 무관 조항보다 높다는 뜻입니다. 회사 네트워크에서 멈추면 휴대폰 핫스팟으로 바꿔 다시 돌립니다. 이미 받은 부분은 이어서 받습니다.

## 6. 인덱스와 baseline 다시 만들기

3회차 결과를 이 폴더에서 다시 만듭니다. 오늘 결과(`hybrid.json`)를 baseline 과 같은 코드로 비교하기 위해서입니다.

**Mac**

```bash
make index       # 10초. 청크를 Qdrant 에 넣는다 (스냅샷 재사용)
make baseline    # 30초. results/baseline.json
```

**Windows (PowerShell)**

```powershell
$env:HF_HOME = ".\models"
.venv\Scripts\python scripts\build_index.py
.venv\Scripts\python pipelines\baseline.py
```

3번에서 청크를 복사하지 못했으면 그 전에 `make ingest`(Windows: `.venv\Scripts\python pipelines\ingest.py`, 약 6분)를 먼저 합니다.

## 7. 문서 다운로드

3번에서 복사했으면 건너뜁니다. 복사하지 않았으면 **Mac**: `make download`, **Windows**: `.venv\Scripts\python data\download_corpus.py` (약 5분).

## 8. 되는지 확인

**Mac**

```bash
make test
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\python -m pytest -q
```

지금은 **실패가 정상**입니다. 채울 자리가 비어 있어서 4회차 테스트 14개가 `NotImplementedError` 로 실패합니다. 마지막 줄이 `14 failed, 36 passed, 1 skipped` (문서와 청크를 아직 안 가져왔으면 `13 failed, 29 passed, 9 skipped`) 이고 `error` 가 없으면 설치는 된 것입니다. **이 마지막 줄과 5번의 `검증:` 줄을 디스코드에 적어 주세요.**

## 막힐 때

| 증상 | 이유 | 이렇게 합니다 |
|---|---|---|
| `uv: command not found` | 설치 뒤 터미널을 새로 안 열었습니다 | 터미널을 닫고 새로 엽니다 |
| kiwipiepy 설치가 실패한다 | 파이썬 3.12 가 아닌 가상환경입니다 | `.venv` 를 지우고 4번을 다시 합니다 |
| 리랭커를 받다가 끊겼다 | 네트워크 | 같은 명령을 다시 돌립니다. 받은 부분은 이어서 받습니다 |
| `make hybrid` 가 "리랭커 모델이 없어 건너뜀" 이라고 한다 | 5번을 안 했거나 다른 폴더에 받았습니다 | 레포지토리 폴더에서 5번을 다시 합니다. 리랭커 없이 돌린 결과는 baseline 보다 낮아서 `check-s04` 의 MRR 항목 두 개가 X 가 됩니다 |
| 메모리 8GB 노트북에서 `make hybrid` 가 느리거나 멈춘다 | 임베딩 모델과 리랭커가 각각 2GB 넘게 메모리를 씁니다 | `.env.example` 을 `.env` 로 복사해 `RERANKER_MODEL=BAAI/bge-reranker-base` 줄의 `#` 을 지우고 `make models-rerank` 를 다시 합니다. 작은 리랭커(1.1GB)를 받아 씁니다. 숫자는 강사와 조금 다르게 나옵니다 |
| `make index` 가 25분째 돌고 있다 | 임베딩 스냅샷이 없거나 이름이 다릅니다 | 멈추고(Ctrl+C) `data/embeddings/BAAI_bge-m3.npz` 가 있는지 봅니다. 없으면 `make embeddings` |
| Qdrant `already accessed` 에러 | 다른 터미널이 Qdrant 를 잡고 있습니다 | 그쪽을 닫고 다시 돌립니다. 오늘 노트북은 Qdrant 를 열지 않습니다 |
| Windows 에서 `make` 가 없다 | 원래 없습니다 | 위의 PowerShell 명령을 씁니다 |
| `make test` 에 `error` 가 있다 | 패키지가 덜 깔렸습니다 | 4번을 다시 합니다. 그래도 그러면 에러 줄을 채팅에 붙입니다 |

## 오늘 못 받았으면

노트북, 실습 1·2, `make test`, `make bm25-compare`, `make hybrid-norerank` 는 리랭커 없이 됩니다. 리랭커가 필요한 것은 `make hybrid` 와 `make rerank-sweep` 뿐이니, 그 둘은 강사 화면으로 따라오고 수업 뒤에 받아서 돌립니다.
