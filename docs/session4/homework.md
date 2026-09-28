# 4회차 과제

필수 둘, 준비 하나, 선택 하나입니다. 다음 회차 전까지 냅니다. 끝났는지는 `make check-s04` 가 **6/6** 인지로 압니다.

## 필수 1. 전체 실행 — 수업에서 끝냈으면 0분, 아니면 10분

채울 자리 셋(`bm25_ko.tokenize`, `hybrid.rrf`, `hybrid.search` 의 합치기)이 채워졌으면 이렇게 끝납니다.

```bash
make test        # 50개 통과, 1개 건너뜀. make hybrid 뒤에는 51개 전부
make hybrid      # 40문항 → results/hybrid.json. 리랭커까지 약 8분 (GPU 가속이 없는 노트북은 더)
make check-s04   # 5/6. 마지막 항목은 필수 2 입니다
```

Windows 는 [install.md](../install.md) 의 PowerShell 명령을 씁니다. 수업에서 못 채운 함수가 있으면 강사가 수업 뒤 올리는 **solution 브랜치**를 봅니다.

https://github.com/hyeonsikseo/finance-agentic-rag-s04/tree/solution/src/finrag

보고 베끼는 게 아니라 **읽고 닫은 다음 자기 손으로 다시 씁니다.** 다음 회차의 에이전트가 이 검색 위에 얹히므로, 지금 이해하고 넘어가야 합니다.

`results/hybrid.json` 의 `overall.mrr` 과 `overall.recall@1` 을 적어 두세요. 강사 값은 MRR 0.652(baseline 0.532), Recall@1 55.0%(baseline 40.0%)입니다. 리랭커 없이 돌렸으면 MRR 이 baseline 보다 낮게 나오고 `check-s04` 의 MRR 항목 두 개가 X 입니다. 리랭커를 받은 뒤 다시 돌리면 됩니다.

## 필수 2. ADR-003 — 25분

오늘 만든 Hybrid 검색에 대해 **결정 기록(ADR)** 을 씁니다. 3회차의 ADR-002 와 같은 방법입니다.

1. `docs/templates/adr.md` 를 `docs/adr/ADR-003-hybrid-retrieval.md` 로 복사합니다.
2. 빈칸(`______`)을 전부 채웁니다. `make check-s04` 가 빈칸이 남았는지 셉니다.

제목은 "Kiwi 형태소 BM25 + RRF + Cross-Encoder 리랭킹" 으로 잡으면 됩니다. 채울 때 볼 것:

- **맥락에 검색 실패 사례 3건을 넣습니다.** 노트북 셀 2 의 "baseline 이 놓친 문항" 중 3개를 골라, 질문에 무엇이 들어 있어서(조항번호·상품명·회사명·숫자) Dense 가 놓쳤는지, Hybrid 에서는 어떻게 됐는지(`results/hybrid.json` 의 같은 문항 `mrr`)를 한 줄씩 적습니다.
- **근거의 숫자**는 세 곳에서 가져옵니다. `make bm25-compare` 표(공백 vs Kiwi), `results/hybrid_norerank.json` 과 `results/hybrid.json`(RRF 만 vs 리랭커까지), `make rerank-sweep` 표(후보 수별 품질과 지연).
- **대안**은 최소 둘. "점수를 정규화해 더한다(RRF 대신)"와 "리랭커 없이 RRF 만"을 넣고, 왜 안 골랐는지를 숫자로 적습니다.
- **되돌릴 조건**을 비워 두지 않습니다. 어떤 일이 생기면 리랭커를 빼거나 후보 수를 바꿀지 한 줄이면 됩니다.
- 길이는 한 쪽. 예시 형식은 [ADR-002 완성본](../session3/examples/01_ADR-002_인제스천_그래프.md) 을 봅니다.

## 준비. OpenAI 키 — 5분, 다음 회차 전까지

다음 회차부터 LLM 을 실제로 부릅니다. 질문에서 필터를 추출하고, 근거가 충분한지 판정하고, 답변을 만듭니다. 키가 없으면 그 실습이 가짜 모델로 돌아서 결과를 볼 수 없습니다.

키는 **메타코드에서 발급해 드립니다.** 직접 결제하거나 계정을 만들 것은 없습니다. 받으면 둘만 합니다.

1. 레포지토리 폴더에서 `.env.example` 을 `.env` 로 복사하고 `OPENAI_API_KEY=` 뒤에 붙입니다. `.env` 는 커밋되지 않습니다. 키를 채팅이나 코드에 붙이지 마세요.
2. 되는지 확인합니다. 한 단어 답이 나오면 된 것이고, "키가 없어 가짜 모델입니다"가 나오면 1번을 다시 봅니다.

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from finrag.llm import get_llm, is_fake; m=get_llm('answer'); print('키가 없어 가짜 모델입니다' if is_fake(m) else m.invoke('한 단어로 답하세요: 안녕하세요').content)"
```

## 선택. Self-Query 대상 질문 5개 — 10분

다음 회차에는 LLM 이 질문에서 필터(발행사 · 문서 종류 · 시행일)를 추출합니다. 오늘 `search_demo --latest` 에서 본 것처럼 판본이 섞이는 질문이 그 대상입니다. 자기 코퍼스에서 "시행일이나 문서 종류를 조건으로 걸어야 답이 하나로 정해지는 질문" 5개를 `docs/selfquery_questions.md` 에 적어 둡니다. 한 줄에 질문 / 걸어야 할 조건 / 그 조건이 없으면 무엇이 섞이는지.

## 제출

1. `make check-s04` 가 6/6 인지 봅니다.
2. 커밋하고 자기 포크에 올립니다.

```bash
git add src/finrag/index/bm25_ko.py src/finrag/retrieval/hybrid.py results/hybrid.json results/check_s04.json docs/adr/ADR-003-hybrid-retrieval.md
git commit -m "4회차 과제"
git push
```

3. 자기 포크 URL(`https://github.com/<계정>/finance-agentic-rag-s04`)를 디스코드에 올립니다. 포크 없이 받았으면 위 파일들을 zip 으로 묶어 올립니다.

## check-s04 가 보는 것

| 항목 | 통과 기준 |
|---|---|
| hybrid.json 존재 | `results/hybrid.json` 에 40문항이 있다 |
| 조항번호형 MRR ≥ baseline | 조항번호 질문의 정답 순위가 baseline 보다 앞이다 |
| 전체 MRR ≥ baseline | 전체 정답 순위가 baseline 보다 앞이다 |
| 전체 Recall@1 ≥ baseline | 1등이 정답인 문항이 baseline 보다 많다 |
| 리랭커 지연 기록 | 문항마다 걸린 시간(`latency_ms`)이 적혀 있다 |
| ADR-003 | `docs/adr/ADR-003-hybrid-retrieval.md` 가 있고 빈칸(`______`)이 없다 |

## 마감 뒤 강사가 올리는 것

`docs/session4/examples/` 에 ADR-003 완성본을 올립니다. 받는 법은 같습니다.

```bash
git pull upstream main
```
