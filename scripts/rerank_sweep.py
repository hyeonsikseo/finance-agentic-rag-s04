#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""리랭커에 넣는 후보 수를 바꿔 가며 품질(MRR · Recall@1)과 지연(ms)을 잰다. 실습 3.

    python scripts/rerank_sweep.py                       # 후보 10 · 20, 문항 10개 (수업용, 약 6분)
    python scripts/rerank_sweep.py --candidates 10 20 30 50 --limit 0   # 후보 50까지, 40문항 전부 (40분 넘게 걸린다)
    python scripts/rerank_sweep.py --candidates 10 20 30                 # 후보 30 도 (약 11분)

후보가 많을수록 정답이 후보 안에 들어올 확률은 오르지만 리랭커가 볼 문서도 늘어난다.
지연은 후보 수에 비례한다. 어디서 멈출지를 이 표를 보고 정하고, 이유를 ADR-003 에 적는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--limit", type=int, default=10, help="문항 수 (기본 10. 0 이면 40문항 전부)")
    args = ap.parse_args()
    if args.limit == 0:
        args.limit = None

    from finrag.eval import harness
    from finrag.retrieval import hybrid, rerank

    if rerank.get_reranker() is None:
        print("리랭커 모델이 없습니다. make models-rerank 를 먼저 하세요.")
        return 1

    table = []
    for n in args.candidates:
        stats: list[dict] = []

        def pipeline(q: str, n=n, stats=stats) -> dict:
            hits = hybrid.search(q, k=n, candidates=n)
            hits, st = rerank.rerank(q, hits, top_k=10)
            stats.append(st)
            return {"chunk_ids": [h["chunk_id"] for h in hits]}

        res = harness.run(pipeline, f"hybrid_c{n}", limit=args.limit)
        harness.save(res)
        o = res["overall"]
        ms = [s["ms"] for s in stats if s.get("reranked")]
        row = {"candidates": n, "recall@1": o["recall@1"], "recall@5": o["recall@5"], "mrr": o["mrr"],
               "latency_ms_avg": o["latency_ms_avg"],
               "rerank_ms_avg": round(sum(ms) / max(len(ms), 1), 1),
               "ms_per_candidate": round(sum(ms) / max(len(ms), 1) / n, 2)}
        table.append(row)
        print(f"  후보 {n:>3}: R@1 {o['recall@1']:.1%}  R@5 {o['recall@5']:.1%}  MRR {o['mrr']:.3f}  "
              f"문항당 {o['latency_ms_avg']:.0f}ms (리랭커 {row['rerank_ms_avg']}ms, 후보당 {row['ms_per_candidate']}ms)")

    out = ROOT / "results" / "rerank_sweep.json"
    out.write_text(json.dumps({"n_questions": args.limit or 40, "rows": table}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n문항 {args.limit or 40}개 기준입니다. 절대값이 아니라 후보 수에 따른 기울기를 봅니다.")
    print(f"\n{'후보':>4} {'R@1':>7} {'R@5':>7} {'MRR':>7} {'문항당 ms':>9} {'리랭커 ms':>9} {'후보당 ms':>9}")
    for r in table:
        print(f"{r['candidates']:>4} {r['recall@1']:>7.1%} {r['recall@5']:>7.1%} {r['mrr']:>7.3f} "
              f"{r['latency_ms_avg']:>9.0f} {r['rerank_ms_avg']:>9.0f} {r['ms_per_candidate']:>9.2f}")
    print(f"→ {out.relative_to(ROOT)}\n고른 후보 수와 이유를 ADR-003 에 적습니다. 기본값은 .env 의 RERANK_CANDIDATES 로 바꿉니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
