"""실행 진입점.

    python app.py           # 전체 실행: 후보 발굴 → 평가 반복 → 보고서(PDF) 생성
    python app.py --stub    # 가짜 노드로 그래프 흐름만 검증 (API 키 불필요)

벡터스토어가 없으면(처음 clone한 경우) 문서 인덱스부터 만든 뒤 실행한다.
"""
import argparse
import json
import sys
from datetime import date

from config import OUTPUT_DIR, RECURSION_LIMIT, VECTORSTORE_DIR, missing_env
from graph.builder import build_graph


def prepare() -> None:
    """실제 실행 전 점검: 필수 환경변수, 벡터스토어."""
    missing = missing_env()
    if missing:
        sys.exit(f".env에 값이 없습니다: {', '.join(missing)} (.env.example 참고)")
    if not (VECTORSTORE_DIR / "chroma.sqlite3").exists():
        print("벡터스토어가 없어 문서 인덱스를 먼저 만듭니다 (처음 한 번, 수 분 소요)")
        from rag.ingest import ingest
        ingest()


def initial_state() -> dict:
    return {
        "domain": "Energy",
        "run_date": date.today().isoformat(),
        "candidates": [],
        "search_round": 0,
        "evaluated": [],
        "market_cache": {},
        "history": [],
        "references": [],
        "log": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true", help="가짜 노드로 흐름만 검증")
    args = parser.parse_args()

    if not args.stub:
        prepare()
    graph = build_graph(stub=args.stub)
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")

    final = graph.invoke(initial_state(), config={"recursion_limit": RECURSION_LIMIT})

    print("\n".join(final["log"]))
    print(f"\n보고서: {final.get('report_path')}")
    run_log = {k: final.get(k) for k in ("run_date", "history", "decision", "total_score", "log")}
    (OUTPUT_DIR / "run_log.json").write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
