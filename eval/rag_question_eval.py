"""Agentic RAG 질문 방식 비교: 팀 원문 질문 vs 분해 질문 (판정 기준은 둘 다 팀 원문).

4개 분야마다 기술 요약이 쓰는 업계 기준선 질문으로 Agentic RAG를 돌려
  통과 문서 수(중복 제외), 재작성 횟수(질문별 최댓값), 웹 보완 여부
를 비교한다. 결과: outputs/rag_question_eval.md (결정 근거: docs/EXPERIMENTS.md 5번)

    python -m eval.rag_question_eval

LLM 판정과 웹 검색을 실제로 호출하므로 API 키가 필요하다.
"""
from agents.tech_summary import SEGMENT_EN, industry_questions
from config import OUTPUT_DIR
from rag import agentic_rag
from schemas import SEGMENT_KO

MODES = ("team", "decomposed")


def measure(segment: str, mode: str) -> dict:
    docs, rewrites, web = set(), 0, False
    for question in industry_questions(segment, mode):
        out = agentic_rag.run(question, web_queries=[question, f"{SEGMENT_EN[segment]} AI technology readiness commercialization"])
        docs |= {f"{d.metadata['doc_id']} p.{d.metadata['page']}" for d in out["relevant"]}
        rewrites = max(rewrites, out["rewrites"])
        web = web or bool(out["web_results"])
    return {"docs": len(docs), "pages": sorted(docs), "rewrites": rewrites, "web": web}


def main() -> None:
    yn = lambda flag: "예" if flag else "아니오"  # noqa: E731
    lines = ["# 질문 방식 비교 (Agentic RAG, 판정 기준: 팀 원문)", "",
             "| 분야 | 팀 질문: 통과 문서 | 재작성 | 웹 보완 | 분해 질문: 통과 문서 | 재작성 | 웹 보완 |",
             "|---|---:|---:|:---:|---:|---:|:---:|"]
    total, detail = {mode: 0 for mode in MODES}, []
    for segment, ko in SEGMENT_KO.items():
        r = {mode: measure(segment, mode) for mode in MODES}
        print(f"{ko}: " + ", ".join(f"{m} {r[m]['docs']}개" for m in MODES), flush=True)
        lines.append(f"| {ko} | {r['team']['docs']} | {r['team']['rewrites']} | {yn(r['team']['web'])} "
                     f"| {r['decomposed']['docs']} | {r['decomposed']['rewrites']} | {yn(r['decomposed']['web'])} |")
        for mode in MODES:
            total[mode] += r[mode]["docs"]
            detail.append(f"- {ko} / {mode}: {', '.join(r[mode]['pages']) or '없음'}")
    lines.append(f"| **합계** | **{total['team']}** | | | **{total['decomposed']}** | | |")
    text = "\n".join(lines + ["", "## 통과한 문서", ""] + detail) + "\n"
    (OUTPUT_DIR / "rag_question_eval.md").write_text(text, encoding="utf-8")
    print(f"\n저장: {OUTPUT_DIR / 'rag_question_eval.md'}")


if __name__ == "__main__":
    main()
