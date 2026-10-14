"""Agentic RAG 서브그래프 (기술 요약 에이전트가 사용).

    retrieve → grade ─┬→ (관련 문서 2개 이상) → END
                      ├→ rewrite → retrieve      (재작성 최대 2회)
                      └→ web_search → END        (재작성해도 부족하면 웹으로 보완)

검색 결과를 그대로 믿지 않고, 판정 LLM이 질문과 관련 있는지 확인한 뒤
부족하면 검색어를 고쳐 다시 찾고, 그래도 없으면 웹으로 보완한다 (수업 3장 Corrective / Agentic RAG).

    python -m rag.agentic_rag "전력 수요 예측 AI의 상용화 수준은?"
"""
import re
import sys
from operator import add
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import PROMPT_DIR, RAG_MAX_REWRITES
from rag.search import search_docs
from tools.llm import ask_structured, get_judge, get_llm
from tools.web_search import search

MIN_RELEVANT = 2
NO_INFO = re.compile(r"(공개\s*정보|(public\s+)?information)\s*(부족|없음|insufficient|unavailable)", re.I)


class RagState(TypedDict, total=False):
    question: str               # 원래 질문 (판정 기준)
    query: str                  # 실제 검색어 (재작성되면 바뀜)
    web_queries: list[str]      # 문서가 부족할 때 쓸 웹 검색어들
    rewrites: int
    relevant: list[Document]    # 판정을 통과한 문서 (누적)
    seen: list[str]             # 이미 판정한 청크 ID (중복 판정 방지)
    candidates: list[Document]  # 이번 검색 결과
    web_results: list[dict]
    log: Annotated[list[str], add]


class Grades(BaseModel):
    relevant: list[bool] = Field(description="문서 번호 순서대로, 질문에 답하는 데 쓸 수 있으면 true")


def _prompt(template: str, **values) -> str:
    return (PROMPT_DIR / f"{template}.md").read_text(encoding="utf-8").format(**values)


def retrieve(state: RagState) -> dict:
    docs = search_docs("tech", state["query"], k=5)
    new = [d for d in docs if d.id not in set(state.get("seen", []))]
    return {"candidates": new, "log": [f"retrieve '{state['query']}' → 새 문서 {len(new)}개"]}


def grade_docs(question: str, docs: list[Document], judge=None) -> list[bool]:
    """문서마다 질문에 답하는 데 쓸 수 있는지 판정 (팀 원문 판정 프롬프트)."""
    listing = "\n\n".join(f"[{i}] {d.page_content[:1200]}" for i, d in enumerate(docs, 1))
    return ask_structured(judge or get_judge(), Grades, _prompt("rag_grade", question=question, docs=listing)).relevant


def grade(state: RagState) -> dict:
    docs = state.get("candidates", [])
    relevant = list(state.get("relevant", []))
    if docs:
        verdicts = grade_docs(state["question"], docs)
        relevant += [d for d, ok in zip(docs, verdicts) if ok]
        marks = ", ".join(f"{d.metadata['doc_id']} p.{d.metadata['page']}{'(O)' if ok else '(X)'}"
                          for d, ok in zip(docs, verdicts))
    else:
        marks = "새 문서 없음"
    seen = list(state.get("seen", [])) + [d.id for d in docs]
    return {"relevant": relevant, "seen": seen, "log": [f"grade [{marks}] → 관련 문서 누적 {len(relevant)}개"]}


def route(state: RagState) -> str:
    if len(state.get("relevant", [])) >= MIN_RELEVANT:
        return "done"
    return "rewrite" if state.get("rewrites", 0) < RAG_MAX_REWRITES else "web_search"


def rewrite(state: RagState) -> dict:
    query = get_llm().invoke(_prompt("rag_rewrite", question=state["question"], query=state["query"])).content.strip()
    # 팀 프롬프트의 "모르면 '공개 정보 부족'" 규칙 때문에 검색어에 이 말(영문 변형 포함)이 붙는 경우가 있어 지운다
    # (프롬프트는 팀 원문 유지). 예: "... 공개 정보 부족", "... public information 부족"
    query = " ".join(NO_INFO.sub(" ", query).split())
    count = state.get("rewrites", 0) + 1
    return {"query": query or state["query"], "rewrites": count, "log": [f"rewrite #{count} → '{query}'"]}


def web_search(state: RagState) -> dict:
    seen, results = set(), []
    for query in state.get("web_queries") or [state["question"]]:
        for r in search(query, max_results=5):
            if r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)
    return {"web_results": results, "log": [f"web_search 보완 → {len(results)}건"]}


def build():
    g = StateGraph(RagState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("web_search", web_search)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, {"done": END, "rewrite": "rewrite", "web_search": "web_search"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("web_search", END)
    return g.compile()


def run(question: str, web_queries: list[str] | None = None) -> dict:
    """반환: {relevant: [Document], web_results: [dict], rewrites: int, log: [str]}"""
    out = build().invoke({"question": question, "query": question, "web_queries": web_queries or [question],
                          "rewrites": 0, "relevant": [], "seen": [], "web_results": [], "log": []})
    return {"relevant": out["relevant"], "web_results": out.get("web_results", []),
            "rewrites": out.get("rewrites", 0), "log": out["log"]}


if __name__ == "__main__":
    result = run(sys.argv[1] if len(sys.argv) > 1 else "전력 수요 예측 AI 기술의 상용화 수준은?")
    print("\n".join(result["log"]))
    for d in result["relevant"]:
        print(f"- {d.metadata['doc_id']} p.{d.metadata['page']}: {d.page_content[:120]}...")
