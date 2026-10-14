"""기술 요약 에이전트 (Agentic RAG + 웹 검색).

1) 웹 검색: 회사 고유 정보 (핵심 기술·제품·성능, 창업팀)
   → 범용 기술 보고서에는 개별 스타트업 정보가 없으므로 웹으로 찾는다
2) Agentic RAG: 업계 기술 수준·상용화 단계(TRL 판단 기준선)
   → 기술 보고서 3종에서 찾고, 관련 문서가 부족하면 검색어 재작성 → 웹 보완
3) LLM이 두 근거를 합쳐 TechSummary 작성. 모든 주장에 근거 번호 [n]을 붙이게 하고,
   번호가 없거나 존재하지 않는 번호를 단 항목은 코드가 지운다 (환각 차단)

    python -m agents.tech_summary            # outputs/discovery_round1.json 첫 후보로 단독 실행
"""
import json

from config import OUTPUT_DIR, PROMPT_DIR
from rag import agentic_rag
from schemas import SEGMENT_KO, TechSummary
from tools.citations import NOT_FOUND, cited, check_text, keep_cited
from tools.llm import ask_structured, get_llm
from tools.references import make_doc_ref, make_web_ref
from tools.web_search import search

SEGMENT_EN = {
    "demand_forecasting": "electricity demand forecasting",
    "generation_vpp": "renewable generation forecasting and virtual power plant",
    "ess_operation": "battery energy storage optimization",
    "grid_management": "power grid management",
}


def industry_questions(segment: str, mode: str = "team") -> list[str]:
    """업계 기준선(TRL·성능 비교 근거) 질문.

    team       : 팀 설계 원문 한 문장 (실제 실행에 사용)
    decomposed : 같은 목적을 상용화 단계 / 성능 수치 두 질문으로 분해 (수업 3장 Query Transformation)
    두 방식 비교는 eval/rag_question_eval.py, 결정 근거는 docs/EXPERIMENTS.md 5번.
    판정 기준(prompts/rag_grade.md)은 어느 쪽이든 팀 원문 그대로다.
    """
    ko = SEGMENT_KO[segment]
    if mode == "decomposed":
        return [f"{ko} 분야에서 AI 기술은 실제로 어디까지 적용·상용화되어 있는가?",
                f"{ko} 분야에서 AI 적용 시 성능 개선 효과(정확도, 비용 절감 등 수치)는?"]
    return [f"{ko} 분야 AI 기술의 업계 수준과 상용화 단계(TRL 판단 근거)는?"]


def _company_evidence(name: str) -> list[dict]:
    """회사 고유 정보 웹 검색 (기술·제품·성과 / 창업팀). 같은 URL은 한 번만."""
    queries = [f'"{name}" AI technology product platform',
               f'"{name}" customers accuracy results case study',
               f'"{name}" founder CEO CTO background']
    seen, results = set(), []
    for q in queries:
        for r in search(q, max_results=5):
            if r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)
    return results


def run(state: dict) -> dict:
    company = state["current"]
    name, segment, run_date = company["name"], company["segment"], state["run_date"]

    # 1) 회사 고유 정보 (웹)
    web = _company_evidence(name)

    # 2) 업계 기준선 (Agentic RAG)
    industry_docs, industry_web, rag_logs, rewrites, seen_docs, seen_urls = [], [], [], 0, set(), set()
    for question in industry_questions(segment):
        rag = agentic_rag.run(question, web_queries=[question, f"{SEGMENT_EN[segment]} AI technology readiness commercialization"])
        for doc in rag["relevant"]:          # 질문이 여러 개면 같은 문서가 겹칠 수 있어 한 번만 담는다
            if doc.id not in seen_docs:
                seen_docs.add(doc.id)
                industry_docs.append(doc)
        for result in rag["web_results"]:
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                industry_web.append(result)
        rewrites += rag["rewrites"]
        rag_logs += [f"Q: {question}"] + rag["log"]
    industry_docs = industry_docs[:6]

    # 근거 목록: 번호 → (종류, 원본)
    evidence = [("web", r) for r in web] + [("doc", d) for d in industry_docs] + [("web_industry", r) for r in industry_web]
    lines = []
    for i, (kind, item) in enumerate(evidence, 1):
        if kind == "doc":
            lines.append(f"[{i}] [업계] {item.metadata['title']} p.{item.metadata['page']}\n{item.page_content[:1500]}")
        else:
            tag = "회사" if kind == "web" else "업계"
            lines.append(f"[{i}] [{tag}] {item['title']} ({item['url']})\n{item['content'][:800]}")
    info = f"{name} / 국가 {company['country']} / 분야 {SEGMENT_KO[segment]} / 단계 {company['stage']} / 투자 {company['funding']}"
    prompt = (PROMPT_DIR / "tech_summary.md").read_text(encoding="utf-8").format(
        company=name, company_info=info, evidence="\n\n".join(lines) or "근거 없음")

    # 3) 요약 생성 + 근거 번호 검증
    summary = ask_structured(get_llm(), TechSummary, prompt)
    n, used = len(evidence), set()
    for field in ("products", "performance", "strengths", "weaknesses"):
        setattr(summary, field, keep_cited(getattr(summary, field), n, used))
    founders = []
    for founder in summary.founders:
        nums = cited(founder.background, n)
        if nums:
            founders.append(founder)
            used |= nums
    summary.founders = founders
    summary.trl = min(9, max(1, summary.trl))
    gaps = list(summary.gaps)
    for field in ("core_tech", "trl_rationale", "scalability", "team_assessment"):
        setattr(summary, field, check_text(getattr(summary, field), n, used))
        if getattr(summary, field) == NOT_FOUND:
            gaps.append(f"{field}: 근거 확인 불가")
    if not summary.founders:
        gaps.append("창업팀 정보 확인 불가")
    summary.gaps = gaps

    references = []
    for i in sorted(used):
        kind, item = evidence[i - 1]
        references.append(make_doc_ref(item.metadata["doc_id"], name, "tech_summary") if kind == "doc"
                          else make_web_ref(item, name, "tech_summary", run_date))

    result = summary.model_dump()
    result["evidence_count"] = {"web": len(web), "industry_docs": len(industry_docs), "industry_web": len(industry_web)}
    return {
        "tech_summary": result,
        "references": references,
        "log": [f"[tech_summary] {name}: TRL {summary.trl}, 웹 근거 {len(web)}건, 업계 문서 {len(industry_docs)}건"
                f"(재작성 {rewrites}회), 인용 {len(used)}건, 부족 {len(gaps)}건"]
                + [f"  [agentic_rag] {line}" for line in rag_logs],
    }


if __name__ == "__main__":
    from datetime import date

    first = json.loads((OUTPUT_DIR / "discovery_round1.json").read_text(encoding="utf-8"))["accepted"][0]
    out = run({"current": first, "run_date": date.today().isoformat()})
    print("\n".join(out["log"]))
    print(json.dumps(out["tech_summary"], ensure_ascii=False, indent=2))
    print(f"출처 {len(out['references'])}건")
