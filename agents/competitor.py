"""경쟁사 비교 에이전트 (웹 검색, RAG 미사용).

기술 요약·시장성 평가 결과를 받아(그래프의 합류 지점) 경쟁사 3~5곳과 차별점·진입장벽·사업 실적을 정리한다.
근거 번호가 없는 주장과 대상 기업 자신은 코드가 지운다.

    python -m agents.competitor   # outputs/discovery_round1.json 첫 후보로 단독 실행
"""
import json

from agents.tech_summary import SEGMENT_EN
from config import OUTPUT_DIR, PROMPT_DIR
from schemas import SEGMENT_KO, CompetitorAnalysis
from tools.citations import NOT_FOUND, cited, check_text, keep_cited, strip
from tools.llm import ask_structured, get_llm
from tools.references import make_web_ref
from tools.web_search import format_results, search


def run(state: dict) -> dict:
    company = state["current"]
    name, segment, run_date = company["name"], company["segment"], state["run_date"]

    seen, results = set(), []
    for query in (f'"{name}" competitors alternatives', f"{SEGMENT_EN[segment]} AI startups companies",
                  f'"{name}" customers partnership contract'):
        for r in search(query, max_results=5):
            if r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)

    tech = state.get("tech_summary") or {}
    market = state.get("market_analysis") or {}
    prompt = (PROMPT_DIR / "competitor.md").read_text(encoding="utf-8").format(
        company=name,
        company_info=f"{name} / {company['country']} / {SEGMENT_KO[segment]} / {company['stage']} / {company['funding']}",
        tech=strip(tech.get("core_tech", "없음")) + " 강점: " + "; ".join(strip(s) for s in tech.get("strengths", [])),
        market=strip(market.get("summary", "없음")),
        evidence=format_results(results) or "근거 없음")
    analysis = ask_structured(get_llm(), CompetitorAnalysis, prompt)

    n, used = len(results), set()
    competitors = []
    for c in analysis.competitors:
        nums = cited(c.difference, n)
        if nums and c.name.strip().casefold() != name.casefold():
            competitors.append(c)
            used |= nums
    analysis.competitors = competitors
    for field in ("moat", "traction", "risks"):
        setattr(analysis, field, keep_cited(getattr(analysis, field), n, used))
    analysis.differentiation = check_text(analysis.differentiation, n, used)
    gaps = list(analysis.gaps)
    if analysis.differentiation == NOT_FOUND:
        gaps.append("차별점 근거 확인 불가")
    if len(competitors) < 3:
        gaps.append(f"근거가 확인된 경쟁사 {len(competitors)}곳 (3곳 미만)")
    if not analysis.traction:
        gaps.append("사업 실적(고객·계약·매출) 확인 불가")
    analysis.gaps = gaps

    refs = [make_web_ref(results[i - 1], name, "competitor", run_date) for i in sorted(used)]
    return {
        "competitor_analysis": analysis.model_dump(),
        "references": refs,
        "log": [f"[competitor] {name}: 경쟁사 {len(competitors)}곳, 실적 {len(analysis.traction)}건, "
                f"진입장벽 {len(analysis.moat)}건, 인용 {len(used)}건"],
    }


if __name__ == "__main__":
    from datetime import date

    first = json.loads((OUTPUT_DIR / "discovery_round1.json").read_text(encoding="utf-8"))["accepted"][0]
    out = run({"current": first, "run_date": date.today().isoformat()})
    print("\n".join(out["log"]))
    print(json.dumps(out["competitor_analysis"], ensure_ascii=False, indent=2))
