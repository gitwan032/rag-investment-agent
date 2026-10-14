"""시장성 평가 에이전트 (Advanced RAG: 시장 보고서 3종).

검색 전 : 국가·세그먼트 메타데이터 필터 + 질문 분해(규모 / 성장 / 수요 / 정책, 팀 설계의 4개 질문)
검색    : Dense(bge-m3) + Reranker (3단계 평가로 결정)
검색 후 : 근거 번호 검증 + 시장 수치는 '원문 문장 인용'이 실제 근거 문서에 있는지 코드로 확인

팀 보고서에서는 시장 수치가 원문과 다른 뜻으로 인용됐다
(예: 전 세계 재생에너지 총투자 5.2조 달러 → "한국 VPP 시장 규모").
그래서 수치마다 원문 문장(quote)과 대상(metric)·지역(scope)을 함께 받고,
quote가 인용한 문서에 실제로 없거나 숫자가 quote에 없으면 그 수치를 버린다.

같은 국가·세그먼트는 시장 분석이 같으므로 market_cache에 저장해 재사용한다.

    python -m agents.market_eval   # outputs/discovery_round1.json 첫 후보로 단독 실행
"""
import json
import re

from config import OUTPUT_DIR, PROMPT_DIR
from rag.search import search_docs
from schemas import SEGMENT_KO, MarketAnalysis
from tools.citations import NOT_FOUND, check_text, keep_cited
from tools.llm import ask_structured, get_llm
from tools.references import make_doc_ref

DOC_COUNTRIES = {"KR", "US"}  # 시장 보고서가 다루는 개별 국가 (나머지는 GLOBAL 보고서만 사용)
COUNTRY_KO = {"KR": "한국", "US": "미국", "GLOBAL": "글로벌"}


def _filter(country: str, segment: str, use_segment: bool) -> dict:
    countries = [country, "GLOBAL"] if country in DOC_COUNTRIES else ["GLOBAL"]
    by_country = {"country": {"$in": countries}}
    return {"$and": [by_country, {f"seg_{segment}": True}]} if use_segment else by_country


def _questions(country: str, segment: str) -> list[str]:
    seg, nation = SEGMENT_KO[segment], COUNTRY_KO.get(country, "글로벌")
    return [f"{seg} 시장 규모와 전망 수치", f"{seg} 시장 성장률과 성장 요인",
            f"{seg} 수요처와 도입 수요", f"{nation} {seg} 관련 정책·규제 환경과 리스크"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _quote_supported(quote: str, value: str, source_text: str) -> bool:
    """quote가 근거 문서에 있고(단어 80% 이상 일치), value의 숫자가 quote에 들어 있는가."""
    q, src = _normalize(quote), _normalize(source_text)
    words = q.split()
    found = q in src or (words and sum(w in src for w in words) / len(words) >= 0.8)
    digits = re.findall(r"\d+(?:[.,]\d+)?", value)
    return bool(found) and all(d in quote for d in digits)


def run(state: dict) -> dict:
    company = state["current"]
    name, segment = company["name"], company["segment"]
    country = company["country"] if company["country"] in DOC_COUNTRIES else "GLOBAL"
    key = f"{country}|{segment}"

    cached = (state.get("market_cache") or {}).get(key)
    if cached:
        refs = [make_doc_ref(doc_id, name, "market_eval") for doc_id in cached["source_docs"]]
        return {"market_analysis": cached, "references": refs,
                "log": [f"[market_eval] {name}: {key} 시장 분석 재사용 (캐시)"]}

    # 검색: 질문 4개 × 상위 4개, 필터 결과가 적으면 세그먼트 조건을 푼다
    chunks, seen = [], set()
    for question in _questions(country, segment):
        docs = search_docs("market", question, k=4, filter=_filter(country, segment, True))
        if len(docs) < 2:
            docs = search_docs("market", question, k=4, filter=_filter(country, segment, False))
        for d in docs:
            if d.id not in seen:
                seen.add(d.id)
                chunks.append(d)

    evidence = "\n\n".join(f"[{i}] {d.metadata['title']} ({d.metadata['year']}) p.{d.metadata['page']}\n{d.page_content}"
                           for i, d in enumerate(chunks, 1))
    prompt = (PROMPT_DIR / "market_eval.md").read_text(encoding="utf-8").format(
        segment_ko=SEGMENT_KO[segment], company=name, country=company["country"], evidence=evidence or "근거 없음")
    analysis = ask_structured(get_llm(), MarketAnalysis, prompt)

    # 근거 검증
    n, used = len(chunks), set()
    figures, dropped = [], 0
    for f in analysis.figures:
        if 1 <= f.source <= n and _quote_supported(f.quote, f.value, chunks[f.source - 1].page_content):
            figures.append(f)
            used.add(f.source)
        else:
            dropped += 1
    analysis.figures = figures
    for field in ("demand_drivers", "customers", "policy_risks"):
        setattr(analysis, field, keep_cited(getattr(analysis, field), n, used))
    gaps = list(analysis.gaps)
    for field in ("growth", "summary"):
        setattr(analysis, field, check_text(getattr(analysis, field), n, used))
        if getattr(analysis, field) == NOT_FOUND:
            gaps.append(f"{field}: 근거 확인 불가")
    if not figures:
        gaps.append("시장 규모 수치를 원문 인용으로 확인하지 못함")
    analysis.gaps = gaps

    result = analysis.model_dump()
    for fig in result["figures"]:  # 보고서 표에 출처를 적기 위해 문서 정보를 붙인다
        meta = chunks[fig["source"] - 1].metadata
        fig["source_title"], fig["source_year"], fig["page"] = meta["title"], meta["year"], meta["page"]
    source_docs = sorted({chunks[i - 1].metadata["doc_id"] for i in used})
    result.update({"segment": segment, "country_scope": country, "source_docs": source_docs})
    refs = [make_doc_ref(doc_id, name, "market_eval") for doc_id in source_docs]
    return {
        "market_analysis": result,
        "market_cache": {key: result},
        "references": refs,
        "log": [f"[market_eval] {name}: {key} 문서 청크 {n}개 검색, 수치 {len(figures)}개 채택 "
                f"(원문 인용 불일치로 {dropped}개 제외), 근거 문서 {len(source_docs)}종"],
    }


if __name__ == "__main__":
    first = json.loads((OUTPUT_DIR / "discovery_round1.json").read_text(encoding="utf-8"))["accepted"][0]
    out = run({"current": first, "market_cache": {}})
    print("\n".join(out["log"]))
    print(json.dumps(out["market_analysis"], ensure_ascii=False, indent=2))
