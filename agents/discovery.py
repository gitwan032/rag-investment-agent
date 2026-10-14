"""스타트업 탐색 에이전트 (웹 검색, RAG 미사용).

1) 세그먼트별 검색어로 투자 뉴스를 찾는다 (한국어·영어)
2) LLM이 검색 결과에서 후보 회사를 뽑는다
3) 후보마다 상장·인수 여부와 투자 단계를 다시 검색해, 판정 LLM이 요건 충족 여부를 판단한다
4) 요건을 충족한 후보를 '공개 정보 충분도' 순으로 정렬해 대기열을 만든다

팀 결과물에서는 정보가 부족한 후보가 많아 점수 대부분이 기본값(5점)으로 채워졌다.
그래서 이 구현은 후보 순서를 정할 때 '평가할 근거가 충분한가'를 기준으로 쓴다.

    python -m agents.discovery   # 탐색만 단독 실행
"""
import json
from typing import Literal

from pydantic import BaseModel, Field

from config import DATA_DIR, MAX_CANDIDATES_PER_ROUND, OUTPUT_DIR, PROMPT_DIR
from schemas import SEGMENT_KO, Candidate
from tools.llm import ask_structured, get_judge, get_llm
from tools.references import make_web_ref
from tools.web_search import format_results, search

QUERIES = {
    "demand_forecasting": ["AI 전력 수요 예측 스타트업 투자 유치", "AI electricity load forecasting startup raises Series"],
    "generation_vpp": ["재생에너지 발전량 예측 VPP 스타트업 투자 유치", "virtual power plant AI startup funding round"],
    "ess_operation": ["ESS 배터리 운영 최적화 AI 스타트업 투자", "battery storage optimization AI startup raises Series A"],
    "grid_management": ["AI 전력망 관리 스타트업 투자 유치", "AI grid management software startup raises funding"],
}
MAX_VERIFY = 10  # 회차당 검증할 최대 후보 수 (검색·LLM 비용 상한)


class CandidateList(BaseModel):
    candidates: list[Candidate]


class Eligibility(BaseModel):
    listed: bool = Field(description="상장 기업이면 true")
    exited: bool = Field(description="인수합병·IPO로 Exit했으면 true")
    stage: Literal["Seed", "Series A", "Series B", "Series C", "Later", "Unknown"]
    is_energy_ai: bool = Field(description="AI로 에너지 문제를 푸는 회사면 true")
    info_score: int = Field(description="평가에 쓸 공개 정보 충분도 1~5")
    reason: str = Field(description="판단 근거 한두 문장")


def _prompt(template: str, **values) -> str:
    return (PROMPT_DIR / f"{template}.md").read_text(encoding="utf-8").format(**values)


def _extract(segment: str, round_no: int, excluded: set[str]) -> tuple[list[Candidate], list[dict]]:
    """세그먼트 검색 → 후보 추출. 회차마다 검색어 순서를 바꿔 다른 결과를 얻는다."""
    queries = QUERIES[segment] if round_no % 2 else list(reversed(QUERIES[segment]))
    results = []
    for query in queries:
        results += search(query, max_results=5, topic="news", time_range="year")
    if not results:
        return [], []
    prompt = _prompt("discovery_extract", segment_ko=SEGMENT_KO[segment],
                     excluded=", ".join(sorted(excluded)) or "없음", results=format_results(results))
    found = ask_structured(get_llm(), CandidateList, prompt).candidates
    return found, results


def _verify(candidate: Candidate) -> tuple[Eligibility, list[dict]]:
    """상장·인수 여부와 투자 단계를 다시 검색해 판정한다."""
    results = search(f'"{candidate.name}" funding round Series investors', max_results=5)
    results += search(f'"{candidate.name}" IPO OR acquired OR 상장 OR 인수', max_results=3)
    prompt = _prompt("discovery_verify", name=candidate.name, country=candidate.country,
                     segment_ko=SEGMENT_KO[candidate.segment], homepage=candidate.homepage or "모름",
                     funding=candidate.funding, results=format_results(results))
    return ask_structured(get_judge(), Eligibility, prompt), results


def _seed_candidates(excluded: set[str]) -> list[dict]:
    seeds = json.loads((DATA_DIR / "seed_candidates.json").read_text(encoding="utf-8"))
    return [{**s, "source": "seed", "info_score": 0, "evidence_urls": []}
            for s in seeds if s["name"].casefold() not in excluded][:MAX_CANDIDATES_PER_ROUND]


def run(state: dict) -> dict:
    round_no = state.get("search_round", 0) + 1
    excluded = {name.casefold() for name in state.get("evaluated", [])}
    run_date = state["run_date"]

    # 1~2) 세그먼트별 검색 → 후보 추출
    pool: dict[str, tuple[Candidate, list[dict]]] = {}
    for segment in QUERIES:
        found, results = _extract(segment, round_no, excluded)
        for c in found:
            key = c.name.casefold()
            if key not in excluded and key not in pool:
                evidence = [results[i - 1] for i in c.evidence if 1 <= i <= len(results)]
                pool[key] = (c, evidence)

    # 3) 요건 검증
    accepted, rejected, references = [], [], []
    for c, evidence in list(pool.values())[:MAX_VERIFY]:
        check, verify_results = _verify(c)
        ok = (not check.listed and not check.exited and check.is_energy_ai
              and check.stage in ("Seed", "Series A", "Series B", "Series C"))
        record = {**c.model_dump(exclude={"evidence"}), "stage": check.stage, "info_score": min(5, max(1, check.info_score)),
                  "reason": check.reason, "source": "web",
                  "evidence_urls": [r["url"] for r in evidence + verify_results]}
        (accepted if ok else rejected).append(record)
        if ok:
            references += [make_web_ref(r, c.name, "discovery", run_date) for r in evidence + verify_results]

    # 4) 공개 정보 충분도 순으로 정렬
    accepted.sort(key=lambda r: r["info_score"], reverse=True)
    candidates = accepted[:MAX_CANDIDATES_PER_ROUND]
    source = "웹 검색"
    if not candidates:  # 웹에서 하나도 못 찾았을 때만 시드 목록 사용 (보고서·로그에 명시)
        candidates, source = _seed_candidates(excluded), "시드 목록(웹 발굴 실패)"
        references = []

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / f"discovery_round{round_no}.json").write_text(
        json.dumps({"source": source, "accepted": candidates, "rejected": rejected}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    names = ", ".join(c["name"] for c in candidates)
    return {
        "candidates": candidates,
        "search_round": round_no,
        "references": references,
        "log": [f"[discover] {round_no}회차 {source}: 후보 {len(candidates)}곳 ({names}), "
                f"검증 탈락 {len(rejected)}곳"],
    }


if __name__ == "__main__":
    from datetime import date

    out = run({"search_round": 0, "evaluated": [], "run_date": date.today().isoformat()})
    print(out["log"][0])
    for c in out["candidates"]:
        print(f"- {c['name']} ({c['country']}, {SEGMENT_KO[c['segment']]}, {c['stage']}, "
              f"정보 {c['info_score']}/5) {c.get('reason', '')}")
    print(f"출처 {len(out['references'])}건, 상세: outputs/discovery_round1.json")
