"""투자 판단 에이전트: LLM은 항목별 점수와 근거만, 합산·판정은 코드가 한다 (팀 설계 그대로).

평가표 config/rubric.yaml (팀 설계): 11개 항목, 가중치 합 100
  항목 점수 = 0~10점 × 가중치 ÷ 10 → 총점 100점 만점
  총점 70점 이상 → 투자 / 미만 → 보류
  필수 탈락: 창업팀 역량 또는 규제·정책 위험이 3점 이하면 총점과 관계없이 보류
  근거가 없는 항목은 중간값 5점 + '정보 부족' 표시 (보고서 '분석의 한계'에 명시)
"""
import json
import re
from functools import lru_cache

import yaml

from config import CONFIG_DIR, INVEST_THRESHOLD, OUTPUT_DIR, PROMPT_DIR
from schemas import JudgeOutput
from tools.citations import strip
from tools.llm import ask_structured, get_judge

CACHE_DIR = OUTPUT_DIR / "cache"


@lru_cache(maxsize=1)
def load_rubric() -> list[dict]:
    rubric = yaml.safe_load((CONFIG_DIR / "rubric.yaml").read_text(encoding="utf-8"))
    items = rubric["items"]
    assert sum(i["weight"] for i in items) == 100 and len(items) == 11, "평가표: 11개 항목, 가중치 합 100"
    return items


def compute(scores: dict) -> tuple[float, str, str | None]:
    """가중 합산 → (총점, 판정, 탈락 사유). LLM 출력과 무관하게 같은 입력이면 같은 결과."""
    total, knockout = 0, None
    for item in load_rubric():
        s = scores[item["id"]]
        total += s["score"] * item["weight"]
        if item.get("knockout") and not s["insufficient"] and s["score"] <= 3:
            knockout = f"{item['name']} {s['score']}점 (3점 이하)"
    total = total / 10
    decision = "투자" if total >= INVEST_THRESHOLD and knockout is None else "보류"
    return total, decision, knockout


def _clean(value):
    """분석 결과의 내부 근거 번호를 지우고 넘긴다 (채점 LLM에는 번호가 의미 없음)."""
    if isinstance(value, str):
        return strip(value)
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in ("source", "quote", "evidence_count")}
    return value


def score(prompt: str, judge=None) -> dict:
    """채점 LLM 호출 → 항목별 점수. 근거가 없다고 답한 항목은 중간값 5점 + 정보 부족."""
    rubric = load_rubric()
    output = None
    for _ in range(2):  # 11개 항목이 빠짐없이 나올 때까지 한 번 더
        output = ask_structured(judge or get_judge(), JudgeOutput, prompt)
        if {i.item_id for i in output.items} == {r["id"] for r in rubric}:
            break
    by_id = {i.item_id: i for i in output.items}
    scores = {}
    for item in rubric:
        s = by_id.get(item["id"])
        insufficient = s is None or not s.evidence_found
        scores[item["id"]] = {
            "name": item["name"], "weight": item["weight"],
            "score": 5 if insufficient else min(10, max(0, s.score)),
            "insufficient": insufficient,
            "rationale": ("정보 부족(중간값 5점 적용). " if insufficient else "") + (s.rationale if s else "채점 누락"),
        }
    return scores


def run(state: dict) -> dict:
    company = state["current"]
    name = company["name"]
    rubric = load_rubric()
    rubric_text = "\n".join(
        f"- {i['id']} | {i['name']} (가중치 {i['weight']}) | 상: {i['high']} / 중: {i['mid']} / 하: {i['low']}"
        for i in rubric)
    analysis = {
        "기업 정보": {k: company.get(k) for k in ("name", "country", "segment", "stage", "funding", "reason")},
        "기술 요약": _clean(state.get("tech_summary") or {}),
        "시장성": _clean(state.get("market_analysis") or {}),
        "경쟁사": _clean(state.get("competitor_analysis") or {}),
    }
    prompt = (PROMPT_DIR / "investment_judge.md").read_text(encoding="utf-8").format(
        company=name, rubric=rubric_text, analysis=json.dumps(analysis, ensure_ascii=False, indent=1))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # 같은 입력으로 다시 채점해 일관성을 재기 위해 저장
    (CACHE_DIR / f"judge_prompt_{re.sub(r'[^0-9A-Za-z가-힣]+', '_', name)}.txt").write_text(prompt, encoding="utf-8")
    scores = score(prompt)

    total, decision, knockout = compute(scores)
    weakest = sorted(rubric, key=lambda i: scores[i["id"]]["score"] * i["weight"])[:3]
    insufficient = [scores[i["id"]]["name"] for i in rubric if scores[i["id"]]["insufficient"]]
    reason = f"총점 {total:.1f}점 (기준 {INVEST_THRESHOLD}점)"
    if knockout:
        reason += f", 필수 탈락: {knockout}"
    record = {
        "startup": name, "country": company["country"], "segment": company["segment"], "stage": company["stage"],
        "total_score": total, "decision": decision, "knockout": knockout,
        "weakest": [scores[i["id"]]["name"] for i in weakest], "insufficient": insufficient,
        "key_reason": reason, "scores": scores,
        # 보류 보고서에서 후보별 비교에 쓰도록 핵심 요약을 함께 남긴다 (팀 보고서는 이전 후보 상세가 사라졌음)
        "profile": {
            "core_tech": strip((state.get("tech_summary") or {}).get("core_tech", "")),
            "trl": (state.get("tech_summary") or {}).get("trl"),
            "differentiation": strip((state.get("competitor_analysis") or {}).get("differentiation", "")),
        },
    }
    return {
        "scores": scores, "total_score": total, "decision": decision,
        "history": [record],
        "log": [f"[judge] {name}: {decision} — {reason}, 정보 부족 {len(insufficient)}개 항목"],
    }
