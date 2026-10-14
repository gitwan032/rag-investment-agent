"""4단계용 가짜 노드. LLM·웹검색 없이 그래프 흐름(루프·분기·병렬)만 검증한다.

STUB_DECISIONS 환경변수로 판정 순서를 정할 수 있다. 예: "보류,보류,투자"
지정한 순서를 다 쓰면 이후는 모두 '보류'.
"""
import os

from config import MAX_CANDIDATES_PER_ROUND, OUTPUT_DIR


def nodes(decisions: list[str] | None = None) -> dict:
    if decisions is None:
        decisions = [d.strip() for d in os.getenv("STUB_DECISIONS", "보류,투자").split(",") if d.strip()]
    queue = list(decisions)

    def discover(state):
        round_no = state.get("search_round", 0) + 1
        found = [{"name": f"가상기업{round_no}-{i}", "segment": "generation_vpp", "country": "KR"}
                 for i in range(1, MAX_CANDIDATES_PER_ROUND + 1)]
        done = set(state.get("evaluated", []))
        found = [c for c in found if c["name"] not in done][:2]  # 흐름 확인용으로 회차당 2곳만
        return {"candidates": found, "search_round": round_no,
                "log": [f"[discover] {round_no}회차 후보 {len(found)}곳"]}

    def select(state):
        queue_left = list(state["candidates"])
        current = queue_left.pop(0)
        return {"candidates": queue_left, "current": current, "evaluated": [current["name"]],
                "tech_summary": None, "market_analysis": None, "competitor_analysis": None,
                "scores": {}, "total_score": 0.0, "decision": None,
                "log": [f"[select] {current['name']} 평가 시작 (남은 후보 {len(queue_left)})"]}

    def tech_summary(state):
        name = state["current"]["name"]
        return {"tech_summary": {"core_tech": f"{name} 핵심 기술(가짜)"},
                "references": [{"id": f"tech:{name}", "startup": name}],
                "log": [f"[tech_summary] {name}"]}

    def market_eval(state):
        name = state["current"]["name"]
        return {"market_analysis": {"summary": "시장 분석(가짜)"},
                "references": [{"id": f"market:{name}", "startup": name}],
                "log": [f"[market_eval] {name}"]}

    def competitor(state):
        name = state["current"]["name"]
        assert state.get("tech_summary") and state.get("market_analysis"), "합류 전에 실행됨"
        return {"competitor_analysis": {"competitors": []}, "log": [f"[competitor] {name}"]}

    def judge(state):
        name = state["current"]["name"]
        decision = queue.pop(0) if queue else "보류"
        total = 75.0 if decision == "투자" else 60.0
        return {"decision": decision, "total_score": total,
                "history": [{"startup": name, "total_score": total, "decision": decision}],
                "log": [f"[judge] {name}: {decision} ({total}점)"]}

    def report(state):
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "report_stub.md"
        lines = [f"# 흐름 검증용 보고서 ({'투자 추천' if state.get('decision') == '투자' else '전체 보류'})", ""]
        lines += [f"- {h['startup']}: {h['decision']} ({h['total_score']}점)" for h in state.get("history", [])]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"report_path": str(path), "log": [f"[report] {path.name} 저장"]}

    return {"discover": discover, "select": select, "tech_summary": tech_summary,
            "market_eval": market_eval, "competitor": competitor, "judge": judge, "report": report}
