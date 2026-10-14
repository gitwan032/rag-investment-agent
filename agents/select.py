"""후보 선택 노드 (LLM 사용 안 함).

대기열 맨 앞 후보를 현재 평가 대상으로 정하고, 이전 후보의 분석 결과를 비운다.
분석 결과를 비우지 않으면 다음 후보 평가에 이전 후보의 결과가 섞일 수 있다.
"""


def run(state: dict) -> dict:
    queue = list(state["candidates"])
    current = queue.pop(0)
    return {
        "candidates": queue,
        "current": current,
        "evaluated": [current["name"]],
        "tech_summary": None,
        "market_analysis": None,
        "competitor_analysis": None,
        "scores": {},
        "total_score": 0.0,
        "decision": None,
        "log": [f"[select] {current['name']} 평가 시작 (남은 후보 {len(queue)})"],
    }
