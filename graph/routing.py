"""조건 분기 함수. 판단은 State 값만 보고 한다 (LLM을 쓰지 않음)."""
from config import MAX_DISCOVERY_ROUNDS, MAX_EVALUATIONS


def next_from_queue(state) -> str:
    """대기열 상태에 따라 다음 노드를 고른다.

    1) 평가 기업 수가 상한에 도달 → report (비용·시간 안전장치)
    2) 대기열에 후보가 남아 있음   → select (다음 후보 평가)
    3) 대기열이 비었고 탐색 여유 있음 → discover (재탐색)
    4) 그 외                       → report (모두 보류로 종료)
    """
    if len(state.get("evaluated", [])) >= MAX_EVALUATIONS:
        return "report"
    if state.get("candidates"):
        return "select"
    if state.get("search_round", 0) < MAX_DISCOVERY_ROUNDS:
        return "discover"
    return "report"


def after_judge(state) -> str:
    """투자면 바로 보고서, 보류면 대기열 규칙을 따른다."""
    return "report" if state.get("decision") == "투자" else next_from_queue(state)
