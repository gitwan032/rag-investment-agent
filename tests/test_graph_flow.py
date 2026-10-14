"""그래프 흐름 테스트 (가짜 노드 사용, LLM·웹검색 없음).  실행: pytest -q"""
from agents import stubs
from app import initial_state
from config import MAX_DISCOVERY_ROUNDS, RECURSION_LIMIT
from graph.builder import build_graph


def run(decisions):
    graph = build_graph(nodes=stubs.nodes(decisions))
    return graph.invoke(initial_state(), config={"recursion_limit": RECURSION_LIMIT})


def test_invest_stops_at_first_investment():
    """두 번째 후보가 '투자'면 거기서 멈추고 보고서로 간다."""
    final = run(["보류", "투자"])
    assert [h["decision"] for h in final["history"]] == ["보류", "투자"]
    assert final["decision"] == "투자"
    assert final["report_path"].endswith("report_stub.md")


def test_all_hold_rediscovers_then_reports():
    """모두 보류면 대기열 소진 → 재탐색을 반복하다가 탐색 상한에서 보고서로 종료한다."""
    final = run([])
    assert final["search_round"] == MAX_DISCOVERY_ROUNDS
    assert all(h["decision"] == "보류" for h in final["history"])
    assert len(final["history"]) == 2 * MAX_DISCOVERY_ROUNDS  # 회차당 2곳


def test_parallel_join_runs_competitor_once_per_candidate():
    """기술·시장 두 분석이 모두 끝난 뒤 경쟁사 비교가 후보당 정확히 한 번 실행된다."""
    final = run(["보류", "투자"])
    competitor_logs = [line for line in final["log"] if line.startswith("[competitor]")]
    assert len(competitor_logs) == 2


def test_references_accumulate_from_parallel_nodes():
    """병렬 노드가 같은 references 키에 써도 리듀서(add)로 모두 누적된다."""
    final = run(["투자"])
    ids = {r["id"] for r in final["references"]}
    assert ids == {"tech:가상기업1-1", "market:가상기업1-1"}
