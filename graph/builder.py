"""LangGraph 그래프 조립.

    START → discover ─┬→ select ─┬→ tech_summary ─┐
                      │          └→ market_eval ──┴→ competitor → judge ─┬→ report → END
                      ├→ discover (재탐색)                                ├→ select (보류: 다음 후보)
                      └→ report (후보 없음)                               └→ discover (보류: 재탐색)
"""
from langgraph.graph import END, START, StateGraph

from graph.routing import after_judge, next_from_queue
from state import InvestState


def load_nodes(stub: bool) -> dict:
    """노드 이름 → 실행 함수. stub=True면 가짜 결과를 내는 함수로 흐름만 검증한다."""
    if stub:
        from agents import stubs

        return stubs.nodes()
    from agents import competitor, discovery, investment_judge, market_eval, report_writer, select, tech_summary

    return {
        "discover": discovery.run,
        "select": select.run,
        "tech_summary": tech_summary.run,
        "market_eval": market_eval.run,
        "competitor": competitor.run,
        "judge": investment_judge.run,
        "report": report_writer.run,
    }


def build_graph(stub: bool = False, nodes: dict | None = None):
    nodes = nodes or load_nodes(stub)
    graph = StateGraph(InvestState)
    for name, func in nodes.items():
        graph.add_node(name, func)

    graph.add_edge(START, "discover")
    graph.add_conditional_edges("discover", next_from_queue,
                                {"select": "select", "discover": "discover", "report": "report"})

    # 병렬(fan-out): 기술 요약과 시장성 평가는 서로 독립적이라 동시에 실행
    graph.add_edge("select", "tech_summary")
    graph.add_edge("select", "market_eval")
    # 합류(fan-in): 두 분석이 모두 끝나야 경쟁사 비교 실행
    graph.add_edge(["tech_summary", "market_eval"], "competitor")

    graph.add_edge("competitor", "judge")
    graph.add_conditional_edges("judge", after_judge,
                                {"report": "report", "select": "select", "discover": "discover"})
    graph.add_edge("report", END)
    return graph.compile()
