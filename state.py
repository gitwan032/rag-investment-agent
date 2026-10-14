"""그래프 전체가 공유하는 State(상태).

State는 에이전트 사이의 '계약서'다. 각 노드는 State를 읽고, 자기가 맡은 키만 dict로 돌려준다.
- 리듀서가 없는 키: 새 값으로 덮어씀 (후보가 바뀔 때마다 새로 채우는 분석 결과)
- 리듀서가 있는 키: 기존 값에 누적 (실행 전체 동안 쌓이는 기록)
"""
from operator import add
from typing import Annotated, Literal, TypedDict


def merge_dict(old: dict | None, new: dict | None) -> dict:
    """시장 분석 캐시용: 기존 dict에 새 항목을 합친다."""
    return {**(old or {}), **(new or {})}


class InvestState(TypedDict, total=False):
    # ── 실행 설정 ──
    domain: str                       # "Energy"
    run_date: str                     # 실행일 (보고서 작성일, 웹 출처 조회일)

    # ── 스타트업 탐색 / 후보 선택 ──
    candidates: list[dict]            # 평가 대기열 (유망도 순)
    search_round: int                 # 탐색 횟수
    evaluated: Annotated[list[str], add]  # 평가를 마친 기업 이름 (재탐색 시 중복 제외)
    current: dict | None              # 지금 평가 중인 기업

    # ── 기업별 분석 결과 (후보가 바뀌면 select가 비움) ──
    tech_summary: dict | None         # 기술 요약 에이전트
    market_analysis: dict | None      # 시장성 평가 에이전트
    competitor_analysis: dict | None  # 경쟁사 비교 에이전트
    scores: dict                      # 11개 항목 점수·근거
    total_score: float                # 가중 합산 점수 (코드가 계산)
    decision: Literal["투자", "보류"] | None

    # ── 실행 전체 누적 ──
    market_cache: Annotated[dict, merge_dict]    # 같은 세그먼트 시장 분석 재사용
    history: Annotated[list[dict], add]          # 후보별 평가 기록 (보고서 비교표용)
    references: Annotated[list[dict], add]       # 실제로 사용한 출처 (REFERENCE용)
    log: Annotated[list[str], add]               # 노드별 진행 기록

    # ── 결과 ──
    report_path: str | None
