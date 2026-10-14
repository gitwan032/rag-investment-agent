"""에이전트 사이에 주고받는 데이터 모양 (Pydantic).

LLM에게 이 모양대로 답하게 하고(구조화 출력), 코드는 이 모양을 믿고 읽는다.
"""
from typing import Literal

from pydantic import BaseModel, Field

Segment = Literal["demand_forecasting", "generation_vpp", "ess_operation", "grid_management"]
SEGMENT_KO = {
    "demand_forecasting": "전력 수요 예측",
    "generation_vpp": "발전 예측·VPP",
    "ess_operation": "배터리(ESS) 운영",
    "grid_management": "전력망 관리",
}


class Candidate(BaseModel):
    name: str = Field(description="회사 공식 이름 하나 (괄호 병기 금지, 해외 기업은 영문)")
    country: str = Field(description="본사 국가 ISO 코드 두 글자, 예: KR, US")
    segment: Segment
    stage: str = Field(description="최근 투자 단계, 예: Seed, Series A. 모르면 Unknown")
    funding: str = Field(description="투자 유치 내용 한 줄 (금액·투자사·시기). 모르면 Unknown")
    homepage: str = Field(description="공식 홈페이지 URL. 모르면 빈 문자열")
    evidence: list[int] = Field(description="이 정보를 확인한 검색 결과 번호들")


class Founder(BaseModel):
    name: str = Field(description="이름")
    background: str = Field(description="역할과 경력 한 줄, 끝에 근거 번호 [n]")


class TechSummary(BaseModel):
    """기술 요약 에이전트 출력. 모든 주장 끝에 근거 번호 [n]을 붙인다 (근거 없는 항목은 코드가 제거)."""
    core_tech: str = Field(description="핵심 기술 2~3문장 [n]")
    products: list[str] = Field(description="주요 제품·서비스, 항목마다 [n]")
    trl: int = Field(description="기술 성숙도 TRL 1~9")
    trl_rationale: str = Field(description="TRL 판단 근거. 업계 기준 문서와 회사 근거를 비교 [n]")
    performance: list[str] = Field(description="정량 성과·성능 지표 (숫자 포함), 항목마다 [n]")
    strengths: list[str] = Field(description="기술 강점, 항목마다 [n]")
    weaknesses: list[str] = Field(description="기술 약점·리스크, 항목마다 [n]")
    scalability: str = Field(description="확장성 (소프트웨어 중심인지, 해외 확장 가능성) [n]")
    founders: list[Founder] = Field(description="핵심 창업자·경영진")
    team_assessment: str = Field(description="팀 역량 평가 1~2문장 [n]")
    gaps: list[str] = Field(description="근거를 찾지 못한 정보 (예: 성능 수치 미공개)")


# ── 시장성 평가 ──────────────────────────────────────────
class MarketFigure(BaseModel):
    metric: str = Field(description="수치가 가리키는 대상을 원문 그대로 (예: '전 세계 재생에너지 총투자')")
    scope: str = Field(description="수치의 지역 범위 (예: 전 세계, 미국, 한국)")
    value: str = Field(description="숫자와 단위를 원문 표기대로 (예: 'U$5.2조', '80-160 GW')")
    year: str = Field(description="기준 연도 또는 전망 시점")
    quote: str = Field(description="수치가 나온 근거 문장을 원문 그대로 발췌 (번역·요약 금지)")
    source: int = Field(description="근거 번호")


class MarketAnalysis(BaseModel):
    figures: list[MarketFigure] = Field(description="시장 규모·전망 수치")
    growth: str = Field(description="성장성과 성장 요인 2~3문장 [n]")
    demand_drivers: list[str] = Field(description="수요 요인, 항목마다 [n]")
    customers: list[str] = Field(description="주요 수요처(고객군), 항목마다 [n]")
    policy_risks: list[str] = Field(description="정책·규제 환경과 리스크, 항목마다 [n]")
    summary: str = Field(description="시장성 종합 2~3문장 [n]")
    gaps: list[str] = Field(description="근거를 찾지 못한 정보")


# ── 경쟁사 비교 ──────────────────────────────────────────
class Competitor(BaseModel):
    name: str
    country: str = Field(description="본사 국가 ISO 두 글자, 모르면 Unknown")
    offering: str = Field(description="그 회사가 파는 제품·기술 한 줄 (40자 이내). 어느 사이트에 언급됐는지가 아니라 무엇을 하는 회사인지")
    difference: str = Field(description="대상 기업과의 차이 한 줄 (50자 이내), 끝에 근거 번호 [n]")


class CompetitorAnalysis(BaseModel):
    competitors: list[Competitor] = Field(description="경쟁사 3~5곳")
    differentiation: str = Field(description="대상 기업의 차별점 2~3문장 [n]")
    moat: list[str] = Field(description="진입장벽(특허, 데이터, 고객 락인 등), 항목마다 [n]")
    traction: list[str] = Field(description="사업 실적(고객, 계약, 파트너십, 매출), 항목마다 [n]")
    risks: list[str] = Field(description="경쟁 리스크, 항목마다 [n]")
    gaps: list[str] = Field(description="근거를 찾지 못한 정보")


# ── 투자 판단 ────────────────────────────────────────────
ItemId = Literal["team", "market_size", "market_demand", "traction", "moat", "trl",
                 "tech_value", "regulation", "scalability", "finance", "deal_terms"]


class ItemScore(BaseModel):
    item_id: ItemId
    score: int = Field(description="0~10점 (상 8~10, 중 4~7, 하 0~3)")
    evidence_found: bool = Field(description="분석 결과 안에 이 항목을 판단할 근거가 있으면 true")
    rationale: str = Field(description="점수 근거 한 문장 (60자 이내). 판단에 쓴 사실만 쓰고 입력 필드 이름은 쓰지 않음")


class JudgeOutput(BaseModel):
    items: list[ItemScore] = Field(description="11개 항목 각각 한 번씩")
