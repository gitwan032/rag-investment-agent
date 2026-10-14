"""근거 번호 [n] 검증 (모든 분석 에이전트 공통).

LLM에게 근거마다 번호를 붙여 주고, 답의 모든 주장 끝에 번호를 달게 한다.
번호가 없거나 존재하지 않는 번호를 단 주장은 근거 없는 주장(환각 가능성)으로 보고 지운다.
"""
import re

CITE = re.compile(r"\[(\d+)\]")
NOT_FOUND = "근거 부족"


def cited(text: str, n: int) -> set[int]:
    """text에 달린 유효한 근거 번호 (1~n)."""
    return {int(x) for x in CITE.findall(text) if 1 <= int(x) <= n}


def keep_cited(items: list[str], n: int, used: set[int]) -> list[str]:
    """유효한 번호가 달린 항목만 남기고, 쓰인 번호를 used에 모은다."""
    kept = []
    for item in items:
        nums = cited(item, n)
        if nums:
            kept.append(item)
            used |= nums
    return kept


def check_text(text: str, n: int, used: set[int]) -> str:
    """한 문단짜리 필드: 번호가 있으면 그대로, 없으면 '근거 부족'으로 바꾼다."""
    nums = cited(text, n)
    if not nums:
        return NOT_FOUND
    used |= nums
    return text


def strip(text: str) -> str:
    """보고서에 옮길 때 내부 근거 번호를 지운다 (출처는 REFERENCE로 따로 표기)."""
    return re.sub(r"\s*(\[\d+\])+", "", text).strip()
