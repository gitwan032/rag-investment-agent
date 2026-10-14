"""LLM 호출 공통 함수. 모든 에이전트는 여기서 모델을 받아 쓴다.

    python -m tools.llm   # 두 모델이 실제로 호출되는지 점검
"""
import re
from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import JUDGE_MODEL, JUDGE_REASONING, LLM_MODEL, LLM_REASONING


def _make(model: str, effort: str, env_name: str) -> ChatOpenAI:
    if not model:
        raise RuntimeError(f".env의 {env_name}가 비어 있습니다.")
    kwargs = {"model": model, "reasoning_effort": effort, "max_retries": 2, "timeout": 180}
    if effort == "none":
        kwargs["temperature"] = 0  # 추론을 끈 경우에만 temperature 지정 가능
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """생성용: 요약·분석·보고서 문장."""
    return _make(LLM_MODEL, LLM_REASONING, "LLM_MODEL")


@lru_cache(maxsize=1)
def get_judge() -> ChatOpenAI:
    """판정용: 후보 검증, 문서 관련성 판정, 투자 채점."""
    return _make(JUDGE_MODEL, JUDGE_REASONING, "JUDGE_MODEL")


def judge_with(effort: str) -> ChatOpenAI:
    """추론 강도를 직접 지정한 판정 모델 (eval/consistency_eval.py에서 설정 비교에 사용)."""
    return _make(JUDGE_MODEL, effort, "JUDGE_MODEL")


# 한글·영문·숫자·문장부호 외에 다른 문자(힌디·아랍·키릴·태국 문자 등)가 섞이면 생성 오류로 본다
FOREIGN_SCRIPT = re.compile(r"[\u0400-\u04FF\u0590-\u06FF\u0900-\u0DFF\u0E00-\u0E7F]")


def ask_structured(llm: ChatOpenAI, schema: type[BaseModel], prompt: str, tries: int = 3) -> BaseModel:
    """Pydantic 스키마 모양으로 답을 받는다. 형식이 틀리거나 다른 나라 문자가 섞이면 다시 시도한다."""
    error = None
    for _ in range(tries):
        try:
            result = llm.with_structured_output(schema).invoke(prompt)
        except Exception as exc:  # 형식 오류·일시적 API 오류
            error = exc
            continue
        if not FOREIGN_SCRIPT.search(result.model_dump_json()):
            return result
        error = "한국어·영어 외 문자 섞임"
        prompt += "\n\n주의: 한국어와 영어(고유명사)만 사용하세요. 다른 언어의 문자를 섞지 마세요."
    raise RuntimeError(f"구조화 응답 실패 ({schema.__name__}): {error}")


if __name__ == "__main__":
    class Probe(BaseModel):
        answer: str

    for label, llm in (("LLM_MODEL", get_llm()), ("JUDGE_MODEL", get_judge())):
        text = llm.invoke("한 단어로만 답하세요: 맑은 날 하늘의 색은?").content
        result = ask_structured(llm, Probe, "answer 필드에 '확인'이라고 쓰세요.")
        print(f"{label} ({llm.model_name}): 일반 응답='{text}', 구조화 응답='{result.answer}'")
