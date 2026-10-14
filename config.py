"""프로젝트 공통 설정: 폴더 경로, 환경변수, 실행 상한값.

다른 모든 모듈은 경로나 모델 이름을 직접 쓰지 않고 이 파일에서 가져온다.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ── 폴더 경로 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"                # RAG 원본 PDF
EVAL_DIR = DATA_DIR / "eval"                # Retriever 평가용 질문-정답 세트
VECTORSTORE_DIR = DATA_DIR / "vectorstore"  # Chroma 저장 위치
CONFIG_DIR = ROOT / "config"
PROMPT_DIR = ROOT / "prompts"
OUTPUT_DIR = ROOT / "outputs"

# ── 환경변수 (.env) ─────────────────────────────────────
load_dotenv(ROOT / ".env")

LLM_MODEL = os.getenv("LLM_MODEL", "")      # 분석·보고서 생성용
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")  # 관련성 판정·투자 채점용

# GPT-5.x는 추론 모델이라 temperature를 대부분 받지 않는다. 추론 강도가 "none"일 때만 temperature=0 사용.
LLM_REASONING = os.getenv("LLM_REASONING", "none")   # 생성: 빠르고 결과가 덜 흔들리게
JUDGE_REASONING = os.getenv("JUDGE_REASONING", "none")  # 판정·채점: 팀 설계대로 temperature 0 (EXPERIMENTS 6번)

# ── RAG 모델 (오픈소스) ─────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ── 실행 상한 (무한 루프·비용 방지) ─────────────────────
MAX_DISCOVERY_ROUNDS = 3      # 스타트업 탐색 최대 횟수
MAX_CANDIDATES_PER_ROUND = 5  # 탐색 1회당 후보 수
MAX_EVALUATIONS = 8           # 심층 평가할 최대 기업 수
RAG_MAX_REWRITES = 2          # Agentic RAG 질의 재작성 최대 횟수
INVEST_THRESHOLD = 70         # 투자 판정 기준 점수
RECURSION_LIMIT = 100         # LangGraph 전체 단계 수 안전장치

REQUIRED_ENV = ("OPENAI_API_KEY", "TAVILY_API_KEY", "LLM_MODEL", "JUDGE_MODEL")


def missing_env() -> list[str]:
    """값이 비어 있는 필수 환경변수 이름 목록을 돌려준다."""
    return [name for name in REQUIRED_ENV if not os.getenv(name)]


if __name__ == "__main__":
    # python config.py → 설정 점검 (키 값 자체는 출력하지 않는다)
    print(f"프로젝트 루트: {ROOT}")
    for name in REQUIRED_ENV:
        print(f"  {name:<15} {'설정됨' if os.getenv(name) else '비어 있음'}")
    pdfs = sorted(DOCS_DIR.rglob("*.pdf"))
    print(f"RAG 문서: {len(pdfs)}개")
    missing = missing_env()
    print("점검 결과:", "OK" if not missing else f"누락 {missing}")
