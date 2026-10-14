"""판정 일관성 측정: 입력을 고정하고 판정 모델을 N번 돌려 결과가 얼마나 흔들리는지 본다.

  ① 문서 관련성 판정 (Agentic RAG): 4개 분야 업계 기준선 질문의 첫 검색 결과 5개를 고정하고 N번 판정
  ② 투자 채점: 마지막 `python app.py` 실행에서 저장된 채점 입력(outputs/cache/judge_prompt_*.txt)을 N번 채점

비교 설정 (판정 모델은 같고 추론 방식만 다름)
  low  : reasoning_effort="low" (temperature 지정 불가, 기존 설정)
  none : reasoning_effort="none" + temperature=0 (팀 설계의 "판정 모델 temperature 0"과 같은 조건)

    python -m eval.consistency_eval --repeat 3

결과: outputs/consistency_eval.md (결정 근거: docs/EXPERIMENTS.md 6번)
"""
import argparse
from statistics import mean

from agents.investment_judge import CACHE_DIR, compute, score
from agents.tech_summary import industry_questions
from config import OUTPUT_DIR
from rag.agentic_rag import grade_docs
from rag.search import search_docs
from schemas import SEGMENT_KO
from tools.llm import judge_with

SETTINGS = ("low", "none")


def grading(repeat: int) -> list[str]:
    lines = ["## ① 문서 관련성 판정 (같은 문서 5개를 반복 판정)", "",
             "| 분야 | 설정 | 회차별 통과 문서 수 | 기준(2개) 충족 | 판정이 바뀐 문서 |",
             "|---|---|---|:---:|---:|"]
    summary = {s: {"flips": 0, "docs": 0} for s in SETTINGS}
    for segment, ko in SEGMENT_KO.items():
        question = industry_questions(segment)[0]
        docs = search_docs("tech", question, k=5)
        for setting in SETTINGS:
            judge = judge_with(setting)
            runs = [grade_docs(question, docs, judge)[:len(docs)] for _ in range(repeat)]
            passed = [sum(r) for r in runs]
            flips = sum(1 for i in range(len(docs)) if len({r[i] for r in runs if i < len(r)}) > 1)
            summary[setting]["flips"] += flips
            summary[setting]["docs"] += len(docs)
            enough = sum(p >= 2 for p in passed)
            lines.append(f"| {ko} | {setting} | {' / '.join(map(str, passed))} | {enough}/{repeat} | {flips}/{len(docs)} |")
            print(f"[grade] {ko} {setting}: {passed}, 바뀐 문서 {flips}", flush=True)
    lines += ["", "| 설정 | 판정이 한 번이라도 바뀐 문서 (전체) |", "|---|---:|"]
    lines += [f"| {s} | {summary[s]['flips']}/{summary[s]['docs']} |" for s in SETTINGS]
    return lines


def scoring(repeat: int) -> list[str]:
    prompts = sorted(CACHE_DIR.glob("judge_prompt_*.txt"))
    lines = ["", "## ② 투자 채점 (같은 분석 결과를 반복 채점)", ""]
    if not prompts:
        return lines + ["_채점 입력이 없습니다. 먼저 `python app.py`를 실행하세요._"]
    lines += ["| 기업 | 설정 | 회차별 총점 | 평균 | 범위(최고-최저) | 판정 | 점수가 바뀐 항목 |",
              "|---|---|---|---:|---:|---|---:|"]
    for path in prompts:
        company = path.stem.removeprefix("judge_prompt_")
        prompt = path.read_text(encoding="utf-8")
        for setting in SETTINGS:
            judge = judge_with(setting)
            runs = [score(prompt, judge) for _ in range(repeat)]
            results = [compute(s) for s in runs]
            totals = [r[0] for r in results]
            decisions = sorted({r[1] for r in results})
            changed = sum(1 for item in runs[0] if len({(s[item]["score"], s[item]["insufficient"]) for s in runs}) > 1)
            lines.append(f"| {company} | {setting} | {' / '.join(f'{t:.1f}' for t in totals)} | {mean(totals):.1f} "
                         f"| {max(totals) - min(totals):.1f} | {', '.join(decisions)} | {changed}/11 |")
            print(f"[score] {company} {setting}: {totals}", flush=True)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3, help="같은 입력을 몇 번 판정할지")
    args = parser.parse_args()
    lines = [f"# 판정 일관성 측정 (반복 {args.repeat}회)", ""] + grading(args.repeat) + scoring(args.repeat)
    (OUTPUT_DIR / "consistency_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUTPUT_DIR / 'consistency_eval.md'}")


if __name__ == "__main__":
    main()
