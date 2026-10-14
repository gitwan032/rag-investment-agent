"""보고서 생성 에이전트.

판정에 따라 두 가지 목차 중 하나로 쓴다 (팀 설계 그대로, 5장 이내, 맨 앞 SUMMARY, 맨 끝 REFERENCE).
  투자 추천: SUMMARY → 1. 기업 및 사업 개요 → 2. 시장 및 경쟁 분석 → 3. 기업 역량 및 리스크 → 4. 종합 투자 평가 → REFERENCE
  전체 보류: SUMMARY → 1. 평가 대상 및 선정 과정 → 2. 후보별 평가 결과 비교 → 3. 주요 보류 사유 및 시장 환경 → 4. 분석의 한계 → REFERENCE

LLM은 본문 문장만 쓴다. 판정·총점이 들어가는 SUMMARY 첫 줄, 모든 표, 분석의 한계, REFERENCE는 코드가 만든다.
→ 판정과 다른 결론을 쓰거나, 표의 수치가 바뀌거나, 사용하지 않은 출처가 들어갈 수 없다.
"""
import re
import unicodedata
from datetime import datetime
from functools import lru_cache

import markdown
from fpdf import FPDF
from pydantic import BaseModel, Field

from config import INVEST_THRESHOLD, OUTPUT_DIR, PROMPT_DIR, ROOT
from schemas import SEGMENT_KO
from tools.citations import strip
from tools.llm import ask_structured, get_llm
from tools.references import format_reference, unique

FONT_DIR = ROOT / "assets" / "fonts"
MAX_PAGES = 5
REF_LABEL = {"report": "기관 보고서", "paper": "학술 논문", "web": "웹페이지"}


class InvestmentDraft(BaseModel):
    summary: str = Field(description="핵심 투자 포인트 3개, 핵심 리스크 2개 (350자 이내)")
    business: str
    market: str
    team_risks: str
    evaluation: str


class HoldDraft(BaseModel):
    summary: str = Field(description="공통 보류 사유와 다음 탐색 방향 (350자 이내)")
    selection: str
    comparison: str
    reasons_market: str


# ── 표 (코드가 직접 만든다) ─────────────────────────────────
def _cell(text, limit: int = 90) -> str:
    text = strip(str(text)).replace("|", "/").replace("\n", " ")
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return (cut.rsplit(" ", 1)[0] if " " in cut[limit // 2:] else cut).rstrip(" ,.·") + "…"


def _table(header: list[str], rows: list[list]) -> str:
    if not rows:
        return "_표시할 데이터가 없습니다 (공개 정보로 확인되지 않음)._\n"
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _figures_table(figures: list[dict]) -> str:
    return _table(["지표 (원문 대상)", "범위", "값", "기준 연도", "출처"],
                  [[f["metric"], f["scope"], f["value"], f["year"], f"{f['source_title'][:30]} p.{f['page']}"]
                   for f in figures])


def _score_table(scores: dict) -> str:
    rows = [[s["name"], s["weight"], f"{s['score']}{' *' if s['insufficient'] else ''}",
             f"{s['score'] * s['weight'] / 10:.1f}", _cell(s["rationale"], 70)] for s in scores.values()]
    return _table(["평가 항목", "가중치", "점수(0~10)", "가중 점수", "근거"], rows) + \
        "\n\\* 공개 정보 부족으로 중간값 5점을 적용한 항목\n"


def _references(refs: list[dict]) -> str:
    out = []
    for kind in ("report", "paper", "web"):
        group = [r for r in refs if r["type"] == kind]
        if group:
            out.append(f"**{REF_LABEL[kind]}**\n")
            out += [f"{i}. {format_reference(r)}" for i, r in enumerate(group, 1)]
            if kind == "web" and any(r.get("accessed") for r in group):
                out.append("\n_게시일을 확인할 수 없는 웹페이지는 조회일을 표기했다._")
            out.append("")
    return "\n".join(out) or "_사용한 출처 없음_"


# ── 경로별 본문 ─────────────────────────────────────────────
def _investment(state: dict) -> tuple[str, list[dict]]:
    c, tech, market, comp = state["current"], state["tech_summary"], state["market_analysis"], state["competitor_analysis"]
    total, name = state["total_score"], c["name"]
    data = {"기업": {k: c.get(k) for k in ("name", "country", "segment", "stage", "funding")},
            "기술 요약": tech, "시장성": market, "경쟁사": comp,
            "점수": {s["name"]: {"점수": s["score"], "근거": s["rationale"]} for s in state["scores"].values()}}
    prompt = (PROMPT_DIR / "report_investment.md").read_text(encoding="utf-8").format(
        company=name, decision="투자 추천", total=f"{total:.1f}", threshold=INVEST_THRESHOLD, data=_dump(data))
    d = ask_structured(get_llm(), InvestmentDraft, prompt)

    limits = _limitations(state, [state["history"][-1]])
    body = f"""# {name} 투자 평가 보고서

작성일 {state['run_date']} · 분야 Energy / {SEGMENT_KO[c['segment']]} · 판정 **투자 추천 ({total:.1f}점)**

## SUMMARY

**결론: 투자 추천.** 총점 {total:.1f}점으로 투자 기준({INVEST_THRESHOLD}점)을 넘었고 필수 탈락 조건에 해당하지 않는다.

{strip(d.summary)}

## 1. 기업 및 사업 개요

{strip(d.business)}

## 2. 시장 및 경쟁 분석

{strip(d.market)}

### 시장 수치 (원문 인용으로 확인된 수치만 표기)

{_figures_table(market.get('figures', []))}
### 주요 경쟁사 비교

{_table(['기업', '국가', '제품·기술', f'{name} 대비 차이'], [[x['name'], x['country'], x['offering'], x['difference']] for x in comp.get('competitors', [])])}
## 3. 기업 역량 및 리스크

{strip(d.team_risks)}

## 4. 종합 투자 평가

{_score_table(state['scores'])}
**총점 {total:.1f}점 / 100점 (투자 기준 {INVEST_THRESHOLD}점) → 투자 추천**

{_held_before(state)}{strip(d.evaluation)}

**분석의 한계**

{limits}
"""
    refs = unique([r for r in state.get("references", []) if r["startup"] == name])
    return body, refs


def _hold(state: dict) -> tuple[str, list[dict]]:
    history = state.get("history", [])
    data = [{k: h[k] for k in ("startup", "country", "segment", "stage", "total_score", "knockout", "weakest",
                                "insufficient", "profile")} for h in history]
    markets = list((state.get("market_cache") or {}).values())
    market_data = [{"segment": m["segment"], "growth": m["growth"], "policy_risks": m["policy_risks"]} for m in markets]
    prompt = (PROMPT_DIR / "report_hold.md").read_text(encoding="utf-8").format(
        count=len(history), threshold=INVEST_THRESHOLD, data=_dump({"후보": data, "시장": market_data}))
    d = ask_structured(get_llm(), HoldDraft, prompt)

    sources = sorted({line.split("회차 ")[1].split(":")[0] for line in state.get("log", []) if line.startswith("[discover]")})
    figures = [f for m in markets for f in m.get("figures", [])][:6]
    body = f"""# Energy AI 스타트업 투자 평가 보고서 (전체 보류)

작성일 {state['run_date']} · 분야 Energy · 판정 **투자 대상 없음 (후보 {len(history)}곳 전원 보류)**

## SUMMARY

**결론: 투자 대상 없음.** 평가한 후보 {len(history)}곳 모두 투자 기준({INVEST_THRESHOLD}점)에 미달하거나 필수 탈락 조건에 해당했다.

{strip(d.summary)}

## 1. 평가 대상 및 선정 과정

{strip(d.selection)}

후보 출처: {', '.join(sources) or '기록 없음'} · 탐색 {state.get('search_round', 0)}회

## 2. 후보별 평가 결과 비교

{_table(['기업', '국가', '분야', '단계', '총점', '취약 항목', '정보 부족'],
        [[h['startup'], h['country'], SEGMENT_KO[h['segment']], h['stage'], f"{h['total_score']:.1f}",
          ', '.join(h['weakest']), f"{len(h['insufficient'])}개"] for h in history])}
{strip(d.comparison)}

## 3. 주요 보류 사유 및 시장 환경

{strip(d.reasons_market)}

### 시장 수치 (원문 인용으로 확인된 수치만 표기)

{_figures_table(figures)}
## 4. 분석의 한계

{_limitations(state, history)}
"""
    names = {h["startup"] for h in history}
    refs, per_company = [], {}
    for r in unique(state.get("references", [])):  # 후보별 웹 출처는 4건까지 (5장 분량 제한)
        if r["startup"] not in names or r["used_by"] == "discovery":
            continue
        if r["type"] == "web":
            per_company[r["startup"]] = per_company.get(r["startup"], 0) + 1
            if per_company[r["startup"]] > 4:
                continue
        refs.append(r)
    return body, unique(refs)


def _held_before(state: dict) -> str:
    """투자 보고서에도 먼저 평가해 보류한 후보를 한 줄씩 남긴다 (어떤 후보를 거쳐 이 기업에 왔는지)."""
    held = [h for h in state.get("history", [])[:-1]]
    if not held:
        return ""
    lines = [f"- {h['startup']}: 보류, {h['key_reason']} · 취약 항목 {', '.join(h['weakest'])}" for h in held]
    return "**먼저 평가해 보류한 후보**\n\n" + "\n".join(lines) + "\n\n"


def _limitations(state: dict, records: list[dict]) -> str:
    lines = [f"- 분석 기준일 {state['run_date']}의 공개 자료(웹 검색, 시장·기술 보고서 6종)에 한정한 평가다."]
    for h in records:
        if h["insufficient"]:
            lines.append(f"- {h['startup']}: 공개 정보 부족으로 중간값 5점을 적용한 항목: {', '.join(h['insufficient'])}")
    if any("시드" in line for line in state.get("log", []) if line.startswith("[discover]")):
        lines.append("- 웹 검색으로 후보를 찾지 못한 회차는 사전에 정한 시드 후보 목록을 사용했다.")
    lines.append("- 비상장 기업의 매출·밸류에이션 등 재무 정보는 대부분 공개되지 않아 투자조건 평가에 한계가 있다.")
    return "\n".join(lines)


def _dump(data) -> str:
    import json

    def clean(v):
        if isinstance(v, str):
            return strip(v)
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items() if k not in ("evidence_count", "source")}
        return v
    return json.dumps(clean(data), ensure_ascii=False, indent=1)


# ── PDF ───────────────────────────────────────────────────
def _visual_len(text: str) -> int:
    """한글·한자는 영문보다 약 두 배 넓으므로 두 칸으로 센다."""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


# SUIT 폰트에 없는 글자 → 비슷한 글자로 바꾼다 (없으면 PDF에서 글자가 빠진다)
GLYPH_FALLBACK = {"—": "-", "–": "-", "•": "-", "Ø": "O", "ø": "o", "美": "미", "中": "중", "日": "일",
                  "韓": "한", "英": "영", "獨": "독", "佛": "불", "\u00a0": " "}


@lru_cache(maxsize=1)
def _font_chars() -> frozenset:
    from fontTools.ttLib import TTFont  # fpdf2 의존성으로 함께 설치됨
    return frozenset(TTFont(str(FONT_DIR / "SUIT-Regular.ttf")).getBestCmap())


def _fit_font(text: str) -> str:
    chars, out = _font_chars(), []
    for ch in text:
        if ord(ch) in chars or ch in "\n\t":
            out.append(ch)
        elif ch in GLYPH_FALLBACK:
            out.append(GLYPH_FALLBACK[ch])
        else:  # 악센트 글자는 기본 글자로 (é → e), 그래도 없으면 뺀다
            base = unicodedata.normalize("NFKD", ch)[0]
            out.append(base if ord(base) in chars else "")
    return "".join(out)


def _size_columns(html: str) -> str:
    """표마다 열 너비를 내용 길이에 비례해 정하고, 칸 안 글자는 왼쪽 정렬한다 (기본은 모든 열이 같은 너비)."""
    def fix(match):
        table = match.group(0)
        rows = re.findall(r"<tr>(.*?)</tr>", table, flags=re.S)
        cells = [re.findall(r"<t[hd](?:\s[^>]*)?>(.*?)</t[hd]>", r, flags=re.S) for r in rows]
        cols = len(cells[0])
        lengths = [max(min(_visual_len(re.sub(r"<[^>]+>", "", row[i])), 60) for row in cells if len(row) > i)
                   for i in range(cols)]
        # 열 너비는 가장 긴 단어보다 좁지 않게 ("GE Vernova"가 "Vernov/a"로 끊기지 않도록)
        words = [max((_visual_len(w) for row in cells if len(row) > i
                      for w in re.sub(r"<[^>]+>", "", row[i]).split()), default=0) for i in range(cols)]
        lengths = [max(n, 9, min(w + 7, 24)) for n, w in zip(lengths, words)]
        widths = [round(100 * n / sum(lengths)) for n in lengths]
        head = iter(widths)
        table = re.sub(r"<th(\s[^>]*)?>", lambda m: f'<th width="{next(head)}%"{m.group(1) or ""}>', table, count=cols)
        return re.sub(r"<td>", '<td align="left">', table)
    return re.sub(r"<table>.*?</table>", fix, html, flags=re.S)


def render_pdf(md_text: str, path, font_size: float = 9.5) -> int:
    """표 제목은 ### 로 쓴다. fpdf2는 표 바로 앞 문단이 굵은 글씨(**...**)면 표 전체를 굵게 그린다."""
    md_text = _fit_font(md_text)
    html = markdown.markdown(md_text, extensions=["tables"])
    html = re.sub(r"<li>\s*<p>(.*?)</p>\s*</li>", r"<li>\1</li>", html, flags=re.S)
    html = _size_columns(html)
    pdf = FPDF(format="A4")
    pdf.set_margins(16, 14, 16)
    pdf.set_auto_page_break(True, margin=14)
    for style, file in (("", "SUIT-Regular.ttf"), ("B", "SUIT-Bold.ttf"), ("I", "SUIT-Regular.ttf"), ("BI", "SUIT-Bold.ttf")):
        pdf.add_font("SUIT", style, str(FONT_DIR / file))
    pdf.add_page()
    pdf.set_font("SUIT", size=font_size)
    pdf.write_html(html, font_family="SUIT", ul_bullet_char="-", table_line_separators=True,
                   heading_sizes={"h1": font_size + 7, "h2": font_size + 3, "h3": font_size + 1})
    pdf.output(str(path))
    return pdf.page_no()


def run(state: dict) -> dict:
    invest = state.get("decision") == "투자"
    body, refs = _investment(state) if invest else _hold(state)
    text = body + "\n## REFERENCE\n\n" + _references(refs) + "\n"

    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path, pdf_path = OUTPUT_DIR / f"report_{stamp}.md", OUTPUT_DIR / f"report_{stamp}.pdf"
    md_path.write_text(text, encoding="utf-8")
    pages, size = 0, 9.5
    while size >= 8:  # 5장을 넘으면 글자 크기를 줄여 다시 조판
        pages = render_pdf(text, pdf_path, size)
        if pages <= MAX_PAGES:
            break
        size -= 0.5
    note = "" if pages <= MAX_PAGES else f" (경고: {MAX_PAGES}장 초과)"
    return {"report_path": str(pdf_path),
            "log": [f"[report] {'투자 추천' if invest else '전체 보류'} 보고서 {pages}장{note}, 출처 {len(refs)}건 → {pdf_path.name}"]}
