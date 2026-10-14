"""보고서 PDF 조판 (팀 보고서 디자인을 따름).

보고서 원문은 report_writer가 만든 Markdown 하나이고, 이 모듈은 그것을 읽어 A4에 직접 그린다.
  제목 영역(제목 · SKALA 로고 · 작성일/판정) → 검은 SUMMARY 상자(빨간 왼쪽 띠) → 점수 막대
  → 장 제목(구분선 + 큰 글씨) → 본문 · 목록 · 표(진한 머리행, 줄무늬) → REFERENCE → 쪽 번호
Markdown은 보고서 작성기가 정해진 모양으로만 만들기 때문에, 아래 몇 가지 형식만 읽으면 된다.
  # 제목 / ## 장 / ### 소제목 / **굵은 한 줄** = 소제목 / | 표 | / - 목록 / 1. 번호 목록 / _안내문_
"""
import re
import unicodedata
from functools import lru_cache

from fpdf import FPDF

from config import ROOT

ASSET_DIR = ROOT / "assets"
FONT_DIR = ASSET_DIR / "fonts"

# 팀 보고서와 같은 색
BACKGROUND = (245, 245, 245)
MAIN = (38, 38, 43)
INK = (26, 26, 26)
RED = (230, 43, 30)
MUTED = (98, 98, 104)
LINE = (218, 218, 221)
WHITE = (255, 255, 255)
MARGIN, WIDTH = 15, 180

# SUIT 폰트에 없는 글자 → 비슷한 글자 (없으면 PDF에서 글자가 빠진다)
GLYPH_FALLBACK = {"—": "-", "–": "-", "•": "-", "Ø": "O", "ø": "o", "美": "미", "中": "중", "日": "일",
                  "韓": "한", "英": "영", "獨": "독", "佛": "불", " ": " ", "→": ">"}


@lru_cache(maxsize=1)
def _font_chars() -> frozenset:
    from fontTools.ttLib import TTFont  # fpdf2 의존성으로 함께 설치됨
    return frozenset(TTFont(str(FONT_DIR / "SUIT-Regular.ttf")).getBestCmap())


def fit_font(text: str) -> str:
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


def visual_len(text: str) -> int:
    """한글·한자는 영문보다 약 두 배 넓으므로 두 칸으로 센다."""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _plain(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text).replace("\\*", "*").strip("_ ")


# ── Markdown → 블록 목록 ──────────────────────────────────
def parse(md: str) -> dict:
    lines = md.splitlines()
    doc = {"title": "", "meta": "", "summary": [], "blocks": []}
    i, section = 0, None
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            doc["title"] = line[2:]
        elif line.startswith("## "):
            section = line[3:]
            if section != "SUMMARY":
                doc["blocks"].append(("chapter", section))
        elif section is None:
            doc["meta"] = line
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"-+", c) for c in cells):
                    rows.append(cells)
                i += 1
            doc["blocks"].append(("table", rows))
            continue
        else:
            if line.startswith("### "):
                block = ("subheading", line[4:])
            elif re.fullmatch(r"\*\*[^*]+\*\*", line):
                block = ("subheading", line[2:-2])
            elif line.startswith("- "):
                block = ("bullet", line[2:])
            elif re.match(r"\d+\. ", line):
                number, text = line.split(". ", 1)
                block = ("numbered", (number, text))
            elif line.startswith(("_", "\\*")):
                block = ("note", _plain(line))
            else:
                block = ("paragraph", line)
            (doc["summary"] if section == "SUMMARY" else doc["blocks"]).append(block)
        i += 1
    return doc


# ── 조판 ──────────────────────────────────────────────────
class _ReportPDF(FPDF):
    def __init__(self, body: float):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(True, margin=MARGIN + 4)
        self.c_margin = 0
        self.body = body
        self.small = body - 1.5
        for style, weight in (("", "Regular"), ("B", "Bold")):
            self.add_font("SUIT", style, str(FONT_DIR / f"SUIT-{weight}.ttf"))
        self.add_font("SUIT-SemiBold", "", str(FONT_DIR / "SUIT-SemiBold.ttf"))
        self.alias_nb_pages()

    def footer(self):
        self.set_draw_color(*LINE)
        self.line(MARGIN, self.h - 12, self.w - MARGIN, self.h - 12)
        self.set_xy(MARGIN, self.h - 10)
        self.set_font("SUIT", size=8)
        self.set_text_color(*MUTED)
        self.cell(WIDTH, 4, f"ENERGY AI  /  {self.page_no()} / {{nb}}", align="R")

    # 기본 도구
    def measure(self, text: str, width: float, size: float, family: str = "SUIT") -> int:
        self.set_font(family, size=size)
        return len(self.multi_cell(width, size * 0.5, text, markdown="**" in text, dry_run=True, output="LINES"))

    def need(self, height: float):
        if self.y + height > self.page_break_trigger:
            self.add_page()

    def text(self, text: str, *, size=None, color=INK, x=MARGIN, width=WIDTH, family="SUIT", gap=1.5):
        size = size or self.body
        self.set_font(family, size=size)
        self.set_text_color(*color)
        self.set_x(x)
        # **굵게**가 있을 때만 markdown 해석 (URL 속 "__"가 밑줄 기호로 읽히지 않게)
        self.multi_cell(width, size * 0.5, text, markdown="**" in text, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_y(self.y + gap)

    # 블록
    def opening(self, title: str, meta: str):
        self.image(str(ASSET_DIR / "skala-logo.png"), x=self.w - MARGIN - 28, y=MARGIN + 1, w=28)
        self.set_xy(MARGIN, MARGIN)
        self.set_font("SUIT", "B", 21)
        self.set_text_color(*INK)
        self.multi_cell(WIDTH - 34, 10, title, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_y(self.y + 2)
        self.text(meta.replace("**", ""), size=9, color=MUTED, gap=4)

    def summary(self, items: list[tuple]):
        inner, size = WIDTH - 14, self.body
        line_h = size * 0.5
        heights = [self.measure(("- " if k == "bullet" else "") + t, inner, size) * line_h + 1 for k, t in items]
        height = 16 + sum(heights)
        y = self.y
        self.set_fill_color(*INK)
        self.rect(MARGIN, y, WIDTH, height, style="F")
        self.set_fill_color(*RED)
        self.rect(MARGIN, y, 1.2, height, style="F")
        self.set_xy(MARGIN + 7, y + 4)
        self.set_font("SUIT", "B", 11)
        self.set_text_color(*BACKGROUND)
        self.cell(inner, 6, "SUMMARY")
        self.set_y(y + 12)
        for kind, text in items:
            self.text(("- " if kind == "bullet" else "") + text, size=size, color=BACKGROUND,
                      x=MARGIN + 7, width=inner, gap=1)
        self.set_y(y + height + 5)

    def bars(self, title: str, values: list[tuple[str, float]], threshold: float, highlight: int | None):
        self.need(10 + 8 * len(values))
        self.text(title, size=10.5, family="SUIT-SemiBold", gap=1)
        x, width = MARGIN + 45, WIDTH - 75
        for index, (label, value) in enumerate(values):
            y = self.y
            self.set_font("SUIT", size=9)
            self.set_text_color(*INK)
            self.set_xy(MARGIN, y)
            self.cell(43, 6, label)
            self.set_fill_color(*LINE)
            self.rect(x, y + 1, width, 4, style="F")
            self.set_fill_color(*(RED if index == highlight else MAIN))
            self.rect(x, y + 1, width * min(value, 100) / 100, 4, style="F")
            self.set_draw_color(*INK)
            self.line(x + width * threshold / 100, y, x + width * threshold / 100, y + 6)
            self.set_xy(x + width + 3, y)
            self.set_font("SUIT-SemiBold", size=9)
            self.cell(27, 6, f"{value:.1f} / 100")
            self.set_y(y + 8)
        self.set_font("SUIT", size=8)
        self.set_text_color(*MUTED)
        self.set_x(x)
        self.cell(WIDTH - 45, 4, f"척도 0~100 · 투자 기준 {threshold:g}점 (세로선)")
        self.set_y(self.y + 8)

    def chapter(self, title: str):
        self.need(22)
        self.set_y(self.y + 2)
        self.set_draw_color(*LINE)
        self.line(MARGIN, self.y, self.w - MARGIN, self.y)
        self.set_y(self.y + 3)
        self.set_font("SUIT", "B", 15)
        self.set_text_color(*INK)
        self.multi_cell(WIDTH, 8, title, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_y(self.y + 1.5)

    def subheading(self, text: str):
        self.need(14)
        self.set_y(self.y + 1)
        self.text(text, size=self.body + 0.5, family="SUIT-SemiBold", gap=1)

    def table(self, rows: list[list[str]]):
        header, body = rows[0], rows[1:]
        size, line_h = self.small, self.small * 0.46
        widths = _column_widths(rows)
        score_col = next((i for i, h in enumerate(header) if h.startswith("점수")), None)

        def layout(values, family):
            self.set_font(family, size=size)
            cells = [self.multi_cell(w - 4, line_h, _plain(v), dry_run=True, output="LINES")
                     for v, w in zip(values, widths)]
            extra = 2.5 if score_col is not None and family == "SUIT" else 0  # 점수 막대 자리
            return cells, max(len(c) for c in cells) * line_h + 4 + extra

        def draw(cells, height, header_row=False, index=0, values=None):
            x, y = MARGIN, self.y
            self.set_fill_color(*(MAIN if header_row else (WHITE if index % 2 == 0 else BACKGROUND)))
            self.rect(x, y, WIDTH, height, style="F")
            family = "SUIT-SemiBold" if header_row else "SUIT"
            self.set_font(family, size=size)
            self.set_text_color(*(BACKGROUND if header_row else INK))
            for col, (lines, w) in enumerate(zip(cells, widths)):
                for k, text in enumerate(lines):
                    self.set_xy(x + 2, y + 2 + k * line_h)
                    self.cell(w - 4, line_h, text)
                if values is not None and col == score_col:  # 점수 칸 아래 작은 막대 (0~10)
                    score = float(re.match(r"[\d.]+", values[col]).group())
                    self.set_fill_color(*LINE)
                    self.rect(x + 2, y + height - 3, w - 6, 0.9, style="F")
                    self.set_fill_color(*MAIN)
                    self.rect(x + 2, y + height - 3, (w - 6) * score / 10, 0.9, style="F")
                x += w
            self.set_draw_color(*LINE)
            self.line(MARGIN, y + height, self.w - MARGIN, y + height)
            self.set_xy(MARGIN, y + height)

        head_cells, head_h = layout(header, "SUIT-SemiBold")
        prepared = [(row, *layout(row, "SUIT")) for row in body]
        self.need(head_h + (prepared[0][2] if prepared else 0))
        draw(head_cells, head_h, header_row=True)
        for index, (values, cells, height) in enumerate(prepared):
            if self.y + height > self.page_break_trigger:
                self.add_page()
                draw(head_cells, head_h, header_row=True)
            draw(cells, height, index=index, values=values)
        self.set_y(self.y + 3)


def _column_widths(rows: list[list[str]]) -> list[float]:
    """열 너비: 내용 길이에 비례하되, 가장 긴 단어가 중간에서 끊기지 않게."""
    cols = len(rows[0])
    weights = []
    for i in range(cols):
        texts = [_plain(r[i]) for r in rows if len(r) > i]
        longest = max(min(visual_len(t), 60) for t in texts)
        word = max((visual_len(w) for t in texts for w in t.split()), default=0)
        weights.append(max(longest, 9, min(word + 7, 24)))
    return [WIDTH * w / sum(weights) for w in weights]


def render(md_text: str, path, body: float = 9.5, bars: dict | None = None) -> int:
    """보고서 Markdown을 PDF로 그리고 쪽수를 돌려준다.

    bars: {"title", "values": [(이름, 점수)], "threshold", "highlight", "after": "summary" | "chapter2"}
    """
    doc = parse(fit_font(md_text))
    pdf = _ReportPDF(body)
    pdf.set_title(doc["title"])
    pdf.add_page()
    pdf.opening(doc["title"], doc["meta"])
    pdf.summary([(k, t) for k, t in doc["summary"]])
    draw_bars = (lambda: pdf.bars(bars["title"], bars["values"], bars["threshold"], bars.get("highlight"))) \
        if bars else None
    if draw_bars and bars.get("after") == "summary":
        draw_bars()
    chapter_no = 0
    for kind, value in doc["blocks"]:
        if kind == "chapter":
            pdf.chapter(value)
            chapter_no += 1
            if draw_bars and bars.get("after") == "chapter2" and chapter_no == 2:
                draw_bars()
        elif kind == "subheading":
            pdf.subheading(value)
        elif kind == "paragraph":
            pdf.text(value)
        elif kind == "bullet":
            pdf.text("- " + value, x=MARGIN + 2, width=WIDTH - 2, gap=0.8)
        elif kind == "numbered":
            number, text = value
            pdf.text(f"{number}. {text}", size=pdf.small, x=MARGIN + 2, width=WIDTH - 2, gap=0.8)
        elif kind == "note":
            pdf.text(value, size=pdf.small, color=MUTED)
        elif kind == "table":
            pdf.table(value)
    pdf.output(str(path))
    return pdf.page_no()
