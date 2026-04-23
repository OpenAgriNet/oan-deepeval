"""
report_pdf.py — rich PDF export for OAN evaluation reports.

Produces a fully structured, colour-coded PDF with:
  • Cover page with overall KPIs
  • Executive summary bar charts (drawn with ReportLab primitives)
  • By-category table with colour-coded pass rates
  • By-section table
  • By-metric table with avg scores and failure reasons
  • By-language table
  • Failed-case detail pages (one compact block per case with metric rows)

Requires: reportlab
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Palette ────────────────────────────────────────────────────────────────
C_BRAND      = colors.HexColor("#1A237E")   # deep navy — headings
C_BRAND_LIGHT= colors.HexColor("#E8EAF6")   # lavender tint — header rows
C_ACCENT     = colors.HexColor("#0288D1")   # sky blue — section rules
C_PASS       = colors.HexColor("#2E7D32")   # dark green
C_FAIL       = colors.HexColor("#C62828")   # dark red
C_WARN       = colors.HexColor("#E65100")   # amber
C_PASS_BG    = colors.HexColor("#E8F5E9")
C_FAIL_BG    = colors.HexColor("#FFEBEE")
C_WARN_BG    = colors.HexColor("#FFF3E0")
C_ROW_ALT    = colors.HexColor("#F5F5F5")
C_BORDER     = colors.HexColor("#BDBDBD")
C_WHITE      = colors.white
C_BLACK      = colors.HexColor("#212121")

W, H = A4
MARGIN = 18 * mm


# ── Helpers ────────────────────────────────────────────────────────────────

def _pct(val: float | None) -> str:
    return f"{(val or 0.0) * 100:.1f}%"


def _pass_color(rate: float) -> colors.Color:
    if rate >= 0.80:
        return C_PASS
    if rate >= 0.60:
        return C_WARN
    return C_FAIL


def _pass_bg(rate: float) -> colors.Color:
    if rate >= 0.80:
        return C_PASS_BG
    if rate >= 0.60:
        return C_WARN_BG
    return C_FAIL_BG


def _score_color(score: float, threshold: float) -> colors.Color:
    return C_PASS if score >= threshold else C_FAIL


def _bar(rate: float, width: int = 80) -> str:
    """ASCII-style progress bar text (rendered inside a Paragraph)."""
    filled = int(rate * width / 100 * 10)  # proportional
    # We'll draw a real bar below; this is a fallback text indicator
    return _pct(rate)


# ── Style factory ──────────────────────────────────────────────────────────

def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    def S(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        "cover_title": S("cover_title", "Title",
                         fontSize=28, textColor=C_BRAND, spaceAfter=4,
                         fontName="Helvetica-Bold"),
        "cover_sub":   S("cover_sub", "Normal",
                         fontSize=13, textColor=C_ACCENT, spaceAfter=2,
                         fontName="Helvetica"),
        "cover_meta":  S("cover_meta", "Normal",
                         fontSize=9, textColor=colors.HexColor("#757575")),
        "h1":          S("h1", "Heading1",
                         fontSize=14, textColor=C_BRAND, spaceBefore=12,
                         spaceAfter=4, fontName="Helvetica-Bold"),
        "h2":          S("h2", "Heading2",
                         fontSize=11, textColor=C_ACCENT, spaceBefore=8,
                         spaceAfter=2, fontName="Helvetica-Bold"),
        "body":        S("body", "Normal",
                         fontSize=8.5, textColor=C_BLACK, leading=12),
        "body_bold":   S("body_bold", "Normal",
                         fontSize=8.5, textColor=C_BLACK, fontName="Helvetica-Bold"),
        "small":       S("small", "Normal",
                         fontSize=7.5, textColor=colors.HexColor("#616161")),
        "cell":        S("cell", "Normal",
                         fontSize=8, textColor=C_BLACK, leading=10),
        "cell_bold":   S("cell_bold", "Normal",
                         fontSize=8, textColor=C_BLACK, fontName="Helvetica-Bold",
                         leading=10),
        "pass_cell":   S("pass_cell", "Normal",
                         fontSize=8, textColor=C_PASS, fontName="Helvetica-Bold"),
        "fail_cell":   S("fail_cell", "Normal",
                         fontSize=8, textColor=C_FAIL, fontName="Helvetica-Bold"),
        "warn_cell":   S("warn_cell", "Normal",
                         fontSize=8, textColor=C_WARN, fontName="Helvetica-Bold"),
        "reason":      S("reason", "Normal",
                         fontSize=7.5, textColor=colors.HexColor("#424242"),
                         leading=10, leftIndent=4),
        "kpi_val":     S("kpi_val", "Normal",
                         fontSize=22, textColor=C_BRAND, fontName="Helvetica-Bold",
                         alignment=1),  # centred
        "kpi_label":   S("kpi_label", "Normal",
                         fontSize=8, textColor=colors.HexColor("#757575"),
                         alignment=1),
    }


# ── Generic table style builder ────────────────────────────────────────────

def _base_table_style(
    header_bg: colors.Color = C_BRAND_LIGHT,
    stripe: bool = True,
    n_rows: int = 0,
) -> list:
    cmds = [
        # Header
        ("BACKGROUND",   (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",    (0, 0), (-1, 0), C_BRAND),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 8.5),
        ("BOTTOMPADDING",(0, 0), (-1, 0), 5),
        ("TOPPADDING",   (0, 0), (-1, 0), 5),
        # All cells
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("TOPPADDING",   (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 1), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID",         (0, 0), (-1, -1), 0.4, C_BORDER),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]
    if stripe and n_rows > 0:
        for i in range(2, n_rows + 1, 2):
            cmds.append(("BACKGROUND", (0, i), (-1, i), C_ROW_ALT))
    return cmds


# ── KPI card row ───────────────────────────────────────────────────────────

def _kpi_table(meta: dict, st: dict) -> Table:
    total   = meta.get("total_cases", 0)
    passed  = meta.get("passed_cases", 0)
    failed  = meta.get("failed_cases", 0)
    errors  = meta.get("error_cases", 0)
    opr     = meta.get("overall_pass_rate", 0.0) or 0.0

    def card(val: str, label: str, color: colors.Color = C_BRAND) -> list:
        return [
            Paragraph(f'<font color="#{color.hexval()[2:]}">{val}</font>', st["kpi_val"]),
            Paragraph(label, st["kpi_label"]),
        ]

    rate_color = _pass_color(opr)
    data = [[
        card(_pct(opr), "Overall Pass Rate", rate_color),
        card(str(total), "Total Cases"),
        card(str(passed), "Passed", C_PASS),
        card(str(failed), "Failed", C_FAIL),
        card(str(errors), "Errors", C_WARN),
    ]]

    col_w = (W - 2 * MARGIN) / 5
    t = Table(data, colWidths=[col_w] * 5, rowHeights=[52])
    t.setStyle(TableStyle([
        ("BOX",          (0, 0), (-1, -1), 0.6, C_BORDER),
        ("INNERGRID",    (0, 0), (-1, -1), 0.4, C_BORDER),
        ("BACKGROUND",   (0, 0), (0, 0), _pass_bg(opr)),
        ("BACKGROUND",   (1, 0), (1, 0), C_BRAND_LIGHT),
        ("BACKGROUND",   (2, 0), (2, 0), C_PASS_BG),
        ("BACKGROUND",   (3, 0), (3, 0), C_FAIL_BG),
        ("BACKGROUND",   (4, 0), (4, 0), C_WARN_BG),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",        (0, 0), (-1, -1), "CENTRE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# ── Inline pass-rate bar (drawn using nested Table) ────────────────────────

def _rate_bar_cell(rate: float, cell_width: float = 80) -> Table:
    """Returns a tiny 1-row table that acts as a horizontal progress bar."""
    filled = max(1, int(rate * cell_width))
    empty  = cell_width - filled
    bar_data = [[Paragraph("", ParagraphStyle("x"))]]
    bar_cols = [filled, empty] if empty > 0 else [cell_width]
    bar = Table([["", ""]] if empty > 0 else [[""]], colWidths=bar_cols, rowHeights=[8])
    cmds = [
        ("BACKGROUND",   (0, 0), (0, 0), _pass_color(rate)),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]
    if empty > 0:
        cmds.append(("BACKGROUND", (1, 0), (1, 0), C_ROW_ALT))
    bar.setStyle(TableStyle(cmds))
    return bar


# ── By-category table ──────────────────────────────────────────────────────

def _category_table(by_cat: list, st: dict) -> Table:
    headers = ["Category", "Total", "Passed", "Failed", "Errors", "Pass Rate", "Trend"]
    rows = [headers]
    bar_w = 70

    for item in sorted(by_cat, key=lambda x: x.get("pass_rate", 0)):
        rate  = item.get("pass_rate", 0.0) or 0.0
        fails = item.get("failed", 0) + item.get("error", 0)
        rate_color = _pass_color(rate)
        style_key  = "pass_cell" if rate >= 0.8 else ("warn_cell" if rate >= 0.6 else "fail_cell")
        rows.append([
            Paragraph(item.get("category", ""), st["cell_bold"]),
            Paragraph(str(item.get("total", 0)), st["cell"]),
            Paragraph(str(item.get("passed", 0)), st["pass_cell"]),
            Paragraph(str(fails), st["fail_cell"] if fails else st["cell"]),
            Paragraph(str(item.get("error", 0)), st["warn_cell"] if item.get("error") else st["cell"]),
            Paragraph(_pct(rate), st[style_key]),
            _rate_bar_cell(rate, bar_w),
        ])

    n = len(rows)
    avail = W - 2 * MARGIN
    col_w = [avail * r for r in [0.26, 0.07, 0.08, 0.08, 0.08, 0.10, bar_w / (avail / mm) * mm]]
    # Recompute: give bar a fixed mm width
    bar_mm = bar_w * 0.9
    rest   = avail - bar_mm
    col_w  = [rest * r for r in [0.30, 0.09, 0.10, 0.10, 0.10, 0.13]] + [bar_mm]

    t = Table(rows, colWidths=col_w, repeatRows=1)
    style_cmds = _base_table_style(n_rows=n - 1)
    # Colour pass-rate column background per row
    for i, item in enumerate(sorted(by_cat, key=lambda x: x.get("pass_rate", 0)), start=1):
        rate = item.get("pass_rate", 0.0) or 0.0
        style_cmds.append(("BACKGROUND", (5, i), (6, i), _pass_bg(rate)))
    t.setStyle(TableStyle(style_cmds))
    return t


# ── By-metric table ────────────────────────────────────────────────────────

def _metric_table(by_metric: list, st: dict) -> Table:
    headers = ["Metric", "Total", "Passed", "Failed", "Avg Score", "Pass Rate", "Top Failure Reason"]
    rows = [headers]

    for item in sorted(by_metric, key=lambda x: x.get("pass_rate", 0)):
        rate    = float(item.get("pass_rate") or 0.0)
        score   = float(item.get("avg_score") or 0.0)
        reasons = item.get("common_failure_reasons", [])
        top_r   = reasons[0][:100] + "…" if reasons and len(reasons[0]) > 100 else (reasons[0] if reasons else "—")
        style_key = "pass_cell" if rate >= 0.8 else ("warn_cell" if rate >= 0.6 else "fail_cell")
        rows.append([
            Paragraph(item.get("metric", ""), st["cell_bold"]),
            Paragraph(str(item.get("total", 0)), st["cell"]),
            Paragraph(str(item.get("passed", 0)), st["pass_cell"]),
            Paragraph(str(item.get("failed", 0)), st["fail_cell"] if item.get("failed") else st["cell"]),
            Paragraph(f"{score:.3f}", st["cell"]),
            Paragraph(_pct(rate), st[style_key]),
            Paragraph(top_r, st["small"]),
        ])

    n = len(rows)
    avail = W - 2 * MARGIN
    col_w = [avail * r for r in [0.20, 0.07, 0.09, 0.08, 0.10, 0.11, 0.35]]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle(_base_table_style(n_rows=n - 1)))
    return t


# ── By-section table ───────────────────────────────────────────────────────

def _section_table(by_sec: list, st: dict) -> Table:
    headers = ["Section", "Total", "Passed", "Failed", "Errors", "Pass Rate"]
    rows = [headers]
    for item in by_sec:
        rate = item.get("pass_rate", 0.0) or 0.0
        style_key = "pass_cell" if rate >= 0.8 else ("warn_cell" if rate >= 0.6 else "fail_cell")
        rows.append([
            Paragraph(item.get("section", ""), st["cell_bold"]),
            Paragraph(str(item.get("total", 0)), st["cell"]),
            Paragraph(str(item.get("passed", 0)), st["pass_cell"]),
            Paragraph(str(item.get("failed", 0)), st["fail_cell"] if item.get("failed") else st["cell"]),
            Paragraph(str(item.get("error", 0)), st["warn_cell"] if item.get("error") else st["cell"]),
            Paragraph(_pct(rate), st[style_key]),
        ])

    n = len(rows)
    avail = W - 2 * MARGIN
    col_w = [avail * r for r in [0.35, 0.10, 0.12, 0.12, 0.12, 0.19]]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle(_base_table_style(n_rows=n - 1)))
    return t


# ── By-language table ──────────────────────────────────────────────────────

def _language_table(by_lang: list, st: dict) -> Table:
    headers = ["Language", "Code", "Total", "Passed", "Failed", "Errors", "Pass Rate"]
    rows = [headers]
    for item in by_lang:
        rate = item.get("pass_rate", 0.0) or 0.0
        style_key = "pass_cell" if rate >= 0.8 else ("warn_cell" if rate >= 0.6 else "fail_cell")
        rows.append([
            Paragraph(item.get("language_label", ""), st["cell_bold"]),
            Paragraph(item.get("language", ""), st["cell"]),
            Paragraph(str(item.get("total", 0)), st["cell"]),
            Paragraph(str(item.get("passed", 0)), st["pass_cell"]),
            Paragraph(str(item.get("failed", 0)), st["fail_cell"] if item.get("failed") else st["cell"]),
            Paragraph(str(item.get("error", 0)), st["warn_cell"] if item.get("error") else st["cell"]),
            Paragraph(_pct(rate), st[style_key]),
        ])

    n = len(rows)
    avail = W - 2 * MARGIN
    col_w = [avail * r for r in [0.22, 0.09, 0.10, 0.12, 0.12, 0.12, 0.23]]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle(_base_table_style(n_rows=n - 1)))
    return t


# ── Top failing categories summary ────────────────────────────────────────

def _top_failing_table(items: list, st: dict) -> Table:
    headers = ["Rank", "Category", "Failed", "Total", "Failure Rate"]
    rows = [headers]
    for i, item in enumerate(items, 1):
        fails = item.get("failed", 0) + item.get("error", 0)
        if fails == 0:
            continue
        rate = item.get("failure_rate", 0.0) or 0.0
        rows.append([
            Paragraph(f"#{i}", st["cell_bold"]),
            Paragraph(item.get("category", ""), st["cell_bold"]),
            Paragraph(str(fails), st["fail_cell"]),
            Paragraph(str(item.get("total", 0)), st["cell"]),
            Paragraph(_pct(rate), st["fail_cell"] if rate >= 0.3 else st["warn_cell"]),
        ])

    if len(rows) == 1:
        return Table([["No failures recorded"]], colWidths=[W - 2 * MARGIN])

    avail = W - 2 * MARGIN
    col_w = [avail * r for r in [0.08, 0.42, 0.15, 0.15, 0.20]]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    cmds = _base_table_style(n_rows=len(rows) - 1)
    # Highlight failure rate cells
    for i in range(1, len(rows)):
        rate = items[i - 1].get("failure_rate", 0.0) or 0.0
        cmds.append(("BACKGROUND", (4, i), (4, i), _pass_bg(1 - rate)))
    t.setStyle(TableStyle(cmds))
    return t


# ── Failed case detail block ───────────────────────────────────────────────

def _failed_case_block(case: dict, st: dict, idx: int) -> list:
    """Returns a list of flowables for one failed case."""
    rate = case.get("pass_rate", 0.0) or 0.0
    name = case.get("name", "unknown")
    cat  = case.get("category", "")
    lang = case.get("language_label", case.get("language", ""))
    sec  = case.get("section", "")
    status = case.get("status", "FAIL")
    is_decline = case.get("is_decline", False)
    question   = (case.get("question") or "")[:300]
    expected   = (case.get("expected_output") or "")[:400]
    output     = (case.get("actual_output") or "")[:400]
    api_err    = case.get("api_error", False)

    # Header bar
    hdr_bg = C_FAIL_BG if status == "FAIL" else C_WARN_BG
    hdr_color = C_FAIL if status == "FAIL" else C_WARN

    meta_line = f"{cat}  ·  {lang}  ·  Section: {sec}  ·  Pass rate: {_pct(rate)}  ·  {'DECLINE' if is_decline else 'RESPOND'}"
    if api_err:
        meta_line += "  ·  ⚠ API ERROR"

    case_header = Table(
        [[
            Paragraph(f'<b>[{idx}] {name}</b>', st["body_bold"]),
            Paragraph(meta_line, st["small"]),
        ]],
        colWidths=[(W - 2 * MARGIN) * 0.35, (W - 2 * MARGIN) * 0.65],
    )
    case_header.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), hdr_bg),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("BOX",          (0, 0), (-1, -1), 0.5, hdr_color),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))

    # Q&A block
    qa_data = [
        [Paragraph("<b>Input</b>", st["small"]),  Paragraph(question, st["cell"])],
    ]
    if expected:
        qa_data.append([Paragraph("<b>Expected</b>", st["small"]), Paragraph(expected, st["cell"])])
    qa_data.append([Paragraph("<b>Output</b>", st["small"]), Paragraph(output,   st["cell"])])
    qa_table = Table(
        qa_data,
        colWidths=[(W - 2 * MARGIN) * 0.10, (W - 2 * MARGIN) * 0.90],
    )
    qa_table.setStyle(TableStyle([
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("BOX",          (0, 0), (-1, -1), 0.4, C_BORDER),
        ("BACKGROUND",   (0, 0), (0, -1), C_ROW_ALT),
    ]))

    # Metrics table
    m_headers = ["Metric", "Score", "Threshold", "Status", "Reason"]
    m_rows = [m_headers]
    for m in case.get("metrics", []):
        score     = float(m.get("score") or 0.0)
        threshold = float(m.get("threshold") or 0.5)
        m_status  = m.get("status", "?")
        reason    = (m.get("reason") or "—")[:200]
        sk        = "pass_cell" if m_status == "PASS" else "fail_cell"
        m_rows.append([
            Paragraph(m.get("name", ""), st["cell_bold"]),
            Paragraph(f"{score:.3f}", st[sk]),
            Paragraph(f"{threshold:.2f}", st["cell"]),
            Paragraph(m_status, st[sk]),
            Paragraph(reason, st["small"]),
        ])

    avail = W - 2 * MARGIN
    m_col_w = [avail * r for r in [0.22, 0.10, 0.12, 0.09, 0.47]]
    m_table = Table(m_rows, colWidths=m_col_w, repeatRows=1)
    m_cmds = _base_table_style(header_bg=colors.HexColor("#ECEFF1"), n_rows=len(m_rows) - 1)
    for i, m in enumerate(case.get("metrics", []), start=1):
        if m.get("status") == "FAIL":
            m_cmds.append(("BACKGROUND", (0, i), (3, i), C_FAIL_BG))
    m_table.setStyle(TableStyle(m_cmds))

    # Failures summary
    failures = case.get("failures", [])
    failure_text = "; ".join(failures) if failures else "—"

    return [
        KeepTogether([
            case_header,
            Spacer(1, 1),
            qa_table,
            Spacer(1, 2),
            m_table,
            Paragraph(f"<b>Failure summary:</b> {failure_text}", st["small"]),
            Spacer(1, 6),
        ])
    ]


# ── Page template callbacks (header / footer) ──────────────────────────────

def _make_page_callbacks(title: str):
    def on_page(canvas, doc):
        canvas.saveState()
        # Top rule
        canvas.setStrokeColor(C_ACCENT)
        canvas.setLineWidth(1.5)
        canvas.line(MARGIN, H - 12 * mm, W - MARGIN, H - 12 * mm)
        # Header text
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#757575"))
        canvas.drawString(MARGIN, H - 10 * mm, title)
        canvas.drawRightString(W - MARGIN, H - 10 * mm,
                               datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        # Footer
        canvas.line(MARGIN, 10 * mm, W - MARGIN, 10 * mm)
        canvas.drawString(MARGIN, 7 * mm, "OAN Evaluation Report  —  Confidential")
        canvas.drawRightString(W - MARGIN, 7 * mm, f"Page {doc.page}")
        canvas.restoreState()

    return on_page


# ── Main entry point ────────────────────────────────────────────────────────

def save_report_pdf(report: dict[str, Any], output_path: "str | Path") -> Path:
    """
    Render a detailed, colour-coded PDF from an OAN evaluation report dict.
    Returns the Path of the written file.

    Requires: reportlab
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta    = report.get("meta", {})
    summary = report.get("summary", {})
    failed  = report.get("failed_cases", [])

    judge   = meta.get("judge_model", "unknown")
    gen_at  = meta.get("generated_at", "")
    pr_thr  = float(meta.get("pass_rate_threshold") or 0.7)
    ge_thr  = float(meta.get("geval_threshold") or 0.5)
    title   = f"OAN Eval · {judge} · {gen_at[:10]}"

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=20 * mm, bottomMargin=16 * mm,
        title="OAN Evaluation Report",
        author="OAN Eval System",
    )
    on_page = _make_page_callbacks(title)

    st    = _styles()
    story = []

    # ── Cover / overview ───────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("OAN Evaluation Report", st["cover_title"]))
    story.append(Paragraph(f"Judge model: {judge}", st["cover_sub"]))
    story.append(Paragraph(
        f"Generated: {gen_at}  ·  "
        f"Pass-rate threshold: {_pct(pr_thr)}  ·  "
        f"G-Eval threshold: {ge_thr}",
        st["cover_meta"],
    ))
    
    run_meta = meta.get("run_meta")
    if run_meta:
        story.append(Spacer(1, 4))
        meta_lines = []
        if run_meta.get("name"): meta_lines.append(f"Name: {run_meta['name']}")
        if run_meta.get("version"): meta_lines.append(f"Version: {run_meta['version']}")
        if run_meta.get("note"): meta_lines.append(f"Note: {run_meta['note']}")
        if meta_lines:
            story.append(Paragraph("<b>Run Metadata: </b>" + "  ·  ".join(meta_lines), st["small"]))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_ACCENT, spaceAfter=8))

    # KPI cards
    story.append(_kpi_table(meta, st))
    story.append(Spacer(1, 10))

    # ── Top failing categories ─────────────────────────────────────────────
    story.append(Paragraph("Top Failing Categories", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=4))
    top_fail = summary.get("top_failing_categories", [])
    story.append(_top_failing_table(top_fail, st))
    story.append(Spacer(1, 8))

    # ── By Category ───────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Results by Category", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=4))
    by_cat = summary.get("by_category", [])
    if by_cat:
        story.append(_category_table(by_cat, st))
    story.append(Spacer(1, 8))

    # ── By Section ────────────────────────────────────────────────────────
    story.append(Paragraph("Results by Section", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=4))
    by_sec = summary.get("by_section", [])
    if by_sec:
        story.append(_section_table(by_sec, st))
    story.append(Spacer(1, 8))

    # ── By Metric ─────────────────────────────────────────────────────────
    story.append(Paragraph("Results by Metric", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=4))
    by_metric = summary.get("by_metric", [])
    if by_metric:
        story.append(_metric_table(by_metric, st))
    story.append(Spacer(1, 8))

    # ── By Language ───────────────────────────────────────────────────────
    story.append(Paragraph("Results by Language", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=4))
    by_lang = summary.get("by_language", [])
    if by_lang:
        story.append(_language_table(by_lang, st))

    # ── Failed Case Detail ────────────────────────────────────────────────
    if failed:
        story.append(PageBreak())
        story.append(Paragraph("Failed Case Detail", st["h1"]))
        story.append(Paragraph(
            f"{len(failed)} case(s) failed or errored — sorted by most metrics failed first.",
            st["body"],
        ))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6))
        for idx, case in enumerate(failed, 1):
            story.extend(_failed_case_block(case, st, idx))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    return path
