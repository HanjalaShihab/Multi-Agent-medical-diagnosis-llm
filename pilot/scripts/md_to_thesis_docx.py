#!/usr/bin/env python3
"""Convert pilot/thesis/THESIS_DRAFT.md into a DIU-style Word draft.

Uses the repository's SWE thesis template for styles, then clears its body and
rebuilds it from the markdown draft. Keeps Times New Roman 12, double spacing,
A4, and the DIU margins (left 4.0 cm; top/bottom/right 2.5 cm).
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "SWE_Thesis Template_Spiral Bound.docx"
DEFAULT_MD = ROOT / "pilot" / "thesis" / "THESIS_DRAFT.md"
DEFAULT_OUT = ROOT / "pilot" / "thesis" / "BTS_Thesis_Draft.docx"


def set_run(run, size=12, bold=False, italic=False, mono=False):
    run.font.name = "Courier New" if mono else "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    return run


def add_page_break(doc):
    doc.add_page_break()


def add_table(doc, rows: list[list[str]]):
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.style = "Table Grid"
    table.autofit = True
    for i, row in enumerate(rows):
        for j in range(ncols):
            cell = table.cell(i, j)
            text = row[j] if j < len(row) else ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            set_run(p.add_run(text), size=10, bold=(i == 0)).font.name = "Times New Roman"
    doc.add_paragraph().paragraph_format.line_spacing = 1.0


def build(md_path: Path, out_path: Path):
    doc = Document(TEMPLATE)
    body = doc.element.body
    sect_pr = body.sectPr
    for child in list(body):
        if child is not sect_pr:
            body.remove(child)

    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(4.0)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 2.0
    normal.paragraph_format.space_after = Pt(0)

    for name, size in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)):
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.line_spacing = 2.0
        st.paragraph_format.space_before = Pt(12)
        st.paragraph_format.space_after = Pt(6)

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    first_title_done = False
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "[PAGEBREAK]":
            add_page_break(doc)
            i += 1
            continue

        if stripped.startswith("# ") and not first_title_done:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 2.0
            set_run(p.add_run(stripped[2:].strip()), size=18, bold=True)
            first_title_done = True
            i += 1
            continue

        if stripped.startswith("## "):
            p = doc.add_paragraph(style="Heading 1")
            set_run(p.add_run(stripped[3:].strip()), size=14, bold=True)
            i += 1
            continue

        if stripped.startswith("### "):
            p = doc.add_paragraph(style="Heading 2")
            set_run(p.add_run(stripped[4:].strip()), size=12, bold=True)
            i += 1
            continue

        if stripped.startswith("#### "):
            p = doc.add_paragraph(style="Heading 3")
            set_run(p.add_run(stripped[5:].strip()), size=12, bold=True)
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            for code_line in block:
                p = doc.add_paragraph()
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.left_indent = Cm(1.0)
                set_run(p.add_run(code_line), size=10, mono=True)
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            rows = []
            while i < len(lines):
                s = lines[i].strip()
                if not (s.startswith("|") and s.endswith("|")):
                    break
                cells = [c.strip() for c in s.strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            add_table(doc, rows)
            continue

        if stripped.startswith("**") and stripped.endswith("**") and (
            stripped.startswith("**Table") or stripped.startswith("**Figure")
        ):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            set_run(p.add_run(stripped[2:-2]), size=11, bold=True)
            i += 1
            continue

        if stripped.startswith("- "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(1.0)
            p.paragraph_format.line_spacing = 2.0
            set_run(p.add_run("• " + stripped[2:]))
            i += 1
            continue

        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 2.0
        set_run(p.add_run(stripped))
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    md = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MD
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    build(md, out)
