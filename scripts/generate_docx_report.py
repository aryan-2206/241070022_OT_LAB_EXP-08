"""
Generate the official DOCX technical report from reports/report_source.md
Conforms to formal academic guidelines and VJTI institutional report standards.
"""

import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_MD_PATH = os.path.join(BASE_DIR, "reports", "report_source.md")
OUTPUT_DOCX_PATH = os.path.join(BASE_DIR, "reports", "241070022_AryanDoshi_OT_Experiment_08_Technical_Report.docx")

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="666666", sz="4"):
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement('w:tblBorders')
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), sz)
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), color)
        tblBorders.append(border)
    tblPr.append(tblBorders)

def make_row_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement('w:tblHeader')
    cantSplit = OxmlElement('w:cantSplit')
    trPr.append(tblHeader)
    trPr.append(cantSplit)

def make_row_cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    cantSplit = OxmlElement('w:cantSplit')
    trPr.append(cantSplit)

def add_inline_runs(paragraph, text, base_font_name="Calibri", base_font_size=11, base_color=(30, 30, 30)):
    """
    Parses bold (**), italic (*), code (`), and basic math ($) in markdown text.
    """
    # Tokenize by bold, inline code, and math
    pattern = re.compile(r'(\*\*.*?\*\*|`.*?`|\$.*?\$|\*.*?\*)')
    parts = pattern.split(text)
    
    for part in parts:
        if not part:
            continue
        run = paragraph.add_run()
        run.font.name = base_font_name
        run.font.size = Pt(base_font_size)
        run.font.color.rgb = RGBColor(*base_color)
        
        if part.startswith('**') and part.endswith('**') and len(part) >= 4:
            run.text = part[2:-2]
            run.bold = True
        elif part.startswith('`') and part.endswith('`') and len(part) >= 2:
            run.text = part[1:-1]
            run.font.name = "Consolas"
            run.font.size = Pt(base_font_size - 1)
            run.font.color.rgb = RGBColor(40, 40, 80)
        elif part.startswith('$') and part.endswith('$') and len(part) >= 2:
            # Inline math: format in Cambria Math or Italic
            run.text = part[1:-1]
            run.italic = True
        elif part.startswith('*') and part.endswith('*') and len(part) >= 2:
            run.text = part[1:-1]
            run.italic = True
        else:
            run.text = part

def build_cover_page(doc):
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(20)
    title_p.paragraph_format.space_after = Pt(2)
    run = title_p.add_run("VEERMATA JIJABAI TECHNOLOGICAL INSTITUTE (VJTI)")
    run.font.name = "Calibri"
    run.font.size = Pt(15)
    run.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)

    sub_inst = doc.add_paragraph()
    sub_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_inst.paragraph_format.space_before = Pt(0)
    sub_inst.paragraph_format.space_after = Pt(2)
    r2 = sub_inst.add_run("(Autonomous Institute Affiliated to the University of Mumbai)\nMatunga, Mumbai – 400 019, Maharashtra, India")
    r2.font.name = "Calibri"
    r2.font.size = Pt(10)
    r2.font.color.rgb = RGBColor(80, 80, 80)

    dept_p = doc.add_paragraph()
    dept_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dept_p.paragraph_format.space_before = Pt(6)
    dept_p.paragraph_format.space_after = Pt(24)
    r3 = dept_p.add_run("DEPARTMENT OF COMPUTER ENGINEERING & INFORMATION TECHNOLOGY")
    r3.font.name = "Calibri"
    r3.font.size = Pt(11)
    r3.bold = True
    r3.font.color.rgb = RGBColor(50, 50, 50)

    # Decorative separator line
    sep_p = doc.add_paragraph()
    sep_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sep_p.paragraph_format.space_before = Pt(0)
    sep_p.paragraph_format.space_after = Pt(24)
    r_sep = sep_p.add_run("―" * 45)
    r_sep.font.color.rgb = RGBColor(0, 51, 102)
    r_sep.bold = True

    # Report Type
    type_p = doc.add_paragraph()
    type_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    type_p.paragraph_format.space_before = Pt(10)
    type_p.paragraph_format.space_after = Pt(12)
    rt = type_p.add_run("LABORATORY TECHNICAL REPORT")
    rt.font.name = "Calibri"
    rt.font.size = Pt(13)
    rt.bold = True
    rt.font.color.rgb = RGBColor(120, 120, 120)

    # Main Title
    main_p = doc.add_paragraph()
    main_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    main_p.paragraph_format.space_before = Pt(6)
    main_p.paragraph_format.space_after = Pt(12)
    rmt = main_p.add_run("OPTIMIZE ANYTHING: THE 14-DAY VIBE-CODING CHALLENGE")
    rmt.font.name = "Calibri"
    rmt.font.size = Pt(21)
    rmt.bold = True
    rmt.font.color.rgb = RGBColor(0, 0, 0)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(36)
    rst = sub_p.add_run("Complete Implementation, Benchmarking, and Empirical Evaluation across Ten Compulsory Tracks")
    rst.font.name = "Calibri"
    rst.font.size = Pt(12)
    rst.italic = True
    rst.font.color.rgb = RGBColor(70, 70, 70)

    # Metadata Box Table
    meta_table = doc.add_table(rows=9, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(meta_table, color="CCCCCC", sz="4")
    
    metadata_fields = [
        ("Course Name", "Optimization Techniques Laboratory"),
        ("Course Code", "CS-702"),
        ("Degree Program", "Bachelor of Technology (B.Tech) in Computer Engineering"),
        ("Class & Year", "Third Year (TY B.Tech CE), Semester VI"),
        ("Student Name", "Aryan Doshi"),
        ("Student ID / Roll No.", "241070022"),
        ("Faculty Evaluator", "Department of Computer Engineering, VJTI Mumbai"),
        ("Date of Submission", "October 26, 2026"),
        ("GitHub Repository", "https://github.com/aryan-2206/241070022_OT_LAB_EXP-08.git"),
    ]
    
    for i, (k, v) in enumerate(metadata_fields):
        row = meta_table.rows[i]
        make_row_cant_split(row)
        c0 = row.cells[0]
        c1 = row.cells[1]
        c0.width = Inches(2.2)
        c1.width = Inches(4.0)
        set_cell_background(c0, "F8F9FA")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=70, bottom=70, left=120, right=120)
        set_cell_margins(c1, top=70, bottom=70, left=120, right=120)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(k)
        r0.font.name = "Calibri"
        r0.font.size = Pt(10)
        r0.bold = True
        r0.font.color.rgb = RGBColor(40, 40, 40)
        
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_before = Pt(0)
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(v)
        r1.font.name = "Calibri"
        r1.font.size = Pt(10)
        r1.font.color.rgb = RGBColor(20, 20, 20)

    doc.add_page_break()

def generate_docx():
    print(f"Reading markdown source from {SOURCE_MD_PATH}...")
    with open(SOURCE_MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = docx.Document()
    
    # Page setup - A4 with 1-inch margins
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    # Set normal style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(30, 30, 30)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(5)

    print("Building cover page...")
    build_cover_page(doc)

    lines = md_text.split("\n")
    
    # Skip cover lines in markdown up to the first '## ABSTRACT'
    start_idx = 0
    for idx, line in enumerate(lines):
        if line.strip() == "## ABSTRACT":
            start_idx = idx
            break

    in_code_block = False
    code_lines = []
    
    in_table = False
    table_lines = []
    
    current_idx = start_idx
    total_lines = len(lines)
    
    while current_idx < total_lines:
        line = lines[current_idx]
        stripped = line.strip()
        
        # Check code block fences
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lines = []
                current_idx += 1
                continue
            else:
                in_code_block = False
                # Flush code block
                table = doc.add_table(rows=1, cols=1)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                cell = table.cell(0, 0)
                cell.width = Inches(6.25)
                set_cell_background(cell, "F4F4F4")
                set_table_borders(table, color="DDDDDD", sz="4")
                set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
                cp = cell.paragraphs[0]
                cp.paragraph_format.space_before = Pt(0)
                cp.paragraph_format.space_after = Pt(0)
                cp.paragraph_format.line_spacing = 1.05
                run = cp.add_run("\n".join(code_lines))
                run.font.name = "Consolas"
                run.font.size = Pt(9.0)
                run.font.color.rgb = RGBColor(20, 20, 20)
                
                sp_after = doc.add_paragraph()
                sp_after.paragraph_format.space_before = Pt(0)
                sp_after.paragraph_format.space_after = Pt(4)
                
                code_lines = []
                current_idx += 1
                continue
                
        if in_code_block:
            code_lines.append(line)
            current_idx += 1
            continue

        # Check table lines
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                in_table = True
                table_lines = [stripped]
            else:
                table_lines.append(stripped)
            current_idx += 1
            continue
        elif in_table:
            # End of table, process collected table lines
            in_table = False
            parse_and_add_table(doc, table_lines)
            table_lines = []
            # Do not continue; allow this line to be processed below

        # Check section break
        if stripped == "---":
            # If preceding was front matter, maybe page break
            current_idx += 1
            continue

        # Check Figure tag: ![Caption](path)
        fig_match = re.match(r'!\[(.*?)\]\((.*?)\)', stripped)
        if fig_match:
            caption_text = fig_match.group(1)
            img_rel_path = fig_match.group(2)
            # Resolve image path
            img_clean_path = img_rel_path.replace("../", "").replace("/", os.sep)
            abs_img_path = os.path.join(BASE_DIR, img_clean_path)
            
            if os.path.exists(abs_img_path):
                img_p = doc.add_paragraph()
                img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                img_p.paragraph_format.space_before = Pt(8)
                img_p.paragraph_format.space_after = Pt(3)
                img_p.add_run().add_picture(abs_img_path, width=Inches(5.6))
                
                cap_p = doc.add_paragraph()
                cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap_p.paragraph_format.space_before = Pt(2)
                cap_p.paragraph_format.space_after = Pt(12)
                
                # Format caption: split by first colon if available
                if ":" in caption_text:
                    prefix, rest = caption_text.split(":", 1)
                    r_pre = cap_p.add_run(prefix.strip() + ":")
                    r_pre.bold = True
                    r_pre.font.name = "Calibri"
                    r_pre.font.size = Pt(10)
                    r_pre.font.color.rgb = RGBColor(50, 50, 50)
                    
                    r_rest = cap_p.add_run(" " + rest.strip())
                    r_rest.italic = True
                    r_rest.font.name = "Calibri"
                    r_rest.font.size = Pt(9.5)
                    r_rest.font.color.rgb = RGBColor(60, 60, 60)
                else:
                    r_all = cap_p.add_run(caption_text)
                    r_all.italic = True
                    r_all.font.name = "Calibri"
                    r_all.font.size = Pt(9.5)
            else:
                print(f"Warning: Figure not found at {abs_img_path}")
            current_idx += 1
            continue

        # Check Headings
        if stripped.startswith("#### "):
            h_text = stripped[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            add_inline_runs(p, h_text, base_font_size=11, base_color=(30, 30, 30))
            for r in p.runs:
                r.bold = True
            current_idx += 1
            continue
            
        if stripped.startswith("### "):
            h_text = stripped[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            add_inline_runs(p, h_text, base_font_size=12.5, base_color=(0, 51, 102))
            for r in p.runs:
                r.bold = True
            current_idx += 1
            continue

        if stripped.startswith("## "):
            h_text = stripped[3:].strip()
            # If it's a chapter heading, add a page break before it unless it's ABSTRACT
            if h_text.startswith("CHAPTER ") or h_text in ["ABSTRACT", "TABLE OF CONTENTS", "REFERENCES", "APPENDICES"]:
                if h_text.startswith("CHAPTER "):
                    doc.add_page_break()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            add_inline_runs(p, h_text, base_font_size=15, base_color=(0, 30, 70))
            for r in p.runs:
                r.bold = True
            current_idx += 1
            continue

        if stripped.startswith("# "):
            h_text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(18)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.keep_with_next = True
            add_inline_runs(p, h_text, base_font_size=17, base_color=(0, 20, 60))
            for r in p.runs:
                r.bold = True
            current_idx += 1
            continue

        # Check Table Caption (e.g. **Table 3: Track A Continuous...**)
        if stripped.startswith("**Table ") and stripped.endswith("**"):
            cap_text = stripped[2:-2].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            if ":" in cap_text:
                pre, post = cap_text.split(":", 1)
                r_pre = p.add_run(pre.strip() + ":")
                r_pre.bold = True
                r_pre.font.name = "Calibri"
                r_pre.font.size = Pt(10.5)
                r_pre.font.color.rgb = RGBColor(0, 51, 102)
                r_post = p.add_run(" " + post.strip())
                r_post.bold = True
                r_post.font.name = "Calibri"
                r_post.font.size = Pt(10.5)
                r_post.font.color.rgb = RGBColor(30, 30, 30)
            else:
                r_all = p.add_run(cap_text)
                r_all.bold = True
                r_all.font.name = "Calibri"
                r_all.font.size = Pt(10.5)
            current_idx += 1
            continue

        # Check Bullet list item
        if stripped.startswith("- ") or stripped.startswith("* "):
            item_text = stripped[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            add_inline_runs(p, item_text, base_font_size=11)
            current_idx += 1
            continue

        # Check Numbered list item: e.g. 1. ...
        num_match = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if num_match:
            num_str = num_match.group(1)
            item_text = num_match.group(2)
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            
            r_num = p.add_run(f"{num_str}.  ")
            r_num.bold = True
            r_num.font.name = "Calibri"
            r_num.font.size = Pt(11)
            
            add_inline_runs(p, item_text, base_font_size=11)
            current_idx += 1
            continue

        # Check Display math: $$ ... $$
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) >= 4:
            math_text = stripped[2:-2].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(5)
            p.paragraph_format.space_after = Pt(5)
            r = p.add_run(math_text)
            r.font.name = "Cambria Math"
            r.font.size = Pt(11)
            r.italic = True
            r.font.color.rgb = RGBColor(10, 10, 50)
            current_idx += 1
            continue

        # Regular non-empty paragraph
        if stripped:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.line_spacing = 1.15
            add_inline_runs(p, stripped, base_font_size=11)

        current_idx += 1

    # In case a table was at the very end
    if in_table and table_lines:
        parse_and_add_table(doc, table_lines)

    print(f"Saving finalized Word document to {OUTPUT_DOCX_PATH}...")
    doc.save(OUTPUT_DOCX_PATH)
    print("Document successfully created and saved.")

def parse_and_add_table(doc, table_lines):
    # Parse rows and columns from markdown
    rows_data = []
    for l in table_lines:
        # Check if separator row |---|---|
        parts = [p.strip() for p in l.split("|")[1:-1]]
        if all(re.match(r'^:?-+:?$', p) for p in parts if p):
            continue
        rows_data.append(parts)
        
    if not rows_data:
        return
        
    num_rows = len(rows_data)
    num_cols = max(len(r) for r in rows_data)
    
    # Normalize row lengths
    for r in rows_data:
        while len(r) < num_cols:
            r.append("")
            
    table = doc.add_table(rows=num_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color="777777", sz="4")
    
    # Calculate column widths to fit 6.27 inches
    avail_width = 6.25
    col_width = Inches(avail_width / num_cols)
    
    for row_idx, r_data in enumerate(rows_data):
        row = table.rows[row_idx]
        if row_idx == 0:
            make_row_header(row)
        else:
            make_row_cant_split(row)
            
        for col_idx, cell_value in enumerate(r_data):
            cell = row.cells[col_idx]
            cell.width = col_width
            set_cell_margins(cell, top=60, bottom=60, left=90, right=90)
            
            if row_idx == 0:
                set_cell_background(cell, "EAECEE")
            elif row_idx % 2 == 1:
                set_cell_background(cell, "FFFFFF")
            else:
                set_cell_background(cell, "F8F9F9")
                
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            
            # Align center if short number or symbol
            if re.match(r'^(\$?-?[\d\.,\+%e\^\{\}\-\*]+(?:\s*[a-zA-Z\%\$]*)?|-|\*\*.*\*\*)$', cell_value.strip()):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                
            if row_idx == 0:
                # Header text bold
                add_inline_runs(p, cell_value, base_font_size=9.5, base_color=(10, 25, 47))
                for r in p.runs:
                    r.bold = True
            else:
                add_inline_runs(p, cell_value, base_font_size=9.0, base_color=(30, 30, 30))

    # Add small spacing after table
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)

if __name__ == "__main__":
    generate_docx()
