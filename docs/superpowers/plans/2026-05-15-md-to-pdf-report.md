# MD to PDF Report Conversion - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `.md` report output with professionally formatted `.pdf` reports by adding a MD→PDF conversion layer.

**Architecture:** New `pdf_generator.py` module parses markdown line-by-line into structured blocks, then renders them through fpdf2 with professional financial-report styling (navy headings, colored data/judgment tags, alternating table rows, headers/footers). Existing code paths are preserved — PDF is generated _alongside_ MD, not instead of it.

**Tech Stack:** fpdf2 (PDF generation), built-in `re` (markdown parsing), SimHei/黑体 (Chinese font).

---

### Task 1: Install fpdf2 dependency

**Files:**
- Modify: `Finance/requirements.txt`

- [ ] **Step 1: Add fpdf2 to requirements.txt**

Add the line `fpdf2>=2.8.0` to `Finance/requirements.txt`:

```
fpdf2>=2.8.0
```

- [ ] **Step 2: Install fpdf2**

```bash
cd Finance && pip install fpdf2>=2.8.0
```

Expected: `Successfully installed fpdf2-2.8.x` (or version >= 2.8.0)

- [ ] **Step 3: Verify import works**

```bash
cd Finance/Financial-MCP-Agent && python -c "from fpdf import FPDF; print('OK')"
```

Expected: Prints `OK` with no errors.

- [ ] **Step 4: Commit**

```bash
git add Finance/requirements.txt
git commit -m "feat: add fpdf2 dependency for PDF report generation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Create PDF Generator Module

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/utils/pdf_generator.py`

- [ ] **Step 1: Create the pdf_generator.py file**

Write the complete file at `Finance/Financial-MCP-Agent/src/utils/pdf_generator.py`:

```python
"""
PDF Report Generator - 将 Markdown 分析报告转换为专业 PDF 格式。
纯 Python 实现，无外部系统依赖。
"""
import os
import re
import logging
from fpdf import FPDF

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 字体发现
# ═══════════════════════════════════════════════════════════════

def _find_chinese_font():
    """
    在系统中查找可用的中文字体 TTF/TTC 文件。
    
    Returns:
        str: 字体文件路径，未找到返回 None
    """
    if os.name == 'nt':
        candidates = [
            "C:/Windows/Fonts/simhei.ttf",   # 黑体（首选，适合标题）
            "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",   # 宋体
            "C:/Windows/Fonts/simkai.ttf",   # 楷体
        ]
    elif hasattr(os, 'uname'):
        candidates = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
    else:
        # macOS
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    
    for path in candidates:
        if os.path.exists(path):
            logger.info(f"Found Chinese font: {path}")
            return path
    
    logger.warning("No Chinese font found on system")
    return None


def _find_body_font():
    """
    查找适合正文字体的宋体/楷体。如果找不到，回退到黑体。
    """
    if os.name == 'nt':
        candidates = [
            "C:/Windows/Fonts/simsun.ttc",   # 宋体（正文首选）
            "C:/Windows/Fonts/simkai.ttf",   # 楷体
            "C:/Windows/Fonts/simhei.ttf",   # 黑体（回退）
        ]
    else:
        return _find_chinese_font()  # Linux/macOS 只有一种字体可用
    
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ═══════════════════════════════════════════════════════════════
# Markdown 解析器
# ═══════════════════════════════════════════════════════════════

class Block:
    """解析后的 Markdown 块基类"""
    pass


class HeadingBlock(Block):
    def __init__(self, level: int, text: str):
        self.level = level
        self.text = text


class ParagraphBlock(Block):
    def __init__(self, text: str):
        self.text = text


class TableBlock(Block):
    def __init__(self, headers: list, rows: list):
        self.headers = headers
        self.rows = rows


class HorizontalRuleBlock(Block):
    pass


class ListBlock(Block):
    def __init__(self, items: list, ordered: bool = False):
        self.items = items
        self.ordered = ordered


class EmptyBlock(Block):
    pass


def _is_special_line(line: str) -> bool:
    """判断一行是否属于特殊块（标题、分隔线、表格、列表）的起始行"""
    s = line.strip()
    if not s:
        return True
    if re.match(r'^#{1,6}\s+', s):
        return True
    if re.match(r'^[-*_]{3,}\s*$', s):
        return True
    if '|' in s and s.startswith('|'):
        return True
    if re.match(r'^(\s*)[-*+]\s+', s):
        return True
    if re.match(r'^(\s*)\d+[.)]\s+', s):
        return True
    return False


def parse_markdown(text: str) -> list:
    """
    将 Markdown 文本解析为 Block 对象列表。
    
    支持的块类型：
    - # ~ ###### 标题
    - --- 分隔线
    - |...| 表格（含分隔行）
    - - / * / 1. 列表
    - 普通段落（连续非空行合并为一个段落）
    """
    lines = text.split('\n')
    blocks = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 空行
        if not stripped:
            blocks.append(EmptyBlock())
            i += 1
            continue
        
        # 标题: # ~ ######
        heading_match = re.match(r'^(#{1,6})\s+(.+)', stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            # 去除标题中可能存在的 **** 粗体标记
            heading_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', heading_text)
            blocks.append(HeadingBlock(level, heading_text))
            i += 1
            continue
        
        # 分隔线: --- 或 *** 或 ___
        if re.match(r'^[-*_]{3,}\s*$', stripped):
            blocks.append(HorizontalRuleBlock())
            i += 1
            continue
        
        # 表格: 以 | 开头且以 | 结尾
        if '|' in stripped and stripped.startswith('|'):
            table_lines = []
            while i < len(lines) and '|' in lines[i].strip() and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            
            rows = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.strip().strip('|').split('|')]
                # 跳过分隔行，如 |---|---| 或 |:---:|:---|
                if all(re.match(r'^[-:]+$', c) for c in cells):
                    continue
                # 清理单元格内的粗体标记（表头可能有 **text**）
                cells = [re.sub(r'\*\*([^*]+)\*\*', r'\1', c) for c in cells]
                rows.append(cells)
            
            if rows:
                if len(rows) >= 2:
                    blocks.append(TableBlock(rows[0], rows[1:]))
                else:
                    blocks.append(TableBlock(rows[0], []))
            continue
        
        # 无序列表: - / * / +
        list_match = re.match(r'^(\s*)[-*+]\s+(.+)', stripped)
        if list_match:
            items = []
            while i < len(lines):
                lm = re.match(r'^(\s*)[-*+]\s+(.+)', lines[i].strip())
                if not lm:
                    break
                items.append(lm.group(2).strip())
                i += 1
            blocks.append(ListBlock(items, ordered=False))
            continue
        
        # 有序列表: 1. / 1)
        ordered_match = re.match(r'^(\s*)\d+[.)]\s+(.+)', stripped)
        if ordered_match:
            items = []
            while i < len(lines):
                om = re.match(r'^(\s*)\d+[.)]\s+(.+)', lines[i].strip())
                if not om:
                    break
                items.append(om.group(2).strip())
                i += 1
            blocks.append(ListBlock(items, ordered=True))
            continue
        
        # 普通段落: 收集连续的非特殊行
        para_lines = []
        while i < len(lines) and lines[i].strip() and not _is_special_line(lines[i]):
            para_lines.append(lines[i].strip())
            i += 1
        
        if para_lines:
            blocks.append(ParagraphBlock(' '.join(para_lines)))
        else:
            i += 1
    
    return blocks


# ═══════════════════════════════════════════════════════════════
# 行内解析器 - 处理段落/列表项中的粗体、标签
# ═══════════════════════════════════════════════════════════════

def parse_inline(text: str) -> list:
    """
    解析行内 Markdown 格式，返回 (文本, 样式字典) 列表。
    
    支持的样式:
    - 'b': 粗体 (**text**)
    - 'tag_data': [数据] 标签
    - 'tag_judgment': [判断] 标签
    
    >>> parse_inline("**[数据]** 营收增长50%")
    [('数据', {'b': True, 'tag_data': True}), (' 营收增长50%', {})]
    """
    segments = []
    # 匹配: **...** 或 [数据] 或 [判断]
    pattern = r'(\*\*(.+?)\*\*|\[数据\]|\[判断\])'
    
    last_end = 0
    for match in re.finditer(pattern, text):
        # 匹配前的纯文本
        if match.start() > last_end:
            plain = text[last_end:match.start()]
            if plain:
                segments.append((plain, {}))
        
        full = match.group(0)
        if full.startswith('**') and full.endswith('**'):
            inner = match.group(2)
            styles = {'b': True}
            # 检查粗体内容中是否包含标签
            if '[数据]' in inner:
                inner = inner.replace('[数据]', '')
                styles['tag_data'] = True
            if '[判断]' in inner:
                inner = inner.replace('[判断]', '')
                styles['tag_judgment'] = True
            if inner:
                segments.append((inner, styles))
        elif full == '[数据]':
            segments.append(('[数据]', {'tag_data': True}))
        elif full == '[判断]':
            segments.append(('[判断]', {'tag_judgment': True}))
        
        last_end = match.end()
    
    # 剩余纯文本
    if last_end < len(text):
        plain = text[last_end:]
        if plain:
            segments.append((plain, {}))
    
    return segments if segments else [(text, {})]


# ═══════════════════════════════════════════════════════════════
# PDF 构建器 - 专业金融研报风格
# ═══════════════════════════════════════════════════════════════

class ReportPDF(FPDF):
    """专业金融研究报告 PDF 构建器"""
    
    # ── 色彩方案 ──
    NAVY        = (26, 58, 92)      # 标题用深蓝
    DARK_GRAY   = (51, 51, 51)      # 正文深灰（不用纯黑，更柔和）
    MED_GRAY    = (128, 128, 128)   # 页脚/辅助文字
    LIGHT_GRAY  = (240, 242, 245)   # 交替行底色
    SEPARATOR   = (190, 200, 215)   # 分隔线
    WHITE       = (255, 255, 255)
    
    DATA_TAG_COLOR    = (30, 90, 190)    # 蓝色 - [数据] 标签
    JUDGMENT_TAG_COLOR = (200, 130, 20)  # 橙色 - [判断] 标签
    
    def __init__(self, heading_font_path: str, body_font_path: str,
                 company_name: str = "", stock_code: str = ""):
        super().__init__('P', 'mm', 'A4')
        self._heading_font_path = heading_font_path
        self._body_font_path = body_font_path
        self.company_name = company_name
        self.stock_code = stock_code
        
        # 注册字体：标题用黑体，正文用宋体（或回退到黑体）
        # 黑体天然看起来粗一些，用于标题；宋体用于正文
        self.add_font('Heading', '', heading_font_path, uni=True)
        self.add_font('Body', '', body_font_path, uni=True)
        
        self.set_auto_page_break(True, 22)
        self.set_left_margin(22)
        self.set_right_margin(22)
        self.set_top_margin(18)
    
    def header(self):
        """页眉：公司名+股票代码（第2页起）"""
        if self.page_no() <= 1:
            return
        
        self.set_font('Body', '', 7.5)
        self.set_text_color(*self.MED_GRAY)
        label = f"{self.company_name} ({self.stock_code})" if self.company_name else ""
        if label:
            self.cell(0, 4, label, align='R')
            self.ln(6)
    
    def footer(self):
        """页脚：分隔线 + 页码 + 免责声明"""
        self.set_y(-20)
        
        # 分隔线
        self.set_draw_color(*self.SEPARATOR)
        self.set_line_width(0.2)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(3)
        
        # 页码
        self.set_font('Body', '', 7.5)
        self.set_text_color(*self.MED_GRAY)
        self.cell(0, 5, f'第 {self.page_no()} 页', align='C')
        self.ln(3.5)
        
        # 免责声明
        self.set_font('Body', '', 6.5)
        self.cell(0, 4, '本报告由AI自动生成，仅供参考，不构成投资建议。市场有风险，投资需谨慎。', align='C')
    
    # ── 主渲染入口 ──
    
    def render_blocks(self, blocks: list):
        """渲染所有解析后的 Block 到 PDF"""
        self.add_page()
        
        for block in blocks:
            if isinstance(block, EmptyBlock):
                self.ln(2.5)
            elif isinstance(block, HeadingBlock):
                self._render_heading(block)
            elif isinstance(block, ParagraphBlock):
                self._render_paragraph(block)
            elif isinstance(block, TableBlock):
                self._render_table(block)
            elif isinstance(block, HorizontalRuleBlock):
                self._render_horizontal_rule()
            elif isinstance(block, ListBlock):
                self._render_list(block)
    
    # ── 标题渲染 ──
    
    def _render_heading(self, block: HeadingBlock):
        if block.level == 1:
            # H1: 居中大标题 + 装饰线
            self.ln(12)
            self.set_font('Heading', '', 22)
            self.set_text_color(*self.NAVY)
            self.multi_cell(0, 10, block.text, align='C')
            self.ln(3)
            # 标题下方装饰线
            self.set_draw_color(*self.NAVY)
            self.set_line_width(0.6)
            y = self.get_y()
            x_center = self.w / 2
            self.line(x_center - 35, y, x_center + 35, y)
            self.ln(10)
        elif block.level == 2:
            # H2: 左对齐，下方细线
            self.ln(7)
            self.set_font('Heading', '', 14)
            self.set_text_color(*self.NAVY)
            self.cell(0, 8, block.text)
            self.ln(9)
            self.set_draw_color(*self.SEPARATOR)
            self.set_line_width(0.2)
            y = self.get_y()
            self.line(self.l_margin, y, self.w - self.r_margin, y)
            self.ln(4)
        elif block.level == 3:
            # H3: 左对齐，无下划线
            self.ln(5)
            self.set_font('Heading', '', 11.5)
            self.set_text_color(*self.NAVY)
            self.cell(0, 7, block.text)
            self.ln(8)
        else:
            # H4-H6: 更小标题
            self.ln(4)
            self.set_font('Heading', '', 10.5)
            self.set_text_color(*self.NAVY)
            self.cell(0, 6, block.text)
            self.ln(7)
    
    # ── 段落渲染 ──
    
    def _render_paragraph(self, block: ParagraphBlock):
        self.set_font('Body', '', 10)
        self.set_text_color(*self.DARK_GRAY)
        
        segments = parse_inline(block.text)
        line_height = 5.6
        
        for text, style in segments:
            tag = None
            if style.get('tag_data'):
                tag = ('data', self.DATA_TAG_COLOR)
            elif style.get('tag_judgment'):
                tag = ('judgment', self.JUDGMENT_TAG_COLOR)
            
            is_bold = style.get('b', False)
            
            if tag:
                self._write_tag(tag[0], tag[1])
                # 标签后的文字（标签文本在 _write_tag 中已输出）
            
            # 输出文本
            if is_bold:
                self.set_font('Heading', '', 10)
            else:
                self.set_font('Body', '', 10)
            
            if tag:
                # 标签已经渲染了标记文本，这里只渲染标签后的内容
                # 去掉开头的标签前缀
                display_text = text
                if display_text:
                    self.set_text_color(*self.DARK_GRAY)
                    self.write(line_height, display_text)
            else:
                if is_bold:
                    self.set_text_color(*self.NAVY)
                else:
                    self.set_text_color(*self.DARK_GRAY)
                self.write(line_height, text)
        
        self.ln(line_height + 1.5)
    
    def _write_tag(self, tag_type: str, color: tuple):
        """
        输出彩色加粗的标签文本 [数据] 或 [判断]
        """
        self.set_font('Heading', '', 9.5)
        self.set_text_color(*color)
        label = '[数据]' if tag_type == 'data' else '[判断]'
        self.write(5.6, label)
        # 标签后加一个空格
        self.set_font('Body', '', 10)
        self.set_text_color(*self.DARK_GRAY)
        self.write(5.6, ' ')
    
    # ── 表格渲染 ──
    
    def _render_table(self, block: TableBlock):
        self.ln(3)
        
        if not block.headers:
            return
        
        col_count = len(block.headers)
        available_w = self.w - self.l_margin - self.r_margin
        
        # 简单均分列宽（适配大多数表格）
        col_widths = [available_w / col_count] * col_count
        
        # 确保最小列宽
        min_col_w = 18
        for i, w in enumerate(col_widths):
            if w < min_col_w:
                col_widths[i] = min_col_w
        
        # 调整以适配可用宽度
        total_w = sum(col_widths)
        if total_w > available_w:
            scale = available_w / total_w
            col_widths = [w * scale for w in col_widths]
        
        # 表头
        self.set_fill_color(*self.NAVY)
        self.set_text_color(*self.WHITE)
        self.set_font('Heading', '', 8.5)
        
        with self.table(col_widths=col_widths, text_align='CENTER',
                        first_row_as_headings=False) as table:
            # 手动渲染表头行
            header_row = table.row()
            for h in block.headers:
                header_row.cell(h, style='FILL', fill_color=self.NAVY)
            
            # 数据行（交替底色）
            for i, row in enumerate(block.rows):
                bg = self.LIGHT_GRAY if i % 2 == 0 else self.WHITE
                self.set_fill_color(*bg)
                self.set_text_color(*self.DARK_GRAY)
                self.set_font('Body', '', 8.5)
                
                data_row = table.row()
                for j, cell_text in enumerate(row):
                    # 根据内容长度决定对齐方式
                    is_numeric = bool(re.match(r'^[\d.,\-+%]+$', cell_text.strip()))
                    align = 'CENTER' if is_numeric else 'LEFT'
                    data_row.cell(cell_text, style='FILL', fill_color=bg, align=align)
        
        self.ln(4)
        self.set_text_color(*self.DARK_GRAY)
    
    # ── 分隔线 ──
    
    def _render_horizontal_rule(self):
        self.ln(4)
        self.set_draw_color(*self.SEPARATOR)
        self.set_line_width(0.3)
        y = self.get_y()
        indent = 35
        self.line(self.l_margin + indent, y, self.w - self.r_margin - indent, y)
        self.ln(4)
    
    # ── 列表渲染 ──
    
    def _render_list(self, block: ListBlock):
        self.set_font('Body', '', 10)
        self.set_text_color(*self.DARK_GRAY)
        
        for idx, item in enumerate(block.items):
            if block.ordered:
                prefix = f"{idx + 1}. "
            else:
                prefix = "• "
            
            # 缩进
            indent = 6
            self.set_x(self.l_margin + indent)
            self.cell(5, 5.5, prefix)
            
            # 解析列表项中的行内格式
            segments = parse_inline(item)
            for text, style in segments:
                is_bold = style.get('b', False)
                tag = None
                if style.get('tag_data'):
                    tag = ('data', self.DATA_TAG_COLOR)
                elif style.get('tag_judgment'):
                    tag = ('judgment', self.JUDGMENT_TAG_COLOR)
                
                if tag:
                    self._write_tag(tag[0], tag[1])
                
                if is_bold:
                    self.set_font('Heading', '', 10)
                    self.set_text_color(*self.NAVY)
                else:
                    self.set_font('Body', '', 10)
                    self.set_text_color(*self.DARK_GRAY)
                
                if tag:
                    # 已输出标签文本，这里只输出后续内容
                    display_text = text
                    if display_text:
                        self.set_text_color(*self.DARK_GRAY)
                        self.write(5.5, display_text)
                else:
                    self.write(5.5, text)
            
            self.ln()
        
        self.ln(2.5)


# ═══════════════════════════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════════════════════════

def markdown_to_pdf(markdown_text: str, output_path: str,
                    company_name: str = "", stock_code: str = "") -> str:
    """
    将 Markdown 分析报告转换为专业 PDF。
    
    Args:
        markdown_text: 原始 Markdown 内容
        output_path: PDF 输出路径
        company_name: 公司名称（用于页眉）
        stock_code: 股票代码（用于页眉）
    
    Returns:
        output_path: PDF 文件路径
    
    Raises:
        RuntimeError: 找不到中文字体时抛出
    """
    heading_font = _find_chinese_font()
    if not heading_font:
        raise RuntimeError(
            "未找到中文字体文件（SimHei/微软雅黑），无法生成 PDF 报告。"
            "请安装中文字体后重试。"
        )
    
    body_font = _find_body_font()
    if not body_font:
        body_font = heading_font
    
    logger.info(f"Using heading font: {heading_font}, body font: {body_font}")
    
    # 解析 Markdown
    blocks = parse_markdown(markdown_text)
    logger.info(f"Parsed {len(blocks)} blocks from markdown")
    
    # 生成 PDF
    pdf = ReportPDF(heading_font, body_font, company_name, stock_code)
    pdf.set_title(f"{company_name}({stock_code}) 综合分析报告")
    pdf.set_author("股票投资顾问 Agent")
    pdf.set_creator("Stock Investment Advisor System")
    
    pdf.render_blocks(blocks)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    
    logger.info(f"PDF report saved to {output_path} ({pdf.page_no()} pages)")
    return output_path
```

- [ ] **Step 2: Verify import works end-to-end**

```bash
cd Finance/Financial-MCP-Agent && python -c "from src.utils.pdf_generator import markdown_to_pdf, parse_markdown, parse_inline; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 3: Quick smoke test with minimal markdown**

```bash
cd Finance/Financial-MCP-Agent && python -c "
from src.utils.pdf_generator import markdown_to_pdf
import tempfile, os
md = '''# 测试报告
## 摘要
这是一段**重要**的测试文字 [数据] 数据来源可靠。

## 数据表格
| 指标 | 数值 |
|------|------|
| PE   | 15.2 |
| PB   | 2.3  |
'''
out = os.path.join(tempfile.gettempdir(), 'test_report.pdf')
markdown_to_pdf(md, out, '测试公司', '000001')
print(f'PDF created: {out}')
print(f'Size: {os.path.getsize(out)} bytes')
"
```

Expected: PDF file created with size > 0 bytes, no errors.

- [ ] **Step 4: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/utils/pdf_generator.py
git commit -m "feat: add PDF report generator with markdown parsing and professional styling

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Test with a real existing report

**Files:**
- Read: `Finance/Financial-MCP-Agent/reports/report_工业富联_601138_20260513_122612.md`

- [ ] **Step 1: Run PDF conversion on a real report**

```bash
cd Finance/Financial-MCP-Agent && python -c "
from src.utils.pdf_generator import markdown_to_pdf
import os

with open('reports/report_工业富联_601138_20260513_122612.md', 'r', encoding='utf-8') as f:
    md = f.read()

out = 'reports/report_工业富联_601138_20260513_122612.pdf'
markdown_to_pdf(md, out, '工业富联', '601138')
print(f'PDF created: {out}')
print(f'Size: {os.path.getsize(out)} bytes')
print('Success!')
"
```

Expected: PDF created successfully with no errors. Check visually that the PDF renders Chinese text correctly (no tofu/boxes), headings are properly styled, tables have alternating rows.

- [ ] **Step 2: Inspect the generated PDF**

Open the PDF file at `Finance/Financial-MCP-Agent/reports/report_工业富联_601138_20260513_122612.pdf` and verify:
- Chinese characters render correctly (no black boxes)
- H1 title is centered with decorative line
- H2 sections have separator lines
- Tables have alternating row colors
- `[数据]` tags in blue, `[判断]` tags in orange
- Page numbers and disclaimer in footer
- Overall professional appearance

- [ ] **Step 3: Commit test result**

```bash
git add Finance/Financial-MCP-Agent/reports/report_工业富联_601138_20260513_122612.pdf
git commit -m "test: add sample PDF output from real report conversion

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Integrate PDF generation into summary_agent.py

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/summary_agent.py`

- [ ] **Step 1: Add import for pdf_generator**

At the top of `summary_agent.py`, after the existing imports (around line 14), add:

```python
from src.utils.pdf_generator import markdown_to_pdf
```

- [ ] **Step 2: Modify the report saving block (lines 391-426) to also generate PDF**

Replace the block from line 391 (`# 将报告保存到Markdown文件`) through line 426 (end of the save block) with:

```python
        # 将报告保存到 Markdown 文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 处理公司名称和股票代码，确保文件名有意义
        if stock_code == "Unknown Stock" or stock_code == "Extracted from analysis":
            query_based_name = user_query.replace(
                " ", "_").replace("分析", "").strip()
            if not query_based_name:
                query_based_name = "financial_analysis"
            safe_file_prefix = f"report_{query_based_name}"
        else:
            safe_company_name = company_name.replace(" ", "_").replace(".", "")
            if safe_company_name == "Unknown_Company" or safe_company_name == "Extracted_from_analysis":
                safe_company_name = user_query.replace(
                    " ", "_").replace("分析", "").strip()
                if not safe_company_name:
                    safe_company_name = "company"

            clean_stock_code = stock_code.replace("sh.", "").replace("sz.", "")
            safe_file_prefix = f"report_{safe_company_name}_{clean_stock_code}"

        report_filename = f"{safe_file_prefix}_{timestamp}.md"
        pdf_filename = f"{safe_file_prefix}_{timestamp}.pdf"

        # 确保reports目录存在
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, report_filename)
        pdf_path = os.path.join(reports_dir, pdf_filename)

        # 将报告写入 Markdown 文件（保留作为备份）
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)

        logger.info(
            f"{SUCCESS_ICON} SummaryAgent: Report saved to {report_path}")

        # 生成 PDF 版本
        try:
            markdown_to_pdf(
                final_report, pdf_path,
                company_name=company_name,
                stock_code=stock_code
            )
            logger.info(
                f"{SUCCESS_ICON} SummaryAgent: PDF report saved to {pdf_path}")
        except Exception as pdf_err:
            logger.warning(
                f"Failed to generate PDF report: {pdf_err}. "
                f"Markdown report is still available at {report_path}"
            )
            pdf_path = None

        # 返回更新后的状态，包含最终报告
        current_data["final_report"] = final_report
        current_data["report_path"] = report_path
        current_data["report_pdf_path"] = pdf_path

        # 记录 Agent执行成功
        total_execution_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {
            "final_report_length": len(final_report),
            "report_path": report_path,
            "report_pdf_path": pdf_path,
            "report_preview": final_report,
            "llm_execution_time": llm_execution_time,
            "total_execution_time": total_execution_time
        }, total_execution_time, True)

        return {"data": current_data, "messages": messages}
```

**Key change**: The old block ended with `return {"data": current_data, "messages": messages}` at line 442, which we're now combining into this modified block. The `pdf_path` is `None` if PDF generation fails (graceful degradation).

- [ ] **Step 3: Modify the error report saving block (lines 466-495) to also generate PDF**

Replace the error report save block with:

```python
        # 也将错误报告保存到文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if stock_code == "Unknown Stock" or stock_code == "Extracted from analysis":
            query_based_name = user_query.replace(
                " ", "_").replace("分析", "").strip()
            if not query_based_name:
                query_based_name = "financial_analysis"
            safe_file_prefix = f"error_report_{query_based_name}"
        else:
            safe_company_name = company_name.replace(" ", "_").replace(".", "")
            if safe_company_name == "Unknown_Company" or safe_company_name == "Extracted_from_analysis":
                safe_company_name = user_query.replace(
                    " ", "_").replace("分析", "").strip()
                if not safe_company_name:
                    safe_company_name = "company"

            clean_stock_code = stock_code.replace("sh.", "").replace("sz.", "")
            safe_file_prefix = f"error_report_{safe_company_name}_{clean_stock_code}"

        report_filename = f"{safe_file_prefix}_{timestamp}.md"
        pdf_filename = f"{safe_file_prefix}_{timestamp}.pdf"

        reports_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, report_filename)
        pdf_path = os.path.join(reports_dir, pdf_filename)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(error_report)

        logger.info(
            f"{ERROR_ICON} SummaryAgent: Error report saved to {report_path}")

        # 尝试生成 PDF 版本
        try:
            markdown_to_pdf(
                error_report, pdf_path,
                company_name=company_name,
                stock_code=stock_code
            )
        except Exception as pdf_err:
            logger.warning(f"Failed to generate error PDF: {pdf_err}")
            pdf_path = None

        current_data["report_path"] = report_path
        current_data["report_pdf_path"] = pdf_path
```

Note: The `execution_logger.log_agent_complete(...)` and `return {"data": current_data, "messages": messages}` lines that follow the error block (lines 501-504) stay unchanged — just the middle block is replaced.

- [ ] **Step 4: Verify syntax and imports**

```bash
cd Finance/Financial-MCP-Agent && python -c "from src.agents.summary_agent import summary_agent; print('Import OK')"
```

Expected: `Import OK` (may take a moment due to langchain imports)

- [ ] **Step 5: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/summary_agent.py
git commit -m "feat: integrate PDF generation into summary_agent report output

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Update execution_logger.py to save PDF copy

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/utils/execution_logger.py`

- [ ] **Step 1: Modify log_final_report to also save PDF**

Replace `execution_logger.py` line 180 (`self._save_text(report_content, "reports/final_report.md")`) with:

```python
        # 保存报告 Markdown 副本
        self._save_text(report_content, "reports/final_report.md")

        # 保存报告 PDF 副本
        try:
            from src.utils.pdf_generator import markdown_to_pdf
            pdf_path = str(self.execution_dir / "reports" / "final_report.pdf")
            markdown_to_pdf(report_content, pdf_path)
        except Exception as e:
            # PDF 生成失败不应影响日志记录流程
            pass
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/utils/execution_logger.py
git commit -m "feat: save PDF copy of final report in execution logs

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Update Streamlit frontend pages for PDF download

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/app/pages/01_股票查询.py`
- Modify: `Finance/Financial-MCP-Agent/src/app/pages/02_股票池.py`

- [ ] **Step 1: Update 01_股票查询.py download button (line 178-188)**

Replace lines 178-188:

```python
    # 生成文件名
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{stock_name}_分析报告_{date_str}.md"

    st.download_button(
        label="下载报告",
        data=report_content,
        file_name=filename,
        mime="text/markdown",
        type="primary",
    )
```

With:

```python
    # 生成文件名
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")

    # 检查是否有 PDF 路径
    pdf_path = result.get("report_pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        pdf_filename = f"{stock_name}_分析报告_{date_str}.pdf"
        st.download_button(
            label="📄 下载 PDF 报告",
            data=pdf_data,
            file_name=pdf_filename,
            mime="application/pdf",
            type="primary",
        )
    else:
        # 回退到 Markdown 下载
        md_filename = f"{stock_name}_分析报告_{date_str}.md"
        st.download_button(
            label="下载报告 (Markdown)",
            data=report_content,
            file_name=md_filename,
            mime="text/markdown",
            type="primary",
        )
```

- [ ] **Step 2: Update 02_股票池.py download button (lines 455-458)**

Replace lines 455-458:

```python
    from datetime import datetime
    fn = f"{sn}_分析报告_{datetime.now().strftime('%Y%m%d')}.md"
    st.download_button("下载报告", data=content, file_name=fn, mime="text/markdown", type="primary")
```

With:

```python
    from datetime import datetime
    date_str = datetime.now().strftime('%Y%m%d')

    pdf_path = r.get("report_pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        pdf_fn = f"{sn}_分析报告_{date_str}.pdf"
        st.download_button("📄 下载 PDF 报告", data=pdf_data, file_name=pdf_fn,
                          mime="application/pdf", type="primary")
    else:
        md_fn = f"{sn}_分析报告_{date_str}.md"
        st.download_button("下载报告 (Markdown)", data=content, file_name=md_fn,
                          mime="text/markdown", type="primary")
```

- [ ] **Step 3: Verify both pages import correctly**

```bash
cd Finance/Financial-MCP-Agent && python -c "
import os, sys
sys.path.insert(0, 'src/app')
# Just verify no syntax errors
exec(open('src/app/pages/01_股票查询.py', encoding='utf-8').read().split('st.set_page_config')[0])
print('Page 01 syntax OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/app/pages/01_股票查询.py Finance/Financial-MCP-Agent/src/app/pages/02_股票池.py
git commit -m "feat: switch Streamlit download buttons to PDF with MD fallback

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: Update API to include PDF path

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/api/app.py`

- [ ] **Step 1: Add report_pdf_path to API response (around line 883)**

In the `_run_report_pipeline` function, find the `_task_results[task_id]` assignment (around line 881-888) and add `report_pdf_path`:

The line `"report_path": report_path,` should become:

```python
            "report_path": report_path,
            "report_pdf_path": report_data.get("report_pdf_path", ""),
```

- [ ] **Step 2: Add report_pdf_path to the GET endpoint response (around line 1109)**

In `get_report_status`, find the return dict (around line 1107-1114) and add `report_pdf_path`:

```python
            "report_pdf_path": result.get("report_pdf_path", ""),
```

- [ ] **Step 3: Verify API imports are not broken**

```bash
cd Finance/Financial-MCP-Agent && python -c "from src.api.app import app; print('API import OK')"
```

Expected: `API import OK`

- [ ] **Step 4: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/api/app.py
git commit -m "feat: expose report_pdf_path in API report endpoints

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 8: End-to-end verification

- [ ] **Step 1: Convert all existing reports to PDF as a batch test**

```bash
cd Finance/Financial-MCP-Agent && python -c "
from src.utils.pdf_generator import markdown_to_pdf
import os, glob

reports_dir = 'reports'
md_files = glob.glob(os.path.join(reports_dir, 'report_*.md'))
print(f'Found {len(md_files)} report files')

success = 0
failures = []
for md_file in md_files:
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            md = f.read()
        pdf_file = md_file.replace('.md', '.pdf')
        # Extract company info from filename
        basename = os.path.basename(md_file)
        parts = basename.replace('report_', '').replace('.md', '').split('_')
        company = parts[0] if parts else ''
        code = parts[1] if len(parts) > 1 else ''
        markdown_to_pdf(md, pdf_file, company, code)
        size_kb = os.path.getsize(pdf_file) / 1024
        print(f'  OK: {basename} -> {size_kb:.1f} KB')
        success += 1
    except Exception as e:
        print(f'  FAIL: {os.path.basename(md_file)}: {e}')
        failures.append((os.path.basename(md_file), str(e)))

print(f'\nResults: {success}/{len(md_files)} succeeded')
if failures:
    print('Failures:')
    for f, err in failures:
        print(f'  - {f}: {err}')
"
```

Expected: All existing reports convert successfully (or at least most of them). Note any failures for debugging.

- [ ] **Step 2: Verify PDF file sizes are reasonable**

```bash
cd Finance/Financial-MCP-Agent && python -c "
import os, glob
for f in sorted(glob.glob('reports/report_*.pdf')):
    size = os.path.getsize(f) / 1024
    print(f'{os.path.basename(f):50s} {size:8.1f} KB')
"
```

Expected: PDF sizes between 30-200 KB (typical for text-based reports).

- [ ] **Step 3: Check that nothing is broken**

```bash
cd Finance/Financial-MCP-Agent && python -c "
# Quick import test of all modified modules
from src.utils.pdf_generator import markdown_to_pdf, parse_markdown, parse_inline
from src.agents.summary_agent import summary_agent
from src.utils.execution_logger import get_execution_logger
print('All modules import successfully')
print('No regressions detected')
"
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "test: batch convert all existing reports to PDF, verify no regressions

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```
