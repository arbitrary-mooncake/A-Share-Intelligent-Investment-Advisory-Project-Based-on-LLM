"""
PDF Report Generator - 将 Markdown 分析报告转换为专业 PDF 格式。
纯 Python 实现，无外部系统依赖（仅需 fpdf2 + 系统中文字体）。
"""
import os
import re
import sys
import uuid
import logging
from fpdf import FPDF, FontFace
from fpdf.enums import AccessPermission, EncryptionMethod

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 字体发现
# ═══════════════════════════════════════════════════════════════

def _find_chinese_font():
    """在系统中查找可用的中文字体 TTF/TTC 文件。"""
    if os.name == 'nt':
        candidates = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simkai.ttf",
        ]
    elif sys.platform == 'darwin':
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            logger.info(f"Found Chinese font: {path}")
            return path

    logger.warning("No Chinese font found on system")
    return None


def _find_body_font():
    """查找适合正文的宋体/楷体。找不到则回退到黑体。"""
    if os.name == 'nt':
        candidates = [
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simkai.ttf",
            "C:/Windows/Fonts/simhei.ttf",
        ]
    else:
        return _find_chinese_font()

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
    """判断一行是否属于特殊块的起始行"""
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

    支持的块类型：标题(#)、分隔线(---)、表格(|...|)、列表(-/*/1.)、普通段落。
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
            # 去除标题中的 **粗体** 标记
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
        if stripped.startswith('|') and stripped.endswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                table_lines.append(lines[i].strip())
                i += 1

            rows = []
            for tl in table_lines:
                cells = [c.strip() for c in tl[1:-1].split('|')]
                # 跳过分隔行 (如 |---|---|)
                if all(re.match(r'^[-:]+$', c) for c in cells):
                    continue
                # 清理单元格内的粗体标记
                cells = [re.sub(r'\*\*([^*]+)\*\*', r'\1', c) for c in cells]
                rows.append(cells)

            if rows:
                if len(rows) >= 2:
                    blocks.append(TableBlock(rows[0], rows[1:]))
                else:
                    blocks.append(TableBlock(rows[0], []))
            continue

        # 表格（备选格式）：含 | 但不以 | 开头，需至少两行确认
        if '|' in stripped and not stripped.startswith('|'):
            if i + 1 < len(lines) and '|' in lines[i + 1].strip():
                # 先收集不移动指针
                peek_i = i
                table_lines = []
                while peek_i < len(lines) and '|' in lines[peek_i].strip():
                    table_lines.append(lines[peek_i].strip())
                    peek_i += 1

                rows = []
                for tl in table_lines:
                    cells = [c.strip() for c in tl.split('|')]
                    if all(re.match(r'^[-:]+$', c) for c in cells):
                        continue
                    cells = [re.sub(r'\*\*([^*]+)\*\*', r'\1', c) for c in cells]
                    rows.append(cells)

                if len(rows) >= 2:
                    blocks.append(TableBlock(rows[0], rows[1:]))
                    i = peek_i
                    continue
            # 单行管道符或解析失败，不作表格处理，继续往下走段落逻辑

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
    NAVY        = (26, 58, 92)
    DARK_GRAY   = (51, 51, 51)
    MED_GRAY    = (128, 128, 128)
    LIGHT_GRAY  = (240, 242, 245)
    SEPARATOR   = (190, 200, 215)
    WHITE       = (255, 255, 255)

    DATA_TAG_COLOR    = (30, 90, 190)
    JUDGMENT_TAG_COLOR = (200, 130, 20)

    def __init__(self, heading_font_path: str, body_font_path: str,
                 company_name: str = "", stock_code: str = ""):
        super().__init__('P', 'mm', 'A4')
        self._heading_font = heading_font_path
        self._body_font = body_font_path
        self.company_name = company_name
        self.stock_code = stock_code

        # 注册字体
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

        self.set_draw_color(*self.SEPARATOR)
        self.set_line_width(0.2)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(3)

        self.set_font('Body', '', 7.5)
        self.set_text_color(*self.MED_GRAY)
        self.cell(0, 5, f'第 {self.page_no()} 页', align='C')
        self.ln(3.5)

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
            self.ln(12)
            self.set_font('Heading', '', 22)
            self.set_text_color(*self.NAVY)
            self.multi_cell(0, 10, block.text, align='C')
            self.ln(3)
            self.set_draw_color(*self.NAVY)
            self.set_line_width(0.6)
            y = self.get_y()
            x_center = self.w / 2
            self.line(x_center - 35, y, x_center + 35, y)
            self.ln(10)
        elif block.level == 2:
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
            self.ln(5)
            self.set_font('Heading', '', 11.5)
            self.set_text_color(*self.NAVY)
            self.cell(0, 7, block.text)
            self.ln(8)
        else:
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
            is_bold = style.get('b', False)
            is_tag_data = style.get('tag_data', False)
            is_tag_judgment = style.get('tag_judgment', False)

            if is_tag_data:
                self._write_tag('data', self.DATA_TAG_COLOR)
            elif is_tag_judgment:
                self._write_tag('judgment', self.JUDGMENT_TAG_COLOR)

            if is_bold:
                self.set_font('Heading', '', 10)
                self.set_text_color(*self.NAVY)
            else:
                self.set_font('Body', '', 10)
                self.set_text_color(*self.DARK_GRAY)

            self.write(line_height, text)

        self.ln(line_height + 1.5)

    def _write_tag(self, tag_type: str, color: tuple):
        """输出彩色加粗的标签文本 [数据] 或 [判断]"""
        self.set_font('Heading', '', 9.5)
        self.set_text_color(*color)
        label = '[数据]' if tag_type == 'data' else '[判断]'
        self.write(5.6, label)
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
        col_widths = [available_w / col_count] * col_count

        min_col_w = 18
        for i, w in enumerate(col_widths):
            if w < min_col_w:
                col_widths[i] = min_col_w

        total_w = sum(col_widths)
        if total_w > available_w:
            scale = available_w / total_w
            col_widths = [w * scale for w in col_widths]

        with self.table(col_widths=col_widths, text_align='CENTER',
                        first_row_as_headings=False) as table:
            # 表头行：深蓝底色白字
            header_face = FontFace(family='Heading', size_pt=8.5,
                                   fill_color=self.NAVY, color=self.WHITE)
            header_row = table.row()
            for h in block.headers:
                header_row.cell(h, style=header_face)

            # 数据行：交替底色
            for i, row in enumerate(block.rows):
                bg = self.LIGHT_GRAY if i % 2 == 0 else self.WHITE
                data_face = FontFace(family='Body', size_pt=8.5,
                                     fill_color=bg, color=self.DARK_GRAY)

                data_row = table.row()
                for j, cell_text in enumerate(row):
                    is_numeric = bool(re.match(r'^[\d.,\-+%]+$', cell_text.strip()))
                    align = 'CENTER' if is_numeric else 'LEFT'
                    data_row.cell(cell_text, style=data_face, align=align)

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
            prefix = f"{idx + 1}. " if block.ordered else "- "

            indent = 6
            self.set_x(self.l_margin + indent)
            self.cell(5, 5.5, prefix)

            segments = parse_inline(item)
            for text, style in segments:
                is_bold = style.get('b', False)
                is_tag_data = style.get('tag_data', False)
                is_tag_judgment = style.get('tag_judgment', False)

                if is_tag_data:
                    self._write_tag('data', self.DATA_TAG_COLOR)
                elif is_tag_judgment:
                    self._write_tag('judgment', self.JUDGMENT_TAG_COLOR)

                if is_bold:
                    self.set_font('Heading', '', 10)
                    self.set_text_color(*self.NAVY)
                else:
                    self.set_font('Body', '', 10)
                    self.set_text_color(*self.DARK_GRAY)

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

    # 加密PDF，限制编辑权限：仅允许打印和复制，禁止修改/注释/填表/组合
    pdf.set_encryption(
        owner_password=str(uuid.uuid4()),
        user_password=None,
        encryption_method=EncryptionMethod.RC4,
        permissions=(AccessPermission.PRINT_LOW_RES | AccessPermission.PRINT_HIGH_RES
                     | AccessPermission.COPY),
        encrypt_metadata=True,
    )

    pdf.render_blocks(blocks)

    # 确保输出目录存在
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pdf.output(output_path)

    logger.info(f"PDF report saved to {output_path} ({pdf.page_no()} pages)")
    return output_path
