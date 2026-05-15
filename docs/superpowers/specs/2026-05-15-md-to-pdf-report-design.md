# MD to PDF Report Conversion Design

**Date**: 2026-05-15
**Status**: Approved

## Goal

Replace `.md` report output with properly formatted `.pdf` reports. LLM continues outputting Markdown — a new conversion layer renders it to PDF with professional financial report styling.

## Technical Decision: fpdf2

- Pure Python, no system dependencies, works on Windows/Linux
- MD→PDF conversion <0.1s per report (negligible vs 60-300s LLM time)
- Chinese font: SimHei/微软雅黑 from system, or bundled Noto Sans CJK fallback

## New Files

### `src/utils/pdf_generator.py`
- `MarkdownToPDF` class: parses MD line-by-line, renders to A4 PDF
- Element types handled: h1/h2/h3 (`#`/`##`/`###`), bold (`**`), italic (`*`), horizontal rules (`---`), tables (`|...|`), lists (`-`/`*`), paragraphs
- `[数据]` and `[判断]` tags rendered as small colored badges (blue/orange)
- Professional styling: dark navy headings, gray rules, alternating table rows, header (company+code), footer (page number + disclaimer)
- Chinese font registration with automatic fallback

## Modified Files

| File | Change |
|------|--------|
| `summary_agent.py:391-426` | After saving `.md`, also call PDF generator, save `.pdf` to reports/ |
| `summary_agent.py:466-495` | Same for error reports |
| `execution_logger.py:180` | Save `final_report.pdf` alongside `final_report.md` |
| `01_股票查询.py:170-193` | Download button → PDF (MIME `application/pdf`), preview stays as `st.markdown()` |
| `02_股票池.py:451-463` | Same as above |
| `requirements.txt` | Add `fpdf2>=2.8.0` |

## Unchanged

- All agent prompts and logic
- Scoring engine / stock pool manager
- FastAPI endpoints
- Batch scoring

## Edge Cases

1. **Missing Chinese font**: Graceful fallback — log warning, use built-in Helvetica (ASCII-only report)
2. **Malformed markdown table**: Skip bad rows, render what's parseable
3. **Very long report**: fpdf2 auto-paginates, no special handling needed
4. **Error during PDF generation**: Log error, still save `.md` as fallback, don't crash
