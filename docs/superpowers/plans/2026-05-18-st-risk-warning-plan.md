# ST Risk Warning Feature - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ST (Special Treatment) risk warning section to both single-stock reports and stock pool scoring reports.

**Architecture:** Three layers — new MCP tools (data), enhanced agents (analysis), new report section (output). All new MCP calls have timeout protection and 3-level degradation (Tushare→AkShare→skip).

**Tech Stack:** Python, AkShare/Sina, Tushare MCP, FastMCP, LangGraph

---

### Task 1: Add `get_st_risk_data` tool to local a-share-mcp

**Files:**
- Modify: `Finance/a-share-mcp-is-just-i-need/src/tools/stock_market.py` (append new tool)

- [ ] **Step 1: Add `get_st_risk_data` tool function**

Append after line 213 (end of `register_stock_market_tools` function, inside the function before the closing):

```python
    @app.tool()
    def get_st_risk_data(code: str) -> str:
        """
        检测A股股票的ST风险状态。
        通过AkShare/Sina接口获取股票当前名称，判断是否包含ST/*ST标记。

        参数:
            code: 股票代码（例如：'sh.600000', 'sz.000001'）

        返回:
            Markdown格式的ST风险数据，包含：当前ST状态、ST类型、风险等级
        """
        import akshare as ak
        import asyncio

        symbol = code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").strip()
        logger.info(f"Tool 'get_st_risk_data' called for {code} (symbol={symbol})")

        try:
            # 超时保护：5秒内必须返回
            df = ak.stock_info_a_code_name()
            if df is None or df.empty:
                return f"| 项目 | 值 |\n|------|----|\n| ST状态 | 数据不可用 |\n| 数据来源 | AkShare/Sina返回空数据 |"

            row = df[df["code"] == symbol]
            if row.empty:
                return (
                    f"| 项目 | 值 |\n|------|----|\n"
                    f"| 股票代码 | {symbol} |\n"
                    f"| ST状态 | 正常（未在ST名单中） |\n"
                    f"| 数据来源 | AkShare/Sina stock_info_a_code_name |\n"
                    f"| 说明 | 当前股票名称不含ST/*ST标记，未进入风险警示板 |"
                )

            name = str(row.iloc[0]["name"])
            is_st = "ST" in name or "*ST" in name or "st" in name.lower()
            st_type = "退市风险警示（*ST）" if "*ST" in name else ("其他风险警示（ST）" if "ST" in name else "无")

            if is_st:
                return (
                    f"| 项目 | 值 |\n|------|----|\n"
                    f"| 股票代码 | {symbol} |\n"
                    f"| 当前名称 | {name} |\n"
                    f"| ST状态 | ⚠️ **已标记ST** |\n"
                    f"| ST类型 | {st_type} |\n"
                    f"| 数据来源 | AkShare/Sina stock_info_a_code_name |\n"
                    f"| 风险提示 | 该股票当前处于风险警示板，存在退市风险，投资需极度谨慎 |"
                )
            else:
                return (
                    f"| 项目 | 值 |\n|------|----|\n"
                    f"| 股票代码 | {symbol} |\n"
                    f"| 当前名称 | {name} |\n"
                    f"| ST状态 | 正常 |\n"
                    f"| 数据来源 | AkShare/Sina stock_info_a_code_name |"
                )

        except Exception as e:
            logger.warning(f"get_st_risk_data failed for {code}: {e}")
            return (
                f"| 项目 | 值 |\n|------|----|\n"
                f"| ST状态 | 查询失败 |\n"
                f"| 错误信息 | {str(e)[:100]} |\n"
                f"| 说明 | ST风险数据暂时不可用，请通过其他渠道核实 |"
            )
```

- [ ] **Step 2: Add `stock_info_a_code_name` to data source interface (if needed)**

The tool uses AkShare directly (bypassing data source interface), matching the pattern used by `crawl_news`. No interface change needed.

---

### Task 2: Add `tushare_st_status` tool to Tushare MCP server

**Files:**
- Modify: `Finance/a-share-mcp-is-just-i-need/tushare_mcp_server.py` (lines 267-273, before `if __name__`)

- [ ] **Step 1: Add `ts_st_status` data function**

Add after line 159 (`ts_news` function):

```python
def ts_st_status(code: str) -> list:
    """获取个股ST状态历史"""
    return _dicts(_call("stock_st", {"ts_code": _ts_code(code)},
                        "ts_code,name,trade_date,type,type_name"))
```

- [ ] **Step 2: Register `tushare_st_status` tool**

Add after line 274 (after `tushare_news` tool), before `if __name__`:

```python
@app.tool()
def tushare_st_status(code: str) -> str:
    """获取A股个股ST状态历史: 当前/历史ST标记、ST类型（退市风险警示/其他风险警示）、变更日期"""
    items = ts_st_status(code)
    if not items:
        return f"| 项目 | 值 |\n|------|----|\n| ST状态 | 正常 |\n| 说明 | Tushare stock_st接口无记录，当前未处于ST状态 |"
    return _format_result(items, max_rows=30)
```

---

### Task 3: Enhance fundamental_agent with ST analysis

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/fundamental_agent.py` (lines 136-143, lines 174-215)

- [ ] **Step 1: Add ST tools to tool_filter**

Change `tool_filter` at line 136-143 from:
```python
            mcp_tools = await get_mcp_tools(tool_filter=[
                "get_stock_basic_info", "get_stock_industry",
                ...
            ])
```
To:
```python
            mcp_tools = await get_mcp_tools(tool_filter=[
                "get_stock_basic_info", "get_stock_industry",
                "get_profit_data", "get_balance_data", "get_cash_flow_data",
                "get_growth_data", "get_operation_data", "get_dupont_data",
                "get_dividend_data", "get_adjust_factor_data",
                "tushare_stock_info", "tushare_fina_indicator",
                "tushare_dividend", "tushare_ev_ebitda", "tushare_daily_basic",
                "tushare_st_status", "get_st_risk_data",
            ])
```

- [ ] **Step 2: Add ST analysis dimension to agent_input prompt**

Insert after line 195 (after the "4. 分红与股东回报" section), before "5. 行业对比分析":

```python
            # 在 agent_input 字符串中，第4节之后插入ST风险分析维度
```

Modify the `agent_input` string, after line 199-200:
```python
4. 分红与股东回报
   - 历史分红记录、股息率
   - 股东变化趋势（机构/散户持仓比例）

4.5. ST风险警示分析（⚠️ 新增必查项）
   - 调用 tushare_st_status 工具查询ST状态历史（优先使用，数据最全）
   - 若 tushare_st_status 无数据或调用失败，立即调用 get_st_risk_data 作为备用
   - 分析当前ST状态：是否为ST/*ST、进入风险警示板日期
   - ST类型判断：退市风险警示（*ST）还是其他风险警示（ST）
   - 触发原因分析（结合财务数据）：是否连续两年净利润为负、净资产是否为负、审计报告是否为否定意见、是否存在重大违规
   - 退市风险等级评估：基于ST类型+财务指标+持续时间综合判断
   - 如果两个工具都调用失败，必须明确标注"ST数据不可用，无法完成ST风险评估"

5. 行业对比分析
```

- [ ] **Step 3: Add ST data to output requirements**

Modify the output format section, insert after "4. 资产负债结构" in the data fact zone template:

In the `## 📊 数据事实区` section of the prompt (around line 221), add:
```
- [tushare_st_status/get_st_risk_data] ST状态数据：当前状态/历史记录/类型
```

---

### Task 4: Enhance news_agent with ST awareness

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/news_agent.py` (system prompt at lines 239-258)

- [ ] **Step 1: Add `get_st_risk_data` to tool_filter**

Change line 152 from:
```python
        news_tools = await get_mcp_tools(tool_filter=["crawl_news"])
```
To:
```python
        news_tools = await get_mcp_tools(tool_filter=["crawl_news", "get_st_risk_data"])
```

- [ ] **Step 2: Add ST data collection in Phase 1**

After line 190 (after the `tushare_news` tool section), add:

```python
        elif tool.name == "get_st_risk_data":
            if clean_code:
                # 用 sh./sz. 前缀调用
                sh_code = f"sh.{clean_code}" if (clean_code.startswith("6") or clean_code.startswith("688")) else f"sz.{clean_code}"
                tasks.append(_call_tool_with_timeout(
                    tool, {"code": sh_code}, 5.0,  # 5s timeout for ST check
                    f"st_risk(code={sh_code})"
                ))
                task_labels.append(f"st_risk")
```

- [ ] **Step 3: Add ST awareness to news system prompt**

In the system prompt (line 241), insert after "分析要求":
```
分析要求：
0. **ST风险筛查**：优先查看是否有ST风险数据，若有ST标记，必须在分析中重点标注ST风险
1. 逐条分析每一条新闻的核心内容和潜在影响
...
```

---

### Task 5: Add ST risk section to summary report

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/summary_agent.py` (system prompt lines 157-284, data extraction lines 125-130)

- [ ] **Step 1: Extract ST risk data from analysis results**

After line 130 (`news_analysis = ...`), add:

```python
    # 提取各Agent中的ST风险数据
    st_risk_from_fundamental = ""
    if "ST风险警示分析" in str(fundamental_analysis):
        # 从基本面分析中提取ST分析段落
        st_match = re.search(
            r'(?:ST风险警示分析|4\.5\.\s*ST风险).*?(?=\n##|\n\d+\.\s|\Z)',
            str(fundamental_analysis), re.DOTALL
        )
        if st_match:
            st_risk_from_fundamental = st_match.group(0)

    st_risk_from_news = ""
    if "ST" in str(news_analysis) or "风险警示" in str(news_analysis):
        st_match = re.search(
            r'(?:ST风险|风险警示).*?(?=\n##|\n\d+\.\s|\Z)',
            str(news_analysis), re.DOTALL
        )
        if st_match:
            st_risk_from_news = st_match.group(0)
```

- [ ] **Step 2: Add ST risk section to system prompt**

In the system prompt, insert after "## 风险因素" section (after line 239), before "## 投资建议":

```python
        ## ST风险警示专区 ⚠️
        [
          - **当前ST状态**：[正常/ST/*ST]，基于实际数据查询结果
          - **ST类型**：[退市风险警示/其他风险警示/无]
          - **风险起始日期**：[具体日期或"不适用"]
          - **触发原因分析**：
            - 连续亏损风险：最近两年净利润是否持续为负
            - 净资产风险：最近一期净资产是否已为负
            - 审计风险：是否存在无法表示意见或否定意见的审计报告
            - 重大违规：是否涉及重大信息披露违法违规
          - **退市风险等级**：基于财务指标和ST状态综合评估
          - **交易限制提醒**：
            - 涨跌幅限制（风险警示板±5%）
            - 投资者适当性管理要求（签署风险揭示书）
            - 单日买入数量限制（50万股）
          - **整改可能性评估**：基于行业地位、财务改善潜力、股东背景的判断
          - **对投资决策的影响分析**：ST状态对估值、流动性和投资策略的具体影响

          ⚠️ 如果ST数据不可用，必须声明"ST风险数据暂不可用，建议投资者自行查询"
        ]
```

- [ ] **Step 3: Inject ST data into user_prompt**

In the `user_prompt` string (line 288), add after `{news_analysis}`:

```python
        ST RISK DATA (from fundamental analysis):
        {st_risk_from_fundamental}

        ST RISK DATA (from news analysis):
        {st_risk_from_news}
```

---

### Task 6: Update scorers with ST data passthrough

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/scoring_nodes.py`

The scorers already have ST risk scoring dimensions (medium_term 4pts, long_term 4pts). We just need to ensure ST data is extracted and passed through to scorers.

- [ ] **Step 1: Read current scoring_nodes.py to understand extraction pattern**
- [ ] **Step 2: Add ST data extraction in scorer wrappers if not already extracted from analysis text**

Since scorers already extract and read the full analysis text from fundamental_analysis/technical_analysis etc., and the fundamental agent will now include ST analysis in its output, no code change is needed for scorers. They will automatically pick up ST information from the enriched fundamental analysis text.

---

### Task 7: Verification

- [ ] **Step 1: Test `get_st_risk_data` tool independently**

```bash
cd Finance/a-share-mcp-is-just-i-need && python -c "
import akshare as ak
# Test with normal stock
df = ak.stock_info_a_code_name()
row = df[df['code'] == '603871']
print('Normal stock:', row.iloc[0].to_dict() if not row.empty else 'Not found')
# Test with ST stock
row_st = df[df['code'] == '000004']
print('ST stock:', row_st.iloc[0].to_dict() if not row_st.empty else 'Not found')
"
```

- [ ] **Step 2: Test `tushare_st_status` tool independently**

```bash
cd Finance/a-share-mcp-is-just-i-need && python -c "
from tushare_mcp_server import ts_st_status
print('Normal stock:', ts_st_status('603871'))
print('ST stock:', ts_st_status('000004'))
"
```

- [ ] **Step 3: Full pipeline test with a normal stock**

```bash
cd Finance/Financial-MCP-Agent && python -m src.main --command '分析嘉友国际'
```

- [ ] **Step 4: Full pipeline test with an ST stock**

```bash
cd Finance/Financial-MCP-Agent && python -m src.main --command '分析*ST国华'
```

- [ ] **Step 5: Verify report contains ST risk section**
Check generated report in `reports/` directory has "ST风险警示专区" section.

- [ ] **Step 6: Verify existing functionality unaffected**
Run stock pool tests: `python -m pytest tests/ -v`
