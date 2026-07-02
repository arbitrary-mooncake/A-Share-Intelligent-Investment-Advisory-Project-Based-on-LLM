# QA L1 快问快答加速优化

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 L1 "快问快答" 数据获取时间从 15-60s 降到 3-8s，同时确保 L2 门槛足够低（稍有难度的问题就路由到 L2+）。

**Architecture:** 在 task_planner 中为 L1 引入精简工具集（每域1-2个核心工具），在 evidence_assembler 中为 L1 降低超时/重试/跳过托底，在 complexity_analyzer 中收紧 L1 评分上限（24→20），在 app.py lifespan 中预热 MCP 客户端消除首问延迟。

**Tech Stack:** Python 3.11+, asyncio, FastAPI, Streamlit, LangChain MCP Adapters

## Global Constraints

- 零功能删减：L2/L3/L4 行为完全不变
- 不影响股票查询、股票池、基金专区等其他功能
- 所有现有缓存逻辑（per-session + global）保持不变
- L1 只缩减工具数量和超时，不改变 LLM 调用逻辑
- MCP 预热在后台进行，不阻塞 FastAPI 启动

---

### Task 1: complexity_analyzer.py — 收紧 L1 评分上限

**Files:**
- Modify: `src/qa/complexity_analyzer.py:231-238`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: L1/L2 分界线从 25 降到 21（即 L1: 0-20, L2: 21-49）

**Goal:** 让评分在 21-24 之间的问题（如"茅台最近走势如何"评分~18-23）路由到 L2，只有真正的简单查询留在 L1。

- [ ] **Step 1: 修改 L1 评分上限**

将 `_score_question()` 中的 L1 上限从 24 改为 20：

```python
# 分级 (line ~231-238)
if score <= 20:          # 原来是 24
    level = "L1"
elif score <= 49:
    level = "L2"
elif score <= 69:
    level = "L3"
else:
    level = "L4"
```

- [ ] **Step 2: 验证修改正确性**

运行已有测试确认无回归：

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/ -v -k "qa" 2>&1 | head -50
```

- [ ] **Step 3: Commit**

```bash
git add src/qa/complexity_analyzer.py
git commit -m "fix: tighten L1 score cap from 24 to 20 for faster routing to L2

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: task_planner.py — L1 精简工具集

**Files:**
- Modify: `src/qa/task_planner.py:241-318` (DATA_DOMAINS), `src/qa/task_planner.py:331-401` (plan_task)

**Interfaces:**
- Consumes: nothing (standalone change)
- Produces: `L1_LITE_TOOLS: Dict[str, List[str]]` — L1 每域精简约 60% 工具；`plan_task()` 新增 `complexity_level` 感知，L1 返回精简工具集且不自动追加估值域和代表性股票

**Goal:** L1 只使用每域 1-2 个核心工具，大幅减少并行工具调用数。

- [ ] **Step 1: 在 DATA_DOMAINS 后新增 L1_LITE_TOOLS 映射**

在 `DATA_DOMAINS` 定义后（第 318 行后）新增：

```python
# L1 精简工具集：每域只保留 1-2 个核心工具，大幅减少数据获取时间
L1_LITE_TOOLS: Dict[str, List[str]] = {
    "行情": ["tushare_kline", "tushare_daily_basic"],
    "估值": ["tushare_pe_percentile"],
    "财务": ["tushare_fina_indicator"],
    "资金": ["tushare_moneyflow"],
    "行业": ["tushare_stock_info"],
    "板块": ["tushare_concept_list"],
    "新闻": ["tushare_news"],
    "宏观": ["tushare_cn_cpi", "tushare_cn_pmi"],
    "国际": ["web_search"],
}
```

- [ ] **Step 2: 修改 plan_task() 根据复杂度返回不同工具集**

修改 `plan_task()` 函数签名和逻辑（第 331-401 行），在工具选择部分添加 L1 判断：

```python
def plan_task(question: str, complexity_level: str, history_text: str = "",
              topic_name: str = "", stock_code: str = "", company_name: str = "") -> TaskPlan:
    """
    根据问题内容和复杂度等级规划数据获取任务。

    L1: 精简工具集（每域1-2核心工具），不限并发，无兜底
    L2: 全量工具，并行拉取
    L3/L4: 全量工具，可选ReAct
    """
    # L0: 无需数据，直接回答
    if complexity_level == "L0":
        return TaskPlan(
            domains=[], tools=[], need_react=False,
            reason="L0: 无需数据，直接回答",
            expected_data_volume="small",
        )

    # 结合历史对话提取额外关键词
    augmented_question = question
    if history_text:
        recent_kw = _extract_domain_keywords_from_text(history_text)
        if recent_kw:
            augmented_question = question + " " + " ".join(recent_kw)

    domains = _identify_domains(augmented_question)

    # 兜底：有股票代码/名称但无数据域匹配时，默认查询行情
    if not domains and (stock_code or company_name):
        domains = ["行情"]
        # L1 不自动追加估值域（L2+才追加）
        if complexity_level not in ("L0", "L1"):
            pure_code_match = re.search(r'(?<!\d)(\d{5,6})(?!\d)', question)
            if pure_code_match and not company_name:
                domains = ["行情", "估值"]

    # 主题/宏观问题自动追加新闻域和板块域（L2+ 才追加额外域）
    if topic_name and complexity_level not in ("L0", "L1"):
        if "新闻" not in domains:
            domains.append("新闻")
        if "板块" not in domains:
            domains.append("板块")
        macro_topics = {"黄金", "白银", "原油", "煤炭", "房地产"}
        if topic_name in macro_topics and "宏观" not in domains:
            domains.append("宏观")
        if topic_name in macro_topics and "国际" not in domains:
            domains.append("国际")

    # L1 使用精简工具集，L2+ 使用全量工具
    if complexity_level == "L1":
        tools = _get_l1_tools_for_domains(domains)
    else:
        tools = _get_tools_for_domains(domains)

    # 全部复杂度统一使用两阶段快路径（并行拉取+单次LLM），不走ReAct
    need_react = False

    data_volume = "small"
    if len(domains) >= 4:
        data_volume = "large"
    elif len(domains) >= 2:
        data_volume = "medium"

    reason = f"问题涉及{len(domains)}个数据域({', '.join(domains)})，复杂度={complexity_level}"

    return TaskPlan(
        domains=domains,
        tools=tools,
        need_react=need_react,
        reason=reason,
        expected_data_volume=data_volume,
    )
```

- [ ] **Step 3: 新增 _get_l1_tools_for_domains() 辅助函数**

在 `_get_tools_for_domains()` 后面新增：

```python
def _get_l1_tools_for_domains(domains: List[str]) -> List[str]:
    """L1 精简版：每域只取1-2个核心工具。去重后返回。"""
    tools: Set[str] = set()
    for domain in domains:
        if domain in L1_LITE_TOOLS:
            tools.update(L1_LITE_TOOLS[domain])
        elif domain in DATA_DOMAINS:
            # 新域未在 L1_LITE_TOOLS 中定义时，只取第一个工具作为兜底
            full_tools = DATA_DOMAINS[domain]["tools"]
            if full_tools:
                tools.add(full_tools[0])
    return sorted(tools)
```

- [ ] **Step 4: 验证修改**

```bash
cd Finance/Financial-MCP-Agent && python -c "
from src.qa.task_planner import plan_task, L1_LITE_TOOLS
# L1 简单价格查询应该只有 2-3 个工具
plan = plan_task('茅台今天多少钱', 'L1', stock_code='sh.600519', company_name='贵州茅台')
print(f'L1 工具数: {len(plan.tools)}, 工具: {plan.tools}')
assert len(plan.tools) <= 3, f'L1 工具数应≤3，实际{len(plan.tools)}'

# L2 同样问题应该保留全量工具
plan2 = plan_task('茅台今天多少钱', 'L2', stock_code='sh.600519', company_name='贵州茅台')
print(f'L2 工具数: {len(plan2.tools)}, 工具: {plan2.tools}')
assert len(plan2.tools) >= 4, f'L2 工具数应≥4，实际{len(plan2.tools)}'

print('All assertions passed!')
"
```

- [ ] **Step 5: Commit**

```bash
git add src/qa/task_planner.py
git commit -m "feat: add L1 lite tool set — 60% fewer MCP calls for quick Q&A

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: evidence_assembler.py — L1 快速通道

**Files:**
- Modify: `src/qa/evidence_assembler.py:21-23` (timeout constants), `src/qa/evidence_assembler.py:41-82` (_call_tool_safe), `src/qa/evidence_assembler.py:85-367` (assemble_evidence_fast)

**Interfaces:**
- Consumes: Task 2's L1_LITE_TOOLS and L1 tool selection logic
- Produces: `assemble_evidence_fast()` 新增 `complexity_level` 参数；L1 时 timeout=15s, max_retries=1, 跳过 AkShare 托底和 ETF 直连

**Goal:** L1 工具调用更快失败、更少重试、不做托底，端到端耗时降到 3-8s。

- [ ] **Step 1: 新增 L1 专用超时常量**

在第 21-23 行后新增：

```python
TOOL_TIMEOUT = 30  # 单个工具超时（秒）
MAX_CONCURRENT_TOOLS = 4  # 最大并发工具调用数（防止 Windows 子进程竞争）
REACT_TOTAL_TIMEOUT = 120  # ReAct路径总超时（秒）

# L1 快速通道常量
L1_TOOL_TIMEOUT = 12  # L1 单工具超时（秒），简单查询应快速返回
L1_MAX_RETRIES = 1     # L1 单工具最多重试 1 次
```

- [ ] **Step 2: 修改 _call_tool_safe() 支持 L1 参数**

修改函数签名，新增 `is_l1: bool = False` 参数（第 42 行）：

```python
async def _call_tool_safe(tool, kwargs: dict, timeout: float, label: str,
                         session_id: str = "", max_retries: int = 2,
                         is_l1: bool = False) -> str:
```

在重试逻辑末尾（第 82 行返回 "" 之前），L1 不做 AkShare 无代码标记：

当前代码不需要改 _call_tool_safe 内部逻辑——timeout 和 max_retries 已经由调用方传入。只需要在 assemble_evidence_fast 的 _sem_tool_call 中传入 L1 专用值即可。

- [ ] **Step 3: 修改 assemble_evidence_fast() 签名和 L1 逻辑**

修改函数签名（第 85-94 行），新增 `complexity_level: str = ""` 参数：

```python
async def assemble_evidence_fast(
    stock_code: str,
    company_name: str,
    tools: List[str],
    question: str,
    current_date: str,
    session_id: str = "",
    topic_name: str = "",
    representative_stocks: list = None,
    complexity_level: str = "",
) -> EvidencePackage:
```

在函数体内部添加 L1 判断逻辑。修改第 195-212 行的工具调用部分：

```python
    is_l1 = (complexity_level == "L1")
    tool_timeout = L1_TOOL_TIMEOUT if is_l1 else TOOL_TIMEOUT
    tool_retries = L1_MAX_RETRIES if is_l1 else 2

    logger.info(
        f"{WAIT_ICON} QA Evidence: 并行调用 {len(all_mcp_tools)} 个工具 "
        f"(并发上限={MAX_CONCURRENT_TOOLS}, L1={is_l1}, timeout={tool_timeout}s), "
        f"stock_code={stock_code}..."
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TOOLS)

    async def _sem_tool_call(tool, kwargs, label):
        async with semaphore:
            return await _call_tool_safe(
                tool, kwargs, tool_timeout, label, session_id,
                max_retries=tool_retries, is_l1=is_l1,
            )
```

- [ ] **Step 4: L1 跳过代表性股票拉取和托底逻辑**

修改代表性股票拉取（第 214-230 行），L1 时跳过：

```python
    # 板块/主题查询：L2+ 才拉取代表性个股数据（L1 精简路径跳过）
    if representative_stocks and not is_l1:
        # ... 保持现有代码不变
```

修改 Tushare→AkShare 托底（第 232-259 行），L1 时跳过：

```python
    # 托底：Tushare 工具成功率低于 50% 时，回退到 AkShare（L2+ 专属，L1 跳过）
    if not is_l1:
        total_called = len(labels)
        initial_success = sum(1 for t in results if t)
        if total_called > 0 and initial_success / total_called < 0.5 and stock_code:
            # ... 保持现有托底代码不变
```

修改 ETF 直连托底（第 261-347 行），L1 时跳过：

```python
    # ETF 专属托底（L2+ 专属，L1 跳过）
    _is_etf = stock_code and stock_code.replace("sh.", "").replace("sz.", "").startswith(("51", "58", "15", "16", "18"))
    if _is_etf and stock_code and not is_l1:
        # ... 保持现有 ETF 直连代码不变
```

- [ ] **Step 5: 验证修改**

```bash
cd Finance/Financial-MCP-Agent && python -c "
from src.qa.evidence_assembler import L1_TOOL_TIMEOUT, L1_MAX_RETRIES, TOOL_TIMEOUT
assert L1_TOOL_TIMEOUT == 12
assert L1_MAX_RETRIES == 1
assert TOOL_TIMEOUT == 30
print('Constants OK')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/qa/evidence_assembler.py
git commit -m "feat: add L1 fast path — 12s timeout, 1 retry, skip fallbacks

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: qa_engine.py — 传递复杂度到证据装配 + L1 跳过主题代表性股票

**Files:**
- Modify: `src/qa/qa_engine.py:200-208` (_assemble_with_fallback call), `src/qa/qa_engine.py:85-105` (topic matching)

**Interfaces:**
- Consumes: Task 2 (L1 tool set), Task 3 (complexity_level param)
- Produces: `assemble_evidence_fast()` 调用时传入 `complexity_level`；L1 时 `topic_rep_stocks` 传空列表

**Goal:** 将复杂度等级从编排层传递到证据装配层，L1 跳过代表性股票并行拉取。

- [ ] **Step 1: 修改 _assemble_with_fallback 传递 complexity_level**

在 `_assemble_with_fallback()` 调用处（约第 201-208 行）添加 `complexity_level` 参数：

```python
        evidence_task = asyncio.create_task(
            _assemble_with_fallback(
                task_plan, complexity, stock_code or "", company_name or "",
                question, current_date, current_time_info, actual_session_id,
                topic_name=matched_topic_name,
                representative_stocks=topic_rep_stocks if complexity.level != "L1" else [],
            )
        )
```

- [ ] **Step 2: 修改 _assemble_with_fallback() 函数签名和调用**

修改函数签名（第 432 行），新增 `complexity_level` 参数：

```python
async def _assemble_with_fallback(
    task_plan, complexity, stock_code, company_name,
    question, current_date, current_time_info, session_id,
    topic_name: str = "",
    representative_stocks: list = None,
) -> EvidencePackage:
    """证据装配 + 降级保护（全部复杂度统一使用两阶段快路径）"""
    try:
        logger.info(f"{WAIT_ICON} QA Engine: 使用快路径并行拉取数据 (复杂度={complexity.level})...")
        evidence = await assemble_evidence_fast(
            stock_code, company_name,
            task_plan.tools, question, current_date,
            session_id=session_id,
            topic_name=topic_name,
            representative_stocks=representative_stocks,
            complexity_level=complexity.level,
        )
    except Exception as e:
        # ... 保持现有异常处理
```

- [ ] **Step 3: 验证修改**

```bash
cd Finance/Financial-MCP-Agent && python -c "
from src.qa.qa_engine import _assemble_with_fallback
import inspect
sig = inspect.signature(_assemble_with_fallback)
print('_assemble_with_fallback params:', list(sig.parameters.keys()))
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/qa/qa_engine.py
git commit -m "feat: pass complexity_level to evidence assembler for L1 fast path

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: app.py lifespan — MCP 预热 + 完整集成测试

**Files:**
- Modify: `src/api/app.py:1334-1350` (lifespan function)

**Interfaces:**
- Consumes: nothing (standalone)
- Produces: FastAPI 启动时后台预初始化 MCP 客户端，消除首问 3-10s 延迟

**Goal:** 用户在服务刚启动后第一个问题也能快速响应。

- [ ] **Step 1: 在 lifespan 中添加 MCP 预热**

在 `lifespan()` 函数中，名称缓存加载后添加 MCP 预热（约第 1348 行，`yield` 之前）：

```python
    # 预加载名称缓存
    await asyncio.get_running_loop().run_in_executor(_thread_pool, _ensure_name_cache)

    # MCP 预热：后台预初始化 MCP 客户端，消除首问 3-10s 启动延迟
    try:
        from src.tools.mcp_client import get_mcp_tools
        logger.info(f"{WAIT_ICON} 预初始化 MCP 客户端（后台预热）...")
        warmup_start = time.time()
        warmup_tools = await get_mcp_tools()
        if warmup_tools:
            logger.info(
                f"{SUCCESS_ICON} MCP 预热完成 — "
                f"已加载 {len(warmup_tools)} 个工具 "
                f"(耗时 {time.time() - warmup_start:.1f}s)"
            )
        else:
            logger.warning(f"{ERROR_ICON} MCP 预热返回空工具列表，首问将触发懒加载")
    except Exception as e:
        logger.warning(f"{ERROR_ICON} MCP 预热失败（不影响正常使用，首问将触发懒加载）: {e}")

    yield
```

需要在 lifespan 顶部添加 `import time`：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool_manager, _scoring_engine
    import time
    logger.info(f"{SUCCESS_ICON} 启动 FastAPI 后端...")
```

- [ ] **Step 2: 验证整个流程无 import 错误**

```bash
cd Finance/Financial-MCP-Agent && python -c "
# 验证所有修改的模块可以正常导入
from src.qa.complexity_analyzer import analyze_complexity
from src.qa.task_planner import plan_task, L1_LITE_TOOLS, _get_l1_tools_for_domains
from src.qa.evidence_assembler import assemble_evidence_fast, L1_TOOL_TIMEOUT, L1_MAX_RETRIES
from src.qa.qa_engine import process_question
print('All modules imported successfully!')

# 端到端验证：L1 问题产生的工具数 ≤ 3
result = analyze_complexity('茅台今天多少钱')
print(f'Complexity: {result.level}, Score: {result.score}')
plan = plan_task('茅台今天多少钱', result.level, stock_code='sh.600519', company_name='贵州茅台')
print(f'Domains: {plan.domains}, Tools ({len(plan.tools)}): {plan.tools}')
assert len(plan.tools) <= 3, f'L1 tools should be ≤3, got {len(plan.tools)}'
print('End-to-end check passed!')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/api/app.py
git commit -m "feat: add MCP warmup on FastAPI startup to eliminate first-query delay

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: 5 轮深度代码检查

**本轮为 code review checklist，不产生新代码，逐轮检查并修复。**

#### Round 1: 语法与 Import 检查

- [ ] **每个修改文件的 import 是否完整且无循环依赖**
- [ ] **新增函数名是否与现有函数名冲突**
- [ ] **所有类型注解是否正确**

Checklist:
1. `complexity_analyzer.py`: 只改了数字 24→20，无语法风险
2. `task_planner.py`: 新增 `L1_LITE_TOOLS` dict + `_get_l1_tools_for_domains()` 函数 + `plan_task()` 修改
3. `evidence_assembler.py`: 新增常量 + `_call_tool_safe` 改签名 + `assemble_evidence_fast` 改签名和逻辑
4. `qa_engine.py`: `_assemble_with_fallback` 改签名和调用
5. `app.py`: lifespan 添加 MCP 预热

重点检查：
- `_get_l1_tools_for_domains` 是否在 `plan_task` 之前定义（是——放在 `_get_tools_for_domains` 后面即可）
- `assemble_evidence_fast` 调用处是否都传了 `complexity_level`（仅 qa_engine.py 调用，已改）

#### Round 2: 功能正确性检查

- [ ] **L1 工具数是否确实减少了（对比 L1 vs L2 同样问题）**
- [ ] **L2/L3/L4 行为是否完全不变**
- [ ] **is_l1 判断是否正确传递到所有分支**
- [ ] **托底跳过逻辑是否正确（L1 跳过三处：代表性股票、AkShare托底、ETF直连）**

#### Round 3: 隐藏逻辑问题检查

- [ ] **L1_LITE_TOOLS 是否覆盖了所有 DATA_DOMAINS 中的域**
- [ ] **未在 L1_LITE_TOOLS 中定义的域是否会 fallback 到全量第一个工具**
- [ ] **L1 主题匹配是否还会触发 topic_context 注入（应该保留——LLM 仍需知道上下文）**
- [ ] **缓存 key 是否仍然匹配（是——工具名和参数不变，缓存命中不受影响）**

#### Round 4: 边界条件检查

- [ ] **L1 + 无股票代码: 宏观工具能否正常工作**
- [ ] **L1 + ETF 代码: 是否跳过 ETF 直连（应该跳过）**
- [ ] **L1 + 缓存命中: 是否秒级返回**
- [ ] **L1 + MCP 未预热: 是否有优雅降级**
- [ ] **complexity_level 为空字符串时: is_l1 是否为 False**

#### Round 5: 最终语法和回归检查

- [ ] **运行全部 QA 相关测试**
- [ ] **人工验证 L1 vs L2 工具数量**
- [ ] **检查 git diff 确认无意外修改**
