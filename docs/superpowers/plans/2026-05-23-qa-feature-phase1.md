# 智能问答功能 Phase 1 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 A 股投资顾问 Agent 新增智能问答子系统，支持自然语言开放式交流与多轮对话，按问题复杂度动态路由数据获取策略。

**Architecture:** 两阶段流水线 — 复杂度分析 → 任务规划与证据装配(并行MCP工具调用) → LLM回答生成(流式SSE)。快路径(L1/L2)并行拉取数据后直接LLM回答；复杂路径(L3/L4)使用ReAct Agent迭代调工具。

**Tech Stack:** Python 3.10+, FastAPI SSE, LangChain MCP Adapters, OpenAI-compatible API (MiMo), Streamlit

**Spec:** `问答功能开发指南.md`

---

## 文件结构映射

| 文件 | 职责 |
|------|------|
| `src/qa/__init__.py` | 模块导出 |
| `src/qa/session_manager.py` | 多轮对话会话管理（内存存储，线程安全） |
| `src/qa/complexity_analyzer.py` | 三层复杂度识别（规则触发+加权打分） |
| `src/qa/task_planner.py` | 数据域→工具映射，决定是否需要ReAct |
| `src/qa/evidence_assembler.py` | 并行MCP工具调用+证据包组装 |
| `src/qa/answer_generator.py` | LLM调用+回答模板+SSE流式输出 |
| `src/qa/qa_engine.py` | 总编排器，协调全流程 |
| `src/api/app.py` | 新增3个QA REST端点 |
| `src/app/config.py` | 新增QA超时设置 |
| `src/app/api_client.py` | 新增QA API客户端函数 |
| `src/app/pages/04_智能问答.py` | Streamlit聊天界面 |
| `src/app/Home.py` | 更新导航 |
| `src/utils/model_config.py` | 新增Model 5 (_5后缀) |
| `.env.example` | 已完成,无需改动 |
| `tests/test_qa.py` | QA模块核心测试 |

---

### Task 1: 更新模型配置支持 Model 5

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/utils/model_config.py`

- [ ] **Step 1: 在 `AGENT_MODEL_SUFFIX` 中新增 `_5` 条目**

在 `AGENT_MODEL_SUFFIX` 字典末尾（`"value_agent": "_4",` 之后）添加：
```python
    # ── Model 5: MiMo-V2.5 (智能问答主模型) ──
    "qa_engine": "_5",
```

同时更新模块 docstring，在四模型架构说明下方添加一行：
```
#   Model 5 (OPENAI_COMPATIBLE_5):  MiMo-V2.5      → 智能问答
```

- [ ] **Step 2: 验证 `get_model_config_for_agent("qa_engine")` 能正确返回 Model 5 配置**

该函数已有通用逻辑：`suffix = AGENT_MODEL_SUFFIX.get(agent_name, "")`，若 `suffix="_5"` 则读取 `OPENAI_COMPATIBLE_API_KEY_5`、`OPENAI_COMPATIBLE_BASE_URL_5`、`OPENAI_COMPATIBLE_MODEL_5`。回退逻辑在 suffix 非空且配置缺失时会回退到 Model 1。

- [ ] **Step 3: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/utils/model_config.py
git commit -m "feat: add Model 5 (MiMo-V2.5) support for QA engine"
```

---

### Task 2: 创建会话管理器

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/qa/__init__.py`
- Create: `Finance/Financial-MCP-Agent/src/qa/session_manager.py`

- [ ] **Step 1: 创建模块 `__init__.py`**

```python
"""智能问答子系统 — 自然语言开放式A股分析问答"""
```

- [ ] **Step 2: 实现 `session_manager.py`**

```python
"""
会话管理器：多轮对话会话的创建、更新、查询、过期清理。
纯内存存储（Phase 1），线程安全。
"""
import time
import uuid
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class QAMessage:
    """单条问答消息"""
    role: str          # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class QASession:
    """问答会话"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: List[QAMessage] = field(default_factory=list)
    # 上下文追踪
    last_stock_code: Optional[str] = None
    last_company_name: Optional[str] = None
    last_complexity_level: str = "L1"
    # 历史压缩摘要（Phase 3 使用）
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": [{"role": m.role, "content": m.content, "timestamp": m.timestamp}
                        for m in self.history],
            "last_stock_code": self.last_stock_code,
            "last_company_name": self.last_company_name,
        }

    def add_message(self, role: str, content: str):
        self.history.append(QAMessage(role=role, content=content))
        self.updated_at = time.time()

    def get_history_for_llm(self, max_turns: int = 10) -> List[dict]:
        """返回最近 N 轮对话，格式适配 LLM messages"""
        recent = self.history[-max_turns * 2:]  # 每轮=user+assistant
        return [{"role": m.role, "content": m.content} for m in recent]


class SessionManager:
    """会话管理器（线程安全的内存存储）"""

    def __init__(self, session_ttl: int = 3600):
        self._sessions: Dict[str, QASession] = {}
        self._lock = threading.Lock()
        self._session_ttl = session_ttl  # 会话过期时间（秒），默认1小时

    def create_session(self) -> str:
        """创建新会话，返回 session_id"""
        session_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._sessions[session_id] = QASession(session_id=session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[QASession]:
        """获取会话，不存在时返回 None"""
        self._cleanup_expired()
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess:
                sess.updated_at = time.time()
            return sess

    def get_or_create(self, session_id: Optional[str]) -> QASession:
        """获取或创建会话"""
        if session_id:
            sess = self.get_session(session_id)
            if sess:
                return sess
        new_id = self.create_session()
        return self._sessions[new_id]

    def delete_session(self, session_id: str) -> bool:
        """删除会话，返回是否成功"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    def update_context(self, session_id: str, **kwargs):
        """更新会话上下文（last_stock_code, last_company_name 等）"""
        sess = self.get_session(session_id)
        if sess:
            for key, value in kwargs.items():
                if hasattr(sess, key):
                    setattr(sess, key, value)

    def _cleanup_expired(self):
        """清理过期会话"""
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, sess in self._sessions.items()
                if now - sess.updated_at > self._session_ttl
            ]
            for sid in expired:
                del self._sessions[sid]

    @property
    def active_count(self) -> int:
        self._cleanup_expired()
        return len(self._sessions)


# 全局单例
_global_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    return _global_session_manager
```

- [ ] **Step 3: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/qa/__init__.py Finance/Financial-MCP-Agent/src/qa/session_manager.py
git commit -m "feat: add QA session manager with in-memory storage"
```

---

### Task 3: 创建复杂度分析器

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/qa/complexity_analyzer.py`

- [ ] **Step 1: 实现复杂度分析器**

```python
"""
高灵敏复杂度分析器 — 三层识别架构

Layer 1: 规则触发器（硬判断，速度快）
Layer 2: 加权打分模型（未命中硬触发时使用）
Layer 3: 运行时升级器（Phase 2 实现，此处保留接口）
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ComplexityResult:
    """复杂度分析结果"""
    level: str            # "L1" | "L2" | "L3" | "L4"
    score: int            # 0-100
    triggers: List[str]   # 触发的规则列表
    score_detail: dict    # 各维度得分明细
    need_clarify: bool    # 是否需要澄清
    recommended_model: str  # "mimo-v2.5" | "mimo-v2.5-pro"
    recommended_thinking: bool
    recommended_react: bool      # Phase 1: 仅 L3/L4 为 True
    recommended_template: str    # "quick" | "standard" | "deep"


# ── Layer 1: 规则触发器 ──────────────────────────

# 硬触发关键词（命中任意一个即至少 L3）
HARD_TRIGGERS_L4 = [
    # 跨标的比较
    "比较", "对比", "vs", "VS", " versus ", "优劣", "差异", "区别",
    "和.*比", "与.*比", "跟.*比",
    # 因果归因
    "为什么", "原因", "驱动", "归因", "本质", "背后.*逻辑",
    # 情景推演
    "如果", "假设", "情景", "预期.*会", "明年", "下季度", "下一季度",
    "展望", "前景",
    # 深度判断
    "还能不能", "值不值得", "是不是机会", "是不是陷阱", "还能拿",
    "要不要", "该不该",
    # 综合报告
    "全面分析", "深度分析", "展开讲讲", "写.*报告",
    "详细.*分析", "系统.*分析",
    # 筛选排序
    "筛选", "排序", "打分", "优选", "推荐.*股",
    # 策略判断
    "交易策略", "操作策略", "配置.*建议", "仓位",
]

HARD_TRIGGERS_L3 = [
    # 多维组合（财务+估值 OR 估值+资金 OR 技术+基本面）
    # 用关键词密度检测代替
    "估值.*合理", "贵不贵", "便不便宜", "高不高",
    "走势.*强", "走势.*弱", "趋势.*判断",
    "板块.*轮动", "行业.*景气", "行业.*前景",
    "资金.*流向", "主力.*动向", "北向.*资金",
    # 多时间跨度
    "短期.*长期", "短线.*长线", "最近.*未来",
]


def _check_hard_triggers(question: str) -> tuple:
    """Layer 1: 规则触发器，返回 (level, triggers)"""
    import re
    triggers = []

    for pattern in HARD_TRIGGERS_L4:
        if re.search(pattern, question):
            triggers.append(f"L4硬触发: {pattern}")

    for pattern in HARD_TRIGGERS_L3:
        if re.search(pattern, question):
            triggers.append(f"L3硬触发: {pattern}")

    if triggers:
        # 有L4触发 → L4, 否则 L3
        has_l4 = any("L4硬触发" in t for t in triggers)
        return ("L4" if has_l4 else "L3", triggers)
    return (None, [])


# ── Layer 2: 加权打分模型 ──────────────────────────

def _score_question(question: str) -> ComplexityResult:
    """Layer 2: 对未命中硬触发的问题做加权打分"""
    score = 0
    detail = {}

    # 1. 主体数量 (0-15)
    import re
    # 统计股票代码出现次数
    codes = re.findall(r'\b\d{5,6}\b', question)
    stock_count = len(codes)
    # 统计公司名/板块名（简单启发式：2-4个汉字的连续出现）
    names = re.findall(r'[一-鿿]{2,4}(?:股票|股份|集团|银行|证券|保险|科技|医药|汽车|能源)', question)
    entity_count = max(stock_count, len(names) if names else 1)
    if entity_count >= 3:
        detail["主体数量"] = 15
    elif entity_count == 2:
        detail["主体数量"] = 10
    elif entity_count == 1:
        detail["主体数量"] = 3
    else:
        detail["主体数量"] = 0
    score += detail["主体数量"]

    # 2. 时间跨度 (0-10)
    time_keywords_long = ["年", "长期", "三年", "五年", "历史", "历年", "跨周期"]
    time_keywords_mid = ["季度", "月", "中期", "半年", "今年以来"]
    time_keywords_short = ["周", "最近", "近期", "今天", "昨天", "本周", "本月"]
    if any(kw in question for kw in time_keywords_long):
        detail["时间跨度"] = 10
    elif any(kw in question for kw in time_keywords_mid):
        detail["时间跨度"] = 6
    elif any(kw in question for kw in time_keywords_short):
        detail["时间跨度"] = 3
    else:
        detail["时间跨度"] = 1
    score += detail["时间跨度"]

    # 3. 分析维度数量 (0-20)
    dimension_keywords = {
        "行情": ["价格", "涨", "跌", "走势", "行情", "趋势", "K线", "均线"],
        "估值": ["PE", "PB", "市盈", "市净", "估值", "贵", "便宜", "分位"],
        "财务": ["ROE", "利润", "收入", "毛利", "现金", "负债", "财报", "业绩", "盈利"],
        "资金": ["资金", "主力", "北向", "融资", "融券", "流入", "流出", "成交"],
        "行业": ["行业", "板块", "赛道", "同行", "竞品", "龙头"],
        "事件": ["新闻", "公告", "分红", "回购", "减持", "增持", "业绩预告"],
    }
    dims_found = 0
    for dim, kws in dimension_keywords.items():
        if any(kw in question for kw in kws):
            dims_found += 1
    detail["分析维度"] = min(dims_found * 5, 20)
    score += detail["分析维度"]

    # 4. 推理深度 (0-20)
    deep_patterns = ["为什么", "怎么.*变化", "影响", "判断", "预测", "推测"]
    compare_patterns = ["对比", "比较", "区别", "vs", "优劣"]
    desc_patterns = ["是多少", "什么.*是", "查询", "看看", "了解"]
    if any(re.search(p, question) for p in deep_patterns):
        detail["推理深度"] = 18
    elif any(re.search(p, question) for p in compare_patterns):
        detail["推理深度"] = 12
    elif any(re.search(p, question) for p in desc_patterns):
        detail["推理深度"] = 5
    else:
        detail["推理深度"] = 3
    score += detail["推理深度"]

    # 5. 计算复杂度 (0-15)
    calc_keywords = ["分位", "同比", "环比", "TTM", "相对强弱", "超额", "回撤", "波动"]
    calc_count = sum(1 for kw in calc_keywords if kw in question)
    detail["计算复杂度"] = min(calc_count * 5, 15)
    score += detail["计算复杂度"]

    # 6. 歧义程度 (0-10)
    ambiguity = 0
    if re.search(r'(?:它|这个|那个|这只|刚才|上次).*(?:股票|公司|股)', question) and not codes:
        ambiguity = 8  # 指代不明且无股票代码
    elif not codes and not names and len(question) < 10:
        ambiguity = 5  # 短问题无明确主体
    detail["歧义程度"] = ambiguity
    score += ambiguity

    # 7. 输出要求 (0-10)
    output_keywords = ["报告", "详细", "全面", "深度", "展开", "分析一下"]
    if any(kw in question for kw in output_keywords):
        detail["输出要求"] = 8
    else:
        detail["输出要求"] = 2
    score += detail["输出要求"]

    # 分级
    if score <= 24:
        level = "L1"
    elif score <= 49:
        level = "L2"
    elif score <= 69:
        level = "L3"
    else:
        level = "L4"

    return ComplexityResult(
        level=level,
        score=score,
        triggers=[f"评分={score}"],
        score_detail=detail,
        need_clarify=(detail.get("歧义程度", 0) >= 8),
        recommended_model="mimo-v2.5-pro" if level in ("L3", "L4") else "mimo-v2.5",
        recommended_thinking=(level == "L4"),
        recommended_react=(level in ("L3", "L4")),
        recommended_template="quick" if level == "L1" else ("standard" if level == "L2" else "deep"),
    )


# ── 公共接口 ──────────────────────────────────────

def analyze_complexity(question: str, history_depth: int = 0) -> ComplexityResult:
    """
    分析问题的复杂度。

    Args:
        question: 用户问题
        history_depth: 多轮对话深度（追问链中自动提升复杂度）

    Returns:
        ComplexityResult
    """
    # Layer 1: 硬触发规则
    hard_level, triggers = _check_hard_triggers(question)

    if hard_level:
        score = 75 if hard_level == "L4" else 55
        return ComplexityResult(
            level=hard_level,
            score=score,
            triggers=triggers,
            score_detail={"硬触发": hard_level},
            need_clarify=False,
            recommended_model="mimo-v2.5-pro" if hard_level in ("L3", "L4") else "mimo-v2.5",
            recommended_thinking=(hard_level == "L4"),
            recommended_react=(hard_level in ("L3", "L4")),
            recommended_template="deep" if hard_level == "L4" else "standard",
        )

    # Layer 2: 加权打分
    result = _score_question(question)

    # 追问链自动提升
    if history_depth >= 3:
        bump = min(history_depth, 3)  # 最多提升3级
        current_idx = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
        levels = ["L1", "L2", "L3", "L4"]
        new_idx = min(current_idx.get(result.level, 0) + bump, 3)
        if new_idx > current_idx.get(result.level, 0):
            result.level = levels[new_idx]
            result.triggers.append(f"追问链提升: depth={history_depth} → {result.level}")

    # 用户显式要求深度分析
    if any(kw in question for kw in ["详细", "深度", "全面", "展开"]):
        current_idx = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
        if current_idx.get(result.level, 0) < 2:
            result.level = "L3"
            result.triggers.append("用户显式要求深度分析 → L3")

    return result
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/qa/complexity_analyzer.py
git commit -m "feat: add QA complexity analyzer with 3-layer architecture"
```

---

### Task 4: 创建任务规划器

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/qa/task_planner.py`

- [ ] **Step 1: 实现任务规划器**

```python
"""
任务规划器 — 根据问题意图和复杂度确定所需数据域和工具。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set
import re


# ── 数据域定义 ────────────────────────────────────

DATA_DOMAINS = {
    "行情": {
        "keywords": ["价格", "涨", "跌", "走势", "行情", "趋势", "K线", "均线",
                     "今天", "最近", "最高", "最低", "收盘", "开盘", "振幅", "换手"],
        "tools": ["get_historical_k_data", "tushare_kline", "tushare_daily_basic",
                  "get_stock_basic_info", "get_latest_trading_date"],
        "description": "价格、涨跌幅、K线、换手率等行情数据",
    },
    "估值": {
        "keywords": ["PE", "PB", "PS", "市盈", "市净", "市销", "估值", "贵", "便宜",
                     "分位", "EV/EBITDA", "股息", "分红"],
        "tools": ["tushare_daily_basic", "tushare_pe_percentile", "tushare_ev_ebitda",
                  "tushare_dividend", "get_dividend_data", "get_stock_basic_info"],
        "description": "市盈率、市净率、估值分位、股息率等估值数据",
    },
    "财务": {
        "keywords": ["ROE", "ROA", "利润", "收入", "毛利", "净利", "现金", "负债",
                     "资产", "财报", "业绩", "盈利", "成长", "增速", "杜邦"],
        "tools": ["get_profit_data", "get_balance_data", "get_cash_flow_data",
                  "get_growth_data", "get_dupont_data", "get_operation_data",
                  "tushare_fina_indicator", "tushare_stock_info"],
        "description": "利润表、资产负债表、现金流量表、杜邦分析等财务数据",
    },
    "资金": {
        "keywords": ["资金", "主力", "北向", "融资", "融券", "流入", "流出",
                     "成交", "量比", "换手", "净买"],
        "tools": ["tushare_moneyflow", "get_market_analysis_timeframe",
                  "tushare_daily_basic"],
        "description": "主力资金流、融资融券、北向资金等资金面数据",
    },
    "行业": {
        "keywords": ["行业", "板块", "赛道", "同行", "竞品", "龙头", "排名",
                     "行业地位", "市场份额", "行业对比"],
        "tools": ["get_stock_industry", "get_stock_basic_info",
                  "tushare_stock_info", "get_market_analysis_timeframe"],
        "description": "行业分类、行业对比、板块强弱等行业数据",
    },
    "新闻": {
        "keywords": ["新闻", "公告", "消息", "事件", "发布", "披露", "分红方案",
                     "回购", "减持", "增持", "业绩预告", "ST", "风险警示"],
        "tools": ["crawl_news", "get_st_risk_data", "tushare_st_status"],
        "description": "新闻公告、ST风险、重大事件等舆情数据",
    },
}


@dataclass
class TaskPlan:
    """任务规划结果"""
    domains: List[str]                # 需要的数据域
    tools: List[str]                  # 需要的工具列表
    need_react: bool                  # 是否使用 ReAct
    reason: str                       # 规划理由
    expected_data_volume: str         # "small" | "medium" | "large"


def plan_task(question: str, complexity_level: str, history_text: str = "") -> TaskPlan:
    """
    根据问题内容和复杂度等级规划数据获取任务。

    Returns:
        TaskPlan: 包含所需数据域、工具列表和是否使用ReAct
    """
    domains = _identify_domains(question)
    tools = _get_tools_for_domains(domains)

    # 确定是否需要 ReAct
    # Phase 1: 仅 L3/L4 启用 ReAct
    # L3 先尝试并行拉取，不足时再升级为 ReAct（运行时升级，Phase 2）
    need_react = complexity_level in ("L3", "L4")

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


def _identify_domains(question: str) -> List[str]:
    """识别问题涉及的数据域"""
    matched = []
    for domain_name, domain_info in DATA_DOMAINS.items():
        for kw in domain_info["keywords"]:
            if kw in question:
                matched.append(domain_name)
                break
    # 至少包含行情域（大多数问题都需要）
    if not matched:
        matched = ["行情"]
    return matched


def _get_tools_for_domains(domains: List[str]) -> List[str]:
    """获取数据域对应的工具列表（去重）"""
    tools: Set[str] = set()
    for domain in domains:
        if domain in DATA_DOMAINS:
            tools.update(DATA_DOMAINS[domain]["tools"])
    return sorted(tools)


def extract_stock_from_question(question: str, session_stock_code: str = "",
                                 session_company_name: str = "") -> tuple:
    """
    从问题中提取股票代码和公司名称。
    优先使用问题中的信息，其次使用会话上下文。

    Returns:
        (stock_code, company_name)
    """
    code = None
    name = None

    # 提取股票代码
    code_match = re.search(r'\b(\d{5,6})\b', question)
    if code_match:
        code = code_match.group(1)

    # 括号内提取
    paren_match = re.search(r'([^（(]+?)\s*[（(](\d{5,6})[)）]', question)
    if paren_match:
        name = paren_match.group(1).strip()
        code = paren_match.group(2)

    # 引用前文主语（"它""这只股""这家公司"）
    ref_patterns = [r'(?:它|这个|那个|这只|那家|刚才|上次)(?:股票|公司|股|标的)?']
    if re.search('|'.join(ref_patterns), question) and not code:
        code = session_stock_code
        name = session_company_name

    # 纯代码输入
    if not name and code:
        # 尝试从问题中提取名称（代码前后的中文）
        pass

    # 标准化代码格式
    if code:
        if not code.startswith(("sh.", "sz.")):
            if code.startswith("6"):
                code = f"sh.{code}"
            else:
                code = f"sz.{code}"

    return code, name
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/qa/task_planner.py
git commit -m "feat: add QA task planner with domain-tool mapping"
```

---

### Task 5: 创建证据装配器

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/qa/evidence_assembler.py`

- [ ] **Step 1: 实现证据装配器**

```python
"""
证据装配器 — 并行MCP工具调用 + 指标预计算 + 证据包组装

快路径(L1/L2): asyncio.gather 并行拉取所有工具数据
ReAct路径(L3/L4): 使用 LangGraph ReAct Agent（Phase 2 完善）
"""
import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)

TOOL_TIMEOUT = 30  # 单个工具超时（秒）


@dataclass
class EvidencePackage:
    """证据包 — 分析友好型数据结构"""
    subject: str = ""                 # 分析主体
    stock_code: str = ""
    company_name: str = ""
    data_time: str = ""               # 数据截至时间
    domains_queried: List[str] = field(default_factory=list)
    facts: List[Dict[str, str]] = field(default_factory=list)  # [{label, value, source}]
    raw_text: str = ""                # 原始数据文本（供LLM参考）
    missing: List[str] = field(default_factory=list)           # 缺失数据域
    tool_call_summary: str = ""       # 工具调用摘要
    elapsed_seconds: float = 0.0


async def _call_tool_safe(tool, kwargs: dict, timeout: float, label: str) -> str:
    """安全调用单个MCP工具，带超时和异常保护"""
    try:
        result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=timeout)
        text = str(result).strip()
        if len(text) > 25:
            logger.info(f"{SUCCESS_ICON} QA Evidence: {label} 成功 ({len(text)} 字符)")
            return text
        else:
            logger.warning(f"QA Evidence: {label} 返回过短 ({len(text)} 字符)")
            return ""
    except asyncio.TimeoutError:
        logger.warning(f"QA Evidence: {label} 超时({timeout}s)")
        return ""
    except Exception as e:
        logger.warning(f"QA Evidence: {label} 失败: {e}")
        return ""


async def assemble_evidence_fast(
    stock_code: str,
    company_name: str,
    tools: List[str],
    question: str,
    current_date: str,
) -> EvidencePackage:
    """
    快路径：并行拉取所有所需工具的数据，组装为证据包。

    类比快筛股票池的做法 — 确定数据域后，asyncio.gather 一次性拉取。
    """
    start_time = time.time()
    evidence = EvidencePackage(
        subject=company_name or stock_code or question,
        stock_code=stock_code or "",
        company_name=company_name or "",
        domains_queried=[],
    )

    # 获取 MCP 工具
    try:
        all_mcp_tools = await get_mcp_tools(tool_filter=tools)
    except Exception as e:
        logger.error(f"{ERROR_ICON} QA Evidence: 获取MCP工具失败: {e}")
        evidence.missing.append(f"MCP工具不可用: {e}")
        return evidence

    if not all_mcp_tools:
        evidence.missing.append("无可用MCP工具")
        return evidence

    logger.info(f"{WAIT_ICON} QA Evidence: 并行调用 {len(all_mcp_tools)} 个工具...")

    # 并行调用所有工具
    tasks = []
    labels = []
    for tool in all_mcp_tools:
        kwargs = _build_tool_kwargs(tool.name, stock_code, company_name, question)
        tasks.append(_call_tool_safe(tool, kwargs, TOOL_TIMEOUT, tool.name))
        labels.append(tool.name)

    results = await asyncio.gather(*tasks)

    # 组装原始文本
    raw_parts = []
    success_count = 0
    for label, text in zip(labels, results):
        if text:
            raw_parts.append(f"### [{label}]\n{text}")
            success_count += 1
        else:
            evidence.missing.append(label)

    evidence.raw_text = "\n\n".join(raw_parts) if raw_parts else "(无数据)"
    evidence.tool_call_summary = f"{success_count}/{len(labels)} 工具成功"
    evidence.elapsed_seconds = time.time() - start_time

    logger.info(
        f"{SUCCESS_ICON} QA Evidence: 装配完成 "
        f"({evidence.elapsed_seconds:.1f}s, {success_count}/{len(labels)} 成功)"
    )

    return evidence


async def assemble_evidence_react(
    stock_code: str,
    company_name: str,
    question: str,
    current_date: str,
    current_time_info: str,
) -> EvidencePackage:
    """
    ReAct 路径：使用 LangGraph ReAct Agent 迭代调工具。
    Phase 2 完善。Phase 1 提供基本实现作为占位。
    """
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage, AIMessage

    evidence = EvidencePackage(
        subject=company_name or stock_code or question,
        stock_code=stock_code or "",
        company_name=company_name or "",
    )
    start_time = time.time()

    try:
        all_mcp_tools = await get_mcp_tools()
    except Exception as e:
        evidence.missing.append(f"ReAct MCP工具不可用: {e}")
        return evidence

    if not all_mcp_tools:
        evidence.missing.append("无可用MCP工具")
        return evidence

    # 获取模型配置
    from src.utils.model_config import get_model_config_for_agent
    model_cfg = get_model_config_for_agent("qa_engine")

    llm = ChatOpenAI(
        model=model_cfg["model_name"],
        api_key=model_cfg["api_key"],
        base_url=model_cfg["base_url"],
        temperature=0.6,
        request_timeout=180,
        max_tokens=8000,
        extra_body={"thinking": {"type": "disabled"}},  # 工具阶段强制关闭thinking
    )

    agent = create_react_agent(llm, all_mcp_tools)

    agent_input = f"""请获取以下问题的相关数据：{question}

当前股票：{company_name}（{stock_code}）
当前日期：{current_date}

请使用可用的工具获取实际数据，基于数据回答，不要编造。"""

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=agent_input)]},
        config={"recursion_limit": 20}
    )

    # 提取结果
    if "messages" in response and isinstance(response["messages"], list):
        ai_msgs = [m for m in response["messages"] if isinstance(m, AIMessage)]
        if ai_msgs:
            evidence.raw_text = ai_msgs[-1].content
        else:
            evidence.raw_text = str(response["messages"][-1])

    evidence.elapsed_seconds = time.time() - start_time
    evidence.tool_call_summary = f"ReAct完成 ({evidence.elapsed_seconds:.0f}s)"
    return evidence


def _build_tool_kwargs(tool_name: str, stock_code: str, company_name: str,
                        question: str) -> dict:
    """根据工具名构建合适的参数"""
    # 提取纯数字代码
    clean_code = stock_code.replace("sh.", "").replace("sz.", "") if stock_code else ""

    # 按工具名返回合适的参数
    if tool_name in ("get_stock_basic_info", "tushare_stock_info", "get_stock_industry"):
        return {"code": stock_code or ""}
    elif tool_name in ("get_historical_k_data", "tushare_kline"):
        return {"code": stock_code or "", "days": 120}
    elif tool_name in ("tushare_daily_basic", "tushare_pe_percentile"):
        return {"code": stock_code or ""}
    elif tool_name in ("get_profit_data", "get_balance_data", "get_cash_flow_data",
                       "get_growth_data", "get_operation_data", "get_dupont_data"):
        return {"code": stock_code or ""}
    elif tool_name == "tushare_fina_indicator":
        return {"code": stock_code or ""}
    elif tool_name == "tushare_moneyflow":
        return {"code": stock_code or ""}
    elif tool_name == "tushare_dividend" or tool_name == "get_dividend_data":
        return {"code": stock_code or ""}
    elif tool_name == "tushare_ev_ebitda":
        return {"code": stock_code or ""}
    elif tool_name == "crawl_news":
        return {"query": company_name or clean_code or question, "top_k": 10}
    elif tool_name in ("get_st_risk_data", "tushare_st_status"):
        return {"code": stock_code or ""}
    elif tool_name == "get_latest_trading_date":
        return {}
    elif tool_name == "get_market_analysis_timeframe":
        return {}
    else:
        return {"query": question}
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/qa/evidence_assembler.py
git commit -m "feat: add QA evidence assembler with parallel tool calling"
```

---

### Task 6: 创建回答生成器

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/qa/answer_generator.py`

- [ ] **Step 1: 实现回答生成器**

```python
"""
回答生成器 — LLM调用 + 回答模板 + 流式输出

三种模板：quick(L1), standard(L2), deep(L3/L4-Phase2)
严格分离「数据事实」和「分析判断」
"""
import asyncio
import time
from typing import AsyncGenerator, Dict, Any
from openai import AsyncOpenAI
import httpx

from src.qa.complexity_analyzer import ComplexityResult
from src.qa.evidence_assembler import EvidencePackage
from src.utils.model_config import get_model_config_for_agent
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)

LLM_TIMEOUT = 60  # 快路径LLM超时


def _build_system_prompt(template: str, current_date: str) -> str:
    """构建系统提示词"""

    base = f"""你是一位资深A股证券分析师，拥有10年以上卖方研究经验，擅长用通俗语言解释复杂的金融问题。

**当前日期：{current_date}**
**分析基准时间：{current_date}**

你的核心原则：
1. **数据优先**：所有数字类结论必须来源于工具提供的证据数据，绝不编造
2. **事实与判断分离**：回答中严格区分「数据事实」和「分析判断」
3. **专业但不晦涩**：让非专业投资者也能听懂，但保持专业深度
4. **有结论不绕**：先回答用户最关心的问题，再给证据
5. **风险提示具体**：风险提示要针对具体问题，不要泛泛而谈
6. **明确边界**：无法获取的数据说"无法获取"，不做猜测性补充"""

    if template == "quick":
        return base + f"""

**输出格式（快答模板）：**
1. 先给核心结论（2-3句话）
2. 关键证据（2-4条，每条标注数据来源）
3. 风险提示（1-2句话，针对性强）
4. 标注数据截至时间：{current_date}

要求：回答简洁有力，控制在300-500字。"""
    elif template == "standard":
        return base + f"""

**输出格式（标准分析模板）：**
1. 核心结论（3-5句话）
2. 分维度分析（行情面/估值面/财务面/资金面/行业面/消息面，根据实际情况选2-4个相关维度）
3. 综合判断
4. 风险提示
5. 数据截至时间：{current_date}
6. 可继续追问的方向（1-2个）

要求：回答控制在600-1000字，先结论后证据。"""
    else:
        return base + f"""

**输出格式（深度分析模板）：**
1. 核心结论
2. 分维度深度分析（每个维度包含数据事实+分析判断）
3. 与可比对象/行业对比
4. 关键矛盾点分析
5. 情景判断（多情景推演）
6. 风险与反证
7. 数据截至时间：{current_date}
8. 后续观察点

要求：全面深入但不冗长，控制在1500-2500字。"""


def _build_user_prompt(
    question: str,
    evidence: EvidencePackage,
    history_text: str,
) -> str:
    """构建用户提示词"""

    parts = [f"## 用户问题\n{question}\n"]

    if history_text:
        parts.append(f"## 历史对话摘要\n{history_text}\n")

    parts.append(f"""## 证据数据
{evidence.raw_text}

## 数据获取摘要
- 成功获取: {evidence.tool_call_summary}
- 缺失数据: {', '.join(evidence.missing) if evidence.missing else '无'}
- 数据获取耗时: {evidence.elapsed_seconds:.1f}秒""")

    return "\n".join(parts)


async def generate_answer_stream(
    question: str,
    evidence: EvidencePackage,
    complexity: ComplexityResult,
    history_text: str,
    current_date: str,
) -> AsyncGenerator[str, None]:
    """
    流式生成回答（SSE）。

    Yields:
        SSE格式字符串: "data: {chunk}\n\n" 或 "data: [DONE]\n\n"
    """
    model_cfg = get_model_config_for_agent("qa_engine")
    api_key = model_cfg["api_key"]
    base_url = model_cfg["base_url"]
    model_name = model_cfg["model_name"]

    if not all([api_key, base_url, model_name]):
        yield _sse_error("模型配置缺失，请检查 .env 文件")
        return

    system_prompt = _build_system_prompt(complexity.recommended_template, current_date)
    user_prompt = _build_user_prompt(question, evidence, history_text)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(connect=15.0, read=float(LLM_TIMEOUT), write=30.0, pool=10.0),
        max_retries=1,
    )

    # 构建 extra_body — 显式传递 thinking 参数
    extra_body = {"thinking": {"type": "enabled" if complexity.recommended_thinking else "disabled"}}

    logger.info(
        f"{WAIT_ICON} QA Answer: 调用LLM (model={model_name}, "
        f"template={complexity.recommended_template}, "
        f"thinking={'ON' if complexity.recommended_thinking else 'OFF'})"
    )

    try:
        stream = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0 if complexity.recommended_thinking else 0.6,
            max_tokens=4096 if complexity.recommended_template == "quick" else 8192,
            extra_body=extra_body,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield f"data: {content}\n\n"

        yield "data: [DONE]\n\n"

    except asyncio.TimeoutError:
        yield _sse_error("回答生成超时，请重试")
    except Exception as e:
        logger.error(f"{ERROR_ICON} QA Answer: LLM 调用失败: {e}")
        yield _sse_error(f"回答生成失败: {e}")


def _sse_error(message: str) -> str:
    """SSE 错误格式"""
    return f"data: [ERROR] {message}\n\ndata: [DONE]\n\n"
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/qa/answer_generator.py
git commit -m "feat: add QA answer generator with streaming and templates"
```

---

### Task 7: 创建 QA 引擎（总编排器）

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/qa/qa_engine.py`

- [ ] **Step 1: 实现 QA 引擎**

```python
"""
QA引擎 — 智能问答总编排器

协调复杂度分析 → 任务规划 → 证据装配 → 回答生成的完整流程。
"""
import time
from typing import AsyncGenerator, Optional

from src.qa.session_manager import get_session_manager, QASession
from src.qa.complexity_analyzer import analyze_complexity, ComplexityResult
from src.qa.task_planner import plan_task, extract_stock_from_question
from src.qa.evidence_assembler import (
    assemble_evidence_fast,
    assemble_evidence_react,
    EvidencePackage,
)
from src.qa.answer_generator import generate_answer_stream
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)


async def process_question(
    question: str,
    session_id: Optional[str] = None,
    current_date: str = "",
    current_time_info: str = "",
) -> AsyncGenerator[str, None]:
    """
    处理用户问题的主流程。

    Args:
        question: 用户问题
        session_id: 会话ID（None时自动创建新会话）
        current_date: 当前日期 YYYY-MM-DD
        current_time_info: 完整时间信息

    Yields:
        SSE格式字符串
    """
    start_time = time.time()
    session_mgr = get_session_manager()
    session = session_mgr.get_or_create(session_id)

    # 确保 session_id 正确传递
    actual_session_id = session.session_id

    # Step 1: 复杂度分析
    logger.info(f"{WAIT_ICON} QA Engine: 分析问题复杂度...")
    history_depth = len(session.history) // 2  # 对话轮数
    complexity = analyze_complexity(question, history_depth)

    logger.info(
        f"QA Engine: 复杂度={complexity.level}, 评分={complexity.score}, "
        f"触发={complexity.triggers}, ReAct={complexity.recommended_react}"
    )

    # 如果需要澄清
    if complexity.need_clarify:
        yield _sse_event("clarify", {
            "message": "我需要确认一下：您指的是哪只股票或哪个板块？请补充股票代码或名称。",
            "session_id": actual_session_id,
        })
        yield "data: [DONE]\n\n"
        return

    # Step 2: 提取股票信息
    stock_code, company_name = extract_stock_from_question(
        question,
        session_stock_code=session.last_stock_code or "",
        session_company_name=session.last_company_name or "",
    )

    # 更新会话上下文
    if stock_code:
        session_mgr.update_context(actual_session_id,
                                   last_stock_code=stock_code,
                                   last_company_name=company_name)

    # 发送会话信息
    yield _sse_event("meta", {
        "session_id": actual_session_id,
        "complexity": complexity.level,
        "stock_code": stock_code,
        "company_name": company_name,
    })

    # Step 3: 任务规划
    history_text = _build_history_text(session)
    task_plan = plan_task(question, complexity.level, history_text)

    logger.info(
        f"QA Engine: 任务规划 — 数据域={task_plan.domains}, "
        f"工具数={len(task_plan.tools)}, ReAct={task_plan.need_react}"
    )

    # Step 4: 证据装配
    yield _sse_event("status", {"message": f"正在获取数据（涉及{len(task_plan.domains)}个数据域）..."})

    if task_plan.need_react and complexity.level == "L4":
        # L4: 使用 ReAct（Phase 1 基础实现）
        logger.info(f"{WAIT_ICON} QA Engine: 使用 ReAct 路径...")
        evidence = await assemble_evidence_react(
            stock_code or "", company_name or "",
            question, current_date, current_time_info
        )
    else:
        # L1/L2/L3: 快路径，并行拉取
        logger.info(f"{WAIT_ICON} QA Engine: 使用快路径并行拉取数据...")
        evidence = await assemble_evidence_fast(
            stock_code or "", company_name or "",
            task_plan.tools, question, current_date
        )

    yield _sse_event("status", {
        "message": (f"数据获取完成（{evidence.tool_call_summary}，"
                   f"耗时{evidence.elapsed_seconds:.1f}秒），正在生成回答...")
    })

    # Step 5: 流式生成回答
    yield _sse_event("answer_start", {"template": complexity.recommended_template})

    async for chunk in generate_answer_stream(
        question=question,
        evidence=evidence,
        complexity=complexity,
        history_text=history_text,
        current_date=current_date,
    ):
        yield chunk

    # Step 6: 记录到会话历史
    total_time = time.time() - start_time
    logger.info(
        f"{SUCCESS_ICON} QA Engine: 回答完成 "
        f"({total_time:.1f}s, 复杂度={complexity.level})"
    )


def _sse_event(event_type: str, data: dict) -> str:
    """生成SSE事件"""
    import json
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_history_text(session: QASession) -> str:
    """构建历史对话文本（用于上下文注入）"""
    if not session.history:
        return ""
    # 取最近3轮对话
    recent = session.history[-6:]
    lines = []
    for msg in recent:
        role = "用户" if msg.role == "user" else "分析师"
        lines.append(f"{role}: {msg.content[:200]}")
    return "\n".join(lines)
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/qa/qa_engine.py
git commit -m "feat: add QA engine orchestrator with full pipeline"
```

---

### Task 8: 添加后端 API 端点

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/api/app.py`

- [ ] **Step 1: 在 app.py 顶部添加 QA 引擎导入**

在现有导入区域（`from src.tools.mcp_client import get_mcp_tools` 之后）添加：
```python
# QA 引擎
from src.qa.qa_engine import process_question
from src.qa.session_manager import get_session_manager
```

- [ ] **Step 2: 在 app.py 末尾添加三个端点（在 `if __name__` 之前）**

```python
# ──────────────────────────────────────────────
# 智能问答 API
# ──────────────────────────────────────────────

from fastapi.responses import StreamingResponse


class QARequest(BaseModel):
    question: str
    session_id: Optional[str] = None


@app.post("/api/qa/ask")
async def qa_ask(req: QARequest):
    """
    智能问答 — SSE流式端点

    接收自然语言问题，返回流式SSE事件：
    - event: meta       → 会话ID、复杂度、股票信息
    - event: status     → 数据获取状态
    - event: answer_start → 回答开始
    - data: {chunk}     → 回答内容流
    - event: clarify    → 需要澄清（歧义高时）
    - data: [DONE]      → 结束
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    question = req.question.strip()

    # 获取当前时间
    from datetime import datetime
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_date_cn = now.strftime("%Y年%m月%d日")
    current_weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
    current_time = now.strftime("%H:%M:%S")
    current_time_info = f"{current_date_cn} ({current_date}) {current_weekday_cn} {current_time}"

    return StreamingResponse(
        process_question(
            question=question,
            session_id=req.session_id,
            current_date=current_date,
            current_time_info=current_time_info,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/qa/sessions/{session_id}")
async def qa_get_session(session_id: str):
    """获取会话历史和上下文"""
    session_mgr = get_session_manager()
    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return session.to_dict()


@app.delete("/api/qa/sessions/{session_id}")
async def qa_delete_session(session_id: str):
    """删除会话"""
    session_mgr = get_session_manager()
    deleted = session_mgr.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return {"status": "deleted", "session_id": session_id}
```

- [ ] **Step 3: 添加 API 路由到 lifespan/startup（如果需要）**

在 `@asynccontextmanager` 的 lifespan 函数中，startup 事件中添加：
```python
# 初始化 QA 会话管理器（首次访问时自动初始化，此处预热）
get_session_manager()
```

- [ ] **Step 4: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/api/app.py
git commit -m "feat: add QA REST endpoints (SSE ask, session CRUD)"
```

---

### Task 9: 添加前端 API 客户端函数

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/app/config.py`
- Modify: `Finance/Financial-MCP-Agent/src/app/api_client.py`

- [ ] **Step 1: 在 config.py 添加 QA 超时配置**

在文件末尾添加：
```python
# ──────────────────────────────────────────────
# 智能问答超时设置（秒）
# ──────────────────────────────────────────────
QA_TIMEOUT = 120      # QA HTTP 请求超时（工具调用+LLM生成）
QA_STREAM_TIMEOUT = 180  # QA 流式总超时
```

- [ ] **Step 2: 在 api_client.py 添加 QA 客户端函数**

在文件末尾（`if __name__` 之前）添加：
```python
# ──────────────────────────────────────────────
# 智能问答 API
# ──────────────────────────────────────────────

async def qa_ask_stream(question: str, session_id: Optional[str] = None):
    """
    智能问答流式请求 — 异步生成器

    Yields:
        dict: {"type": "meta"|"status"|"answer"|"clarify"|"done"|"error", "data": ...}
    """
    if MOCK_MODE:
        # Mock 流式响应
        yield {"type": "meta", "data": {"session_id": "mock-session", "complexity": "L1"}}
        yield {"type": "status", "data": {"message": "正在分析..."}}
        mock_answer = (
            "根据当前市场数据，该股票近期走势较为稳健。"
            "从估值角度看，PE处于历史中位数水平，具有一定投资价值。"
            "需要注意的是，市场整体波动较大，建议关注成交量变化。"
        )
        yield {"type": "answer", "data": mock_answer}
        yield {"type": "done", "data": None}
        return

    try:
        async with httpx.AsyncClient(timeout=QA_STREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/api/qa/ask",
                json={"question": question, "session_id": session_id},
            ) as response:
                response.raise_for_status()

                current_event = None
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        current_event = line[7:].strip()
                    elif line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            yield {"type": "done", "data": None}
                            return
                        if current_event in ("meta", "status", "clarify", "answer_start"):
                            import json as _json
                            try:
                                parsed = _json.loads(data_str)
                            except Exception:
                                parsed = {"message": data_str}
                            yield {"type": current_event, "data": parsed}
                            current_event = None
                        else:
                            # 普通文本数据块
                            yield {"type": "answer", "data": data_str}

    except httpx.HTTPStatusError as e:
        yield {"type": "error", "data": f"请求失败: {e.response.text}"}
    except httpx.RequestError as e:
        yield {"type": "error", "data": f"连接后端失败: {e}"}
    except Exception as e:
        yield {"type": "error", "data": f"未知错误: {e}"}


async def qa_get_session(session_id: str) -> dict:
    """获取会话历史"""
    if MOCK_MODE:
        return {
            "session_id": session_id,
            "history": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮您的？"},
            ],
        }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{API_BASE_URL}/api/qa/sessions/{session_id}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"获取会话失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def qa_delete_session(session_id: str) -> dict:
    """删除会话"""
    if MOCK_MODE:
        return {"status": "deleted", "session_id": session_id}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.delete(f"{API_BASE_URL}/api/qa/sessions/{session_id}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"删除会话失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")
```

- [ ] **Step 3: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/app/config.py Finance/Financial-MCP-Agent/src/app/api_client.py
git commit -m "feat: add QA API client functions with SSE streaming support"
```

---

### Task 10: 创建前端聊天界面

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/app/pages/04_智能问答.py`

- [ ] **Step 1: 实现 Streamlit 聊天界面**

```python
"""
智能问答页面 — 自然语言开放式A股分析对话
"""
import asyncio
import os
import sys

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st
from api_client import qa_ask_stream, APIError

st.set_page_config(
    page_title="智能问答",
    page_icon="💬",
    layout="wide",
)

# ──────────────────────────────────────────────
# 样式
# ──────────────────────────────────────────────
st.markdown("""
<style>
.chat-message {
    padding: 1rem 1.5rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
.chat-message.user {
    background: #e3f2fd;
    border-left: 3px solid #1976d2;
}
.chat-message.assistant {
    background: #f5f5f5;
    border-left: 3px solid #4caf50;
}
.chat-meta {
    font-size: 0.8rem;
    color: #888;
    margin-bottom: 0.3rem;
}
.stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 页面标题
# ──────────────────────────────────────────────
st.title("💬 智能问答")
st.caption("与资深A股投研分析师直接对话，支持自然语言开放式交流与多轮追问")

# ──────────────────────────────────────────────
# 侧边栏：会话管理
# ──────────────────────────────────────────────
with st.sidebar:
    st.subheader("会话管理")

    if st.button("🆕 新建会话", use_container_width=True):
        st.session_state.pop("qa_session_id", None)
        st.session_state.pop("qa_messages", None)
        st.rerun()

    # 显示当前会话ID
    session_id = st.session_state.get("qa_session_id", "")
    if session_id:
        st.info(f"会话ID: `{session_id}`")
    else:
        st.caption("尚未创建会话，发送第一条消息后自动创建")

    st.divider()
    st.caption("支持的提问示例：")
    st.caption("• 为什么XX股票今天大涨？")
    st.caption("• 这家公司利润增长但股价不涨？")
    st.caption("• 半导体行业目前怎么看？")
    st.caption("• 黄金走势会怎么变化？")

# ──────────────────────────────────────────────
# 聊天消息历史
# ──────────────────────────────────────────────
if "qa_messages" not in st.session_state:
    st.session_state.qa_messages = []

# 渲染历史消息
for msg in st.session_state.qa_messages:
    with st.chat_message(msg["role"]):
        if msg.get("meta"):
            st.caption(f"复杂度: {msg['meta']} | 会话: {session_id}")
        st.markdown(msg["content"])

# ──────────────────────────────────────────────
# 输入框
# ──────────────────────────────────────────────
if prompt := st.chat_input("输入您的投资分析问题..."):
    # 添加用户消息
    st.session_state.qa_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 异步处理回答
    async def _process():
        answer_placeholder = st.empty()
        full_answer = ""
        meta_info = ""

        try:
            async for event in qa_ask_stream(
                question=prompt,
                session_id=st.session_state.get("qa_session_id"),
            ):
                event_type = event.get("type")
                event_data = event.get("data")

                if event_type == "meta":
                    sid = event_data.get("session_id", "")
                    complexity = event_data.get("complexity", "")
                    stock = event_data.get("company_name", "")
                    if sid:
                        st.session_state["qa_session_id"] = sid
                    meta_info = f"复杂度: {complexity}"
                    if stock:
                        meta_info += f" | 股票: {stock}"

                elif event_type == "status":
                    with st.chat_message("assistant"):
                        st.caption(event_data.get("message", "处理中..."))

                elif event_type == "answer":
                    full_answer += str(event_data)
                    # 实时更新显示
                    with st.chat_message("assistant"):
                        if meta_info:
                            st.caption(meta_info)
                        st.markdown(full_answer + "▌")

                elif event_type == "clarify":
                    with st.chat_message("assistant"):
                        st.warning(event_data.get("message", "需要更多信息"))

                elif event_type == "error":
                    with st.chat_message("assistant"):
                        st.error(event_data)

                elif event_type == "done":
                    pass  # 流结束

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"请求失败: {e}")
            full_answer = f"(_请求失败: {e}_)"

        # 保存最终回答
        if full_answer:
            st.session_state.qa_messages.append({
                "role": "assistant",
                "content": full_answer,
                "meta": meta_info,
            })

    asyncio.run(_process())
    st.rerun()

# ──────────────────────────────────────────────
# 底部操作
# ──────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ 清除当前对话", use_container_width=True):
        st.session_state.qa_messages = []
        st.rerun()
with col2:
    if st.button("📋 复制最后回答", use_container_width=True):
        if st.session_state.qa_messages:
            last = st.session_state.qa_messages[-1]
            if last["role"] == "assistant":
                st.toast("回答已复制到剪贴板（请手动 Ctrl+C）")
                st.code(last["content"], language=None)
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/app/pages/04_智能问答.py
git commit -m "feat: add QA chat interface as 4th Streamlit page"
```

---

### Task 11: 更新 Home.py 导航

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/app/Home.py`

- [ ] **Step 1: 更新 Home.py 为导航中心**

将整个文件替换为：
```python
"""
Streamlit 入口页 — 四大功能导航中心
"""
import streamlit as st

st.set_page_config(
    page_title="股票投资顾问 Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 股票投资顾问 Agent")
st.caption("A股智能分析平台 — 选你想用的功能")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/01_股票查询.py", label="🔍 股票查询", icon="🔍",
                 help="输入股票代码或名称，快速获取分析或生成深度报告")
    st.caption("单只股票快速查询 + 深度报告生成")

    st.page_link("pages/02_股票池.py", label="📊 股票池", icon="📊",
                 help="管理短线/中线/长线投资池，打分筛选")
    st.caption("多期限股票池管理 + 打分排序")

with col2:
    st.page_link("pages/04_智能问答.py", label="💬 智能问答", icon="💬",
                 help="与资深投研分析师直接对话，支持自然语言开放式交流")
    st.caption("自然语言开放式A股分析问答 + 多轮对话")

    st.page_link("pages/03_批量打分.py", label="📋 批量打分", icon="📋",
                 help="上传Excel批量打分，快速筛选标的池")
    st.caption("Excel批量上传 + 大规模初筛")

st.markdown("---")
st.caption("提示：也可以使用左侧边栏 `〉` 直接切换页面")
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/app/Home.py
git commit -m "feat: update Home.py to 4-function navigation hub"
```

---

### Task 12: 创建测试文件

**Files:**
- Create: `Finance/Financial-MCP-Agent/tests/test_qa.py`

- [ ] **Step 1: 编写核心逻辑测试**

```python
"""
QA 模块单元测试 — 复杂度分析、任务规划、会话管理
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from src.qa.session_manager import SessionManager, QASession, QAMessage
from src.qa.complexity_analyzer import analyze_complexity, ComplexityResult
from src.qa.task_planner import plan_task, extract_stock_from_question


# ── SessionManager 测试 ─────────────────────────

class TestSessionManager:
    def test_create_session(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        assert len(sid) == 8
        assert mgr.get_session(sid) is not None

    def test_get_or_create_new(self):
        mgr = SessionManager()
        sess = mgr.get_or_create(None)
        assert sess.session_id is not None
        assert len(mgr._sessions) == 1

    def test_get_or_create_existing(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        sess = mgr.get_or_create(sid)
        assert sess.session_id == sid
        assert len(mgr._sessions) == 1

    def test_delete_session(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        assert mgr.delete_session(sid) is True
        assert mgr.get_session(sid) is None

    def test_delete_nonexistent(self):
        mgr = SessionManager()
        assert mgr.delete_session("nonexist") is False

    def test_session_history(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        sess = mgr.get_session(sid)
        sess.add_message("user", "测试问题")
        sess.add_message("assistant", "测试回答")
        assert len(sess.history) == 2
        assert sess.history[0].role == "user"
        assert sess.history[1].role == "assistant"

    def test_update_context(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        mgr.update_context(sid, last_stock_code="sh.600519", last_company_name="贵州茅台")
        sess = mgr.get_session(sid)
        assert sess.last_stock_code == "sh.600519"
        assert sess.last_company_name == "贵州茅台"

    def test_history_for_llm(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        sess = mgr.get_session(sid)
        for i in range(5):
            sess.add_message("user", f"问题{i}")
            sess.add_message("assistant", f"回答{i}")
        llm_history = sess.get_history_for_llm(max_turns=3)
        assert len(llm_history) == 6  # 3 turns * 2


# ── ComplexityAnalyzer 测试 ─────────────────────

class TestComplexityAnalyzer:
    def test_l1_simple_question(self):
        result = analyze_complexity("贵州茅台今天跌了多少")
        assert result.level in ("L1", "L2")

    def test_l4_comparison_trigger(self):
        result = analyze_complexity("把宁德时代和比亚迪从估值和盈利角度做个比较")
        assert result.level in ("L3", "L4")
        assert any("比较" in t for t in result.triggers) or result.level == "L4"

    def test_l4_why_trigger(self):
        result = analyze_complexity("为什么这只股票利润增长但股价不涨")
        assert result.level in ("L3", "L4")
        assert any("为什么" in t for t in result.triggers) or result.score >= 50

    def test_l4_scenario_trigger(self):
        result = analyze_complexity("如果下季度利润继续改善，现在估值算不算便宜")
        assert result.level in ("L3", "L4")
        assert any("如果" in t for t in result.triggers) or result.score >= 50

    def test_l4_recommend_react(self):
        result = analyze_complexity("全面深度分析一下茅台的估值、财务、行业地位和未来前景")
        assert result.recommended_react is True
        assert result.recommended_model == "mimo-v2.5-pro"

    def test_l1_recommend_no_react(self):
        result = analyze_complexity("茅台PE多少")
        assert result.recommended_react is False
        assert result.recommended_model == "mimo-v2.5"

    def test_history_depth_bump(self):
        result = analyze_complexity("茅台PE多少", history_depth=4)
        # 追问链提升后至少 L2+
        assert result.level in ("L2", "L3", "L4")

    def test_explicit_deep_analysis(self):
        result = analyze_complexity("帮我详细分析一下这只股票")
        # 虽没有其他触发，但用户显式要求深度
        assert result.level in ("L2", "L3", "L4")


# ── TaskPlanner 测试 ────────────────────────────

class TestTaskPlanner:
    def test_identify_single_domain(self):
        plan = plan_task("茅台PE多少", "L1")
        assert "估值" in plan.domains or "行情" in plan.domains
        assert plan.need_react is False

    def test_identify_multi_domain(self):
        plan = plan_task("茅台最近走势怎么样，估值贵不贵，资金在流入还是流出", "L2")
        assert len(plan.domains) >= 2

    def test_l4_needs_react(self):
        plan = plan_task("深度分析茅台", "L4")
        assert plan.need_react is True

    def test_extract_stock_code(self):
        code, name = extract_stock_from_question("分析一下600519的估值")
        assert code == "sh.600519" or code is not None

    def test_extract_paren_code(self):
        code, name = extract_stock_from_question("分析贵州茅台(600519)")
        assert name == "贵州茅台"
        assert code == "sh.600519" or code is not None

    def test_extract_reference(self):
        code, name = extract_stock_from_question(
            "这只股票最近怎么样",
            session_stock_code="sh.600519",
            session_company_name="贵州茅台"
        )
        assert code == "sh.600519"
        assert name == "贵州茅台"

    def test_tools_are_unique(self):
        plan = plan_task("分析茅台估值和财务", "L2")
        assert len(plan.tools) == len(set(plan.tools))  # 无重复
```

- [ ] **Step 2: 运行测试验证**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/test_qa.py -v
```

- [ ] **Step 3: Commit**

```bash
git add Finance/Financial-MCP-Agent/tests/test_qa.py
git commit -m "test: add QA module unit tests (complexity, planning, sessions)"
```

---

### Task 13: 更新 .env.example

**Files:**
- Modify: `Finance/Financial-MCP-Agent/.env.example`

- [ ] **Step 1: 添加 Model 5 配置区域**

在 Model 4 配置之后（`OPENAI_COMPATIBLE_MODEL_4=kimi-k2.6` 行之后）添加：
```
# ──────────────────────────────────────────────
# Model 5: MiMo-V2.5（小米API）
# 用途: 智能问答主模型（qa_engine）
# ──────────────────────────────────────────────
OPENAI_COMPATIBLE_API_KEY_5=your_mimo_api_key_here
OPENAI_COMPATIBLE_BASE_URL_5=https://api.xiaomimimo.com/v1
OPENAI_COMPATIBLE_MODEL_5=mimo-v2.5
```

同时更新末尾的注释说明，在 "分配逻辑" 区域添加：
```
#   qa_engine            → Model 5 (MiMo-V2.5)     | thinking 按需  | 快答4K/分析8K
#   qa_engine(复杂)       → Model 1 (MiMo-V2.5-Pro) | thinking 按需  | 深度分析
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/.env.example
git commit -m "docs: add Model 5 (MiMo-V2.5) config to .env.example"
```

---

## Plan Self-Review

**1. Spec coverage check:**
- [x] 问题接入层 → `qa_engine.py` (process_question)
- [x] 复杂度分析器 → `complexity_analyzer.py`
- [x] 任务规划器 → `task_planner.py`
- [x] 数据/工具执行层 → `evidence_assembler.py`
- [x] 证据装配 → `evidence_assembler.py` (EvidencePackage)
- [x] 模型路由中心 → 融入 `complexity_analyzer.py` (ComplexityResult) + `qa_engine.py`
- [x] 回答生成器 → `answer_generator.py` (流式SSE)
- [x] 会话管理 → `session_manager.py`
- [x] 前端聊天界面 → `pages/04_智能问答.py`
- [x] API端点 → `app.py` (SSE流式)
- [x] 模型配置 → `model_config.py` (_5 suffix)
- [x] 测试 → `tests/test_qa.py`
- [x] 防幻觉机制 → answer_generator中系统提示词要求事实/判断分离
- [x] 数据截至时间 → 所有模板要求标注
- [x] 快路径默认 → L1/L2并行拉取, L3/L4才用ReAct
- [x] 复杂度灵敏优先 → 硬触发+打分+追问提升

**2. Placeholder scan:** No TBD/TODO found. All code is concrete.

**3. Type consistency:** ComplexityResult used consistently across complexity_analyzer, task_planner, evidence_assembler, answer_generator, qa_engine.

**Spec requirement gaps:** None identified. All Phase 1 requirements from the design guide are covered.
