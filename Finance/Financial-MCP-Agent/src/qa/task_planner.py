"""
任务规划器 — 根据问题意图和复杂度确定所需数据域和工具。

快路径(L1/L2): 确定数据域 → 并行拉取全部数据 → LLM直接回答
慢路径(L3/L4): 确定数据域 → 先尝试并行拉取 → 不足则启用ReAct
"""
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional
import re


# ── 主题→标的映射（无股票代码时自动匹配代表性ETF/个股）──

TOPIC_STOCK_MAP = {
    "黄金": {
        "etfs": [("sh.159934", "黄金ETF"), ("sh.518880", "黄金ETF"), ("sh.518800", "黄金基金")],
        "stocks": [("sh.600489", "中金黄金"), ("sh.601899", "紫金矿业"), ("sh.600547", "山东黄金")],
        "keywords": ["黄金", "金价", "金价走势", "黄金价格", "贵金属", "现货黄金"],
    },
    "白银": {
        "etfs": [("sh.518880", "黄金ETF")],
        "stocks": [("sz.000603", "盛达资源"), ("sh.601899", "紫金矿业")],
        "keywords": ["白银", "银价", "现货白银"],
    },
    "原油": {
        "etfs": [("sh.510410", "资源ETF"), ("sz.159945", "能源ETF")],
        "stocks": [("sh.601857", "中国石油"), ("sh.600028", "中国石化"), ("sh.600938", "中国海油")],
        "keywords": ["原油", "石油", "油价", "国际油价", "布伦特", "WTI"],
    },
    "新能源": {
        "etfs": [("sh.516160", "新能源ETF"), ("sz.159875", "新能源ETF")],
        "stocks": [("sz.300750", "宁德时代"), ("sz.002594", "比亚迪"), ("sh.601012", "隆基绿能")],
        "keywords": ["新能源", "光伏", "锂电", "储能", "风电", "电动车"],
    },
    "半导体": {
        "etfs": [("sh.512480", "半导体ETF"), ("sz.159995", "芯片ETF")],
        "stocks": [("sh.688981", "中芯国际"), ("sz.002049", "紫光国微"), ("sh.603986", "兆易创新")],
        "keywords": ["半导体", "芯片", "集成电路", "光刻", "晶圆"],
    },
    "白酒": {
        "etfs": [("sh.512690", "酒ETF")],
        "stocks": [("sh.600519", "贵州茅台"), ("sz.000858", "五粮液"), ("sz.000568", "泸州老窖")],
        "keywords": ["白酒", "茅台", "五粮液", "高端酒", "次高端"],
    },
    "银行": {
        "etfs": [("sh.512800", "银行ETF")],
        "stocks": [("sh.600036", "招商银行"), ("sh.601398", "工商银行"), ("sh.000001", "平安银行")],
        "keywords": ["银行", "银行股", "银行业", "商业银行", "大金融"],
    },
    "人工智能": {
        "etfs": [("sh.515070", "人工智能ETF"), ("sz.159819", "AI智能ETF")],
        "stocks": [("sz.002230", "科大讯飞"), ("sh.688256", "寒武纪"), ("sz.300033", "同花顺")],
        "keywords": ["人工智能", "AI", "大模型", "ChatGPT", "智能", "算力"],
    },
    "医药": {
        "etfs": [("sh.512010", "医药ETF"), ("sh.512170", "医疗ETF")],
        "stocks": [("sh.600276", "恒瑞医药"), ("sz.300760", "迈瑞医疗"), ("sz.300015", "爱尔眼科")],
        "keywords": ["医药", "医疗", "创新药", "CXO", "生物医药", "中药"],
    },
    "房地产": {
        "etfs": [("sh.512200", "房地产ETF")],
        "stocks": [("sh.600048", "保利发展"), ("sz.000002", "万科A"), ("sh.001979", "招商蛇口")],
        "keywords": ["房地产", "楼市", "房价", "地产", "开发商"],
    },
    "消费": {
        "etfs": [("sh.510150", "消费ETF"), ("sz.159928", "消费ETF")],
        "stocks": [("sh.600519", "贵州茅台"), ("sz.000858", "五粮液"), ("sz.002714", "牧原股份")],
        "keywords": ["消费", "大消费", "内需", "消费升级", "消费降级"],
    },
    "军工": {
        "etfs": [("sh.512660", "军工ETF"), ("sh.512710", "军工龙头ETF")],
        "stocks": [("sh.600760", "中航沈飞"), ("sz.002025", "航天电器"), ("sh.600893", "航发动力")],
        "keywords": ["军工", "国防", "军事", "航天", "武器装备"],
    },
    "煤炭": {
        "etfs": [("sh.515220", "煤炭ETF")],
        "stocks": [("sh.601088", "中国神华"), ("sh.601225", "陕西煤业"), ("sh.600188", "兖矿能源")],
        "keywords": ["煤炭", "煤价", "动力煤", "焦煤"],
    },
    "汽车": {
        "etfs": [("sh.516110", "汽车ETF")],
        "stocks": [("sz.002594", "比亚迪"), ("sh.600104", "上汽集团"), ("sh.601633", "长城汽车")],
        "keywords": ["汽车", "整车", "新能源汽车", "智能驾驶", "自动驾驶"],
    },
}


def match_topic(question: str) -> Optional[Tuple[str, dict]]:
    """从问题中识别投资主题，返回 (主题名, 标的映射)"""
    for topic, info in TOPIC_STOCK_MAP.items():
        for kw in info["keywords"]:
            if kw in question:
                return topic, info
    return None


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


def plan_task(question: str, complexity_level: str, history_text: str = "",
              topic_matched: bool = False) -> TaskPlan:
    """
    根据问题内容和复杂度等级规划数据获取任务。

    L1/L2: 并行拉取全量数据
    L3: 先并行拉取，不足时运行时升级为ReAct
    L4: 直接使用ReAct
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
        # 提取历史中最近提到的数据域关键词追加到问题中
        recent_kw = _extract_domain_keywords_from_text(history_text)
        if recent_kw:
            augmented_question = question + " " + " ".join(recent_kw)

    domains = _identify_domains(augmented_question)

    # 主题/宏观问题自动追加新闻域
    if topic_matched and "新闻" not in domains:
        domains.append("新闻")

    tools = _get_tools_for_domains(domains)

    # 确定是否需要 ReAct
    # L1/L2: 永不用ReAct
    # L3: 默认不走ReAct，仅运行时升级触发（证据矛盾/数据大量缺失等极端情况）
    # L4: 稳定走ReAct
    need_react = (complexity_level == "L4")

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


def _extract_domain_keywords_from_text(text: str) -> List[str]:
    """从历史对话文本中提取数据域关键词"""
    all_kw = []
    for domain_info in DATA_DOMAINS.values():
        for kw in domain_info["keywords"]:
            if kw in text:
                all_kw.append(kw)
    return all_kw[-10:]  # 最多取10个


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
    """
    code = None
    name = None

    # 提取股票代码
    code_match = re.search(r'\b(\d{5,6})\b', question)
    if code_match:
        code = code_match.group(1)

    # 括号内提取: 嘉友国际(603871)
    paren_match = re.search(r'([^（(]+?)\s*[（(](\d{5,6})[)）]', question)
    if paren_match:
        name = paren_match.group(1).strip()
        code = paren_match.group(2)

    # 引用前文主语
    ref_pattern = r'(?:它|这个|那个|这只|那家|刚才|上次)(?:股票|公司|股|标的)?'
    if re.search(ref_pattern, question) and not code:
        code = session_stock_code
        name = session_company_name

    # 标准化代码格式（仅限合法A股前缀: 6=沪市, 0/3=深市, 688=科创板, 8=北交所）
    if code:
        if not code.startswith(("sh.", "sz.")):
            if code.startswith(("6", "688", "5", "8")):
                code = f"sh.{code}"
            elif code.startswith(("0", "3", "1", "4")):
                code = f"sz.{code}"
            # 其他前缀（如999）不添加交易所前缀，保持原样

    return code, name
