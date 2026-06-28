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
        "keywords": ["新能源", "风电", "核电", "氢能", "生物质能"],
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
        "keywords": ["人工智能", "AI", "大模型", "ChatGPT", "GPT", "AIGC"],
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
        "keywords": ["汽车", "整车", "智能驾驶", "自动驾驶", "新能源汽车", "电动车"],
    },
    # ── 以下为扩展板块（排序：细分→广义，避免关键词被父级吞掉）──
    "光伏": {
        "etfs": [("sh.515790", "光伏ETF"), ("sz.159857", "光伏ETF")],
        "stocks": [("sh.601012", "隆基绿能"), ("sz.300274", "阳光电源"), ("sh.600438", "通威股份")],
        "keywords": ["光伏", "太阳能", "硅料", "硅片", "逆变器", "HJT", "TOPCon"],
    },
    "锂电池": {
        "etfs": [("sz.159840", "锂电池ETF")],
        "stocks": [("sz.300750", "宁德时代"), ("sz.300014", "亿纬锂能"), ("sz.002460", "赣锋锂业")],
        "keywords": ["锂电池", "锂电", "锂矿", "正极", "负极", "电解液", "隔膜", "碳酸锂"],
    },
    "储能": {
        "etfs": [("sz.159611", "储能ETF")],
        "stocks": [("sz.300274", "阳光电源"), ("sz.300750", "宁德时代"), ("sh.688063", "派能科技")],
        "keywords": ["储能", "储能电池", "户储", "大储", "构网"],
    },
    "新能源车": {
        "etfs": [("sh.515030", "新能源车ETF"), ("sh.516380", "智能电车ETF")],
        "stocks": [("sz.002594", "比亚迪"), ("sz.300750", "宁德时代"), ("sh.601633", "长城汽车")],
        "keywords": ["新能源车", "电动汽车", "电车", "充电桩", "换电"],
    },
    "芯片": {
        "etfs": [("sz.159995", "芯片ETF"), ("sh.512760", "芯片ETF")],
        "stocks": [("sh.688981", "中芯国际"), ("sz.002049", "紫光国微"), ("sh.603986", "兆易创新")],
        "keywords": ["芯片", "集成电路", "晶圆", "光刻", "封装", "存储芯片", "GPU", "CPU", "FPGA"],
    },
    "PCB": {
        "etfs": [("sh.515880", "通信ETF")],
        "stocks": [("sz.002938", "鹏鼎控股"), ("sz.002916", "深南电路"), ("sz.002463", "沪电股份")],
        "keywords": ["PCB", "印制电路板", "电路板", "HDI", "载板", "覆铜板"],
    },
    "CPO": {
        "etfs": [("sh.515880", "通信ETF")],
        "stocks": [("sz.300308", "中际旭创"), ("sz.300502", "新易盛"), ("sz.300394", "天孚通信")],
        "keywords": ["CPO", "光模块", "光通信", "硅光", "共封装", "800G", "1.6T"],
    },
    "机器人": {
        "etfs": [("sh.562500", "机器人ETF")],
        "stocks": [("sh.688017", "绿的谐波"), ("sz.002747", "埃斯顿"), ("sz.300124", "汇川技术")],
        "keywords": ["机器人", "人形机器人", "减速器", "伺服", "具身智能"],
    },
    "算力": {
        "etfs": [("sh.515070", "人工智能ETF")],
        "stocks": [("sz.300308", "中际旭创"), ("sz.002230", "科大讯飞"), ("sh.603019", "中科曙光")],
        "keywords": ["算力", "智算", "数据中心", "服务器", "算力租赁", "云计算"],
    },
    "消费电子": {
        "etfs": [("sh.159732", "消费电子ETF")],
        "stocks": [("sz.002475", "立讯精密"), ("sz.002241", "歌尔股份"), ("sz.300433", "蓝思科技")],
        "keywords": ["消费电子", "手机", "穿戴", "VR", "AR", "MR", "果链"],
    },
    "通信": {
        "etfs": [("sh.515880", "通信ETF")],
        "stocks": [("sz.000063", "中兴通讯"), ("sh.600050", "中国联通"), ("sz.300308", "中际旭创")],
        "keywords": ["通信", "5G", "6G", "基站", "光缆", "卫星通信", "低轨卫星"],
    },
    "计算机": {
        "etfs": [("sh.512720", "计算机ETF")],
        "stocks": [("sz.002415", "海康威视"), ("sz.002230", "科大讯飞"), ("sh.603019", "中科曙光")],
        "keywords": ["计算机", "信创", "数字经济", "数据要素", "信息安全", "操作系统"],
    },
    "软件": {
        "etfs": [("sz.159852", "软件ETF")],
        "stocks": [("sh.688111", "金山办公"), ("sh.600588", "用友网络"), ("sh.600570", "恒生电子")],
        "keywords": ["软件", "SaaS", "ERP", "工业软件", "国产软件", "AI应用"],
    },
    "创新药": {
        "etfs": [("sh.515120", "创新药ETF")],
        "stocks": [("sh.688235", "百济神州"), ("sh.600276", "恒瑞医药"), ("sh.688180", "君实生物")],
        "keywords": ["创新药", "生物药", "ADC", "PD-1", "CAR-T", "双抗", "GLP-1", "临床"],
    },
    "中药": {
        "etfs": [("sz.159647", "中药ETF")],
        "stocks": [("sh.600436", "片仔癀"), ("sz.000538", "云南白药"), ("sh.600085", "同仁堂")],
        "keywords": ["中药", "中成药", "饮片", "配方颗粒"],
    },
    "证券": {
        "etfs": [("sh.512880", "证券ETF")],
        "stocks": [("sh.600030", "中信证券"), ("sz.300059", "东方财富"), ("sh.601688", "华泰证券")],
        "keywords": ["证券", "券商", "投行", "经纪", "牛市旗手"],
    },
    "保险": {
        "etfs": [("sh.512070", "证券保险ETF")],
        "stocks": [("sh.601318", "中国平安"), ("sh.601628", "中国人寿"), ("sh.601601", "中国太保")],
        "keywords": ["保险", "寿险", "财险", "保单", "保费"],
    },
    "家电": {
        "etfs": [("sh.159996", "家电ETF")],
        "stocks": [("sz.000333", "美的集团"), ("sz.000651", "格力电器"), ("sh.600690", "海尔智家")],
        "keywords": ["家电", "白电", "黑电", "小家电", "空调"],
    },
    "有色金属": {
        "etfs": [("sh.512400", "有色金属ETF")],
        "stocks": [("sh.601899", "紫金矿业"), ("sh.603993", "洛阳钼业"), ("sh.600111", "北方稀土")],
        "keywords": ["有色", "有色金属", "铜", "铝", "稀土", "黄金股"],
    },
    "化工": {
        "etfs": [("sh.516020", "化工ETF")],
        "stocks": [("sh.600309", "万华化学"), ("sh.600346", "恒力石化"), ("sh.600989", "宝丰能源")],
        "keywords": ["化工", "石化", "煤化工", "精细化工", "MDI", "涤纶"],
    },
    "钢铁": {
        "etfs": [("sh.515210", "钢铁ETF")],
        "stocks": [("sh.600019", "宝钢股份"), ("sz.000898", "鞍钢股份"), ("sz.000932", "华菱钢铁")],
        "keywords": ["钢铁", "螺纹钢", "热卷", "铁矿石", "粗钢"],
    },
    "建材": {
        "etfs": [("sh.516750", "建材ETF")],
        "stocks": [("sh.600585", "海螺水泥"), ("sz.002271", "东方雨虹"), ("sh.600176", "中国巨石")],
        "keywords": ["建材", "水泥", "玻璃", "防水", "玻纤"],
    },
    "电力": {
        "etfs": [("sh.561560", "电力ETF")],
        "stocks": [("sh.600900", "长江电力"), ("sh.600011", "华能国际"), ("sh.600795", "国电电力")],
        "keywords": ["电力", "火电", "水电", "核电", "绿电", "电力改革", "电价"],
    },
    "传媒": {
        "etfs": [("sh.512980", "传媒ETF")],
        "stocks": [("sz.002027", "分众传媒"), ("sz.300413", "芒果超媒"), ("sh.603000", "人民网")],
        "keywords": ["传媒", "广告", "影视", "出版", "广电", "短视频"],
    },
    "游戏": {
        "etfs": [("sz.159869", "游戏ETF")],
        "stocks": [("sz.002555", "三七互娱"), ("sz.002624", "完美世界"), ("sh.603444", "吉比特")],
        "keywords": ["游戏", "网游", "手游", "电竞", "版号"],
    },
    "农业": {
        "etfs": [("sz.159825", "农业ETF")],
        "stocks": [("sz.002714", "牧原股份"), ("sz.300498", "温氏股份"), ("sz.002311", "海大集团")],
        "keywords": ["农业", "种业", "粮食", "猪肉", "养殖", "饲料", "转基因"],
    },
    "旅游": {
        "etfs": [("sh.562510", "旅游ETF")],
        "stocks": [("sh.601888", "中国中免"), ("sh.600754", "锦江酒店"), ("sh.600258", "首旅酒店")],
        "keywords": ["旅游", "酒店", "免税", "出境游", "景区"],
    },
    "低空经济": {
        "etfs": [("sh.563080", "低空经济ETF"), ("sh.512760", "芯片ETF")],
        "stocks": [("sz.002389", "航天彩虹"), ("sh.688122", "西部超导"), ("sz.300699", "光威复材")],
        "keywords": ["低空经济", "飞行汽车", "eVTOL", "无人机", "通航"],
    },
}


def match_topic(question: str) -> Optional[Tuple[str, dict]]:
    """从问题中识别投资主题，返回 (主题名, 标的映射)。大小写不敏感"""
    q_lower = question.lower()
    for topic, info in TOPIC_STOCK_MAP.items():
        for kw in info["keywords"]:
            if kw.lower() in q_lower:
                return topic, info
    return None


# ── 数据域定义 ────────────────────────────────────

DATA_DOMAINS = {
    "行情": {
        "keywords": ["价格", "涨", "跌", "走势", "行情", "趋势", "K线", "均线",
                     "今天", "最近", "最高", "最低", "收盘", "开盘", "振幅", "换手"],
        "tools": ["tushare_kline", "tushare_daily_basic", "tushare_stock_info",
                  "tushare_adj_factor", "tushare_latest_trading_date"],
        "description": "价格、涨跌幅、K线、换手率等行情数据",
    },
    "估值": {
        "keywords": ["PE", "PB", "PS", "市盈", "市净", "市销", "估值", "贵", "便宜",
                     "分位", "EV/EBITDA", "股息", "分红"],
        "tools": ["tushare_daily_basic", "tushare_pe_percentile", "tushare_ev_ebitda",
                  "tushare_dividend", "tushare_stock_info"],
        "description": "市盈率、市净率、估值分位、股息率等估值数据",
    },
    "财务": {
        "keywords": ["ROE", "ROA", "利润", "收入", "毛利", "净利", "现金", "负债",
                     "资产", "财报", "业绩", "盈利", "成长", "增速", "杜邦"],
        "tools": ["tushare_fina_indicator", "tushare_income", "tushare_balancesheet",
                  "tushare_cashflow", "tushare_stock_info"],
        "description": "利润表、资产负债表、现金流量表、杜邦分析等财务数据（Tushare）",
    },
    "资金": {
        "keywords": ["资金", "主力", "北向", "融资", "融券", "流入", "流出",
                     "成交", "量比", "换手", "净买"],
        "tools": ["tushare_moneyflow", "tushare_hsgt_flow", "tushare_daily_basic"],
        "description": "主力资金流、北向资金、融资融券等资金面数据",
    },
    "行业": {
        "keywords": ["行业", "板块", "赛道", "同行", "竞品", "龙头", "排名",
                     "行业地位", "市场份额", "行业对比"],
        "tools": ["tushare_stock_info"],
        "description": "行业分类、行业对比、板块强弱等行业数据",
    },
    "板块": {
        "keywords": ["板块", "赛道", "行业", "概念", "题材", "热点", "风口",
                     "行业分析", "板块轮动", "赛道股", "概念股", "主题投资"],
        "tools": ["tushare_concept_list", "tushare_ths_index", "tushare_dc_index"],
        "description": "板块/概念/行业分类搜索（成分股明细由ReAct路径二次调用）",
    },
    "新闻": {
        "keywords": ["新闻", "公告", "消息", "事件", "发布", "披露", "分红方案",
                     "回购", "减持", "增持", "业绩预告", "ST", "风险警示"],
        "tools": ["tushare_news", "tushare_st_status", "tushare_major_news", "web_search"],
        "description": "新闻公告、ST风险、重大事件等舆情数据（Tushare+Web Search）",
    },
    "宏观": {
        "keywords": ["利率", "汇率", "CPI", "PPI", "GDP", "PMI", "通胀", "通货膨胀",
                     "货币", "美联储", "央行", "加息", "降息", "经济形势", "宏观",
                     "人民币", "美元", "外汇", "准备金", "LPR", "社融",
                     "经济", "复苏", "繁荣", "衰退", "周期", "景气",
                     "价格指数", "采购经理", "M0", "M1", "M2", "SHIBOR",
                     "就业", "失业", "非农", "零售", "贸易", "关税"],
        "tools": ["tushare_cn_cpi", "tushare_cn_gdp", "tushare_cn_pmi",
                  "tushare_cn_ppi", "tushare_cn_m", "tushare_shibor",
                  "tushare_fx_daily", "tushare_eco_cal", "tushare_stock_info",
                  "get_us_cpi", "get_us_pmi", "get_us_non_farm",
                  "get_us_unemployment", "get_us_gdp", "get_us_retail_sales",
                  "get_comex_inventory", "get_spot_gold_sge",
                  "web_search"],
        "description": "CPI、GDP、PMI、PPI、M2、SHIBOR、汇率等宏观经济指标"
                       "（Tushare中国宏观 + AKShare美国宏观 + Web Search国际事件）",
    },
    "国际": {
        "keywords": ["国际", "全球", "美国", "美元", "美联储", "美债", "COMEX",
                     "伦敦金", "纽约", "欧央行", "日央行", "OPEC",
                     "地缘", "制裁", "贸易战", "关税", "冲突"],
        "tools": ["web_search",
                  # AKShare 国际数据（无速率限制，主力数据源）
                  "get_us_cpi", "get_us_pmi", "get_us_non_farm",
                  "get_us_unemployment", "get_us_gdp", "get_us_retail_sales",
                  "get_comex_inventory", "get_spot_gold_sge",
                  "get_global_futures_spot", "get_sge_spot_prices", "get_fx_rates",
                  # Yahoo Finance（有限速，补充数据源）
                  "get_commodity_price", "get_us_treasury_yield", "get_dollar_index"],
        "description": "国际宏观、商品期货、美国国债、美元指数等数据"
                       "（Web Search + AKShare国际 + Yahoo Finance）",
    },
}

# ── 指数代码映射（非个股，tushare_search_stock 无法查到）──
INDEX_CODE_MAP = {
    "上证指数": "sh.000001",
    "上证综指": "sh.000001",
    "上证": "sh.000001",
    "大盘": "sh.000001",
    "深证成指": "sz.399001",
    "深成指": "sz.399001",
    "深证": "sz.399001",
    "创业板指": "sz.399006",
    "创业板": "sz.399006",
    "沪深300": "sh.000300",
    "科创50": "sh.000688",
    "科创板": "sh.000688",
    "中证500": "sh.000905",
    "上证50": "sh.000016",
    "北证50": "bj.899050",
}


def resolve_index_name(name: str) -> Optional[str]:
    """将常见指数名称解析为标准化代码。用于个股名称反查失败后的兜底。"""
    if not name:
        return None
    clean = name.strip()
    # 精确匹配
    if clean in INDEX_CODE_MAP:
        return INDEX_CODE_MAP[clean]
    # 包含匹配（如"上证指数走势" → 匹配"上证指数"）
    for key, code in INDEX_CODE_MAP.items():
        if len(key) >= 3 and key in clean:
            return code
    return None


# L1 精简工具集：每域只保留 1-2 个核心工具，大幅减少数据获取时间
L1_LITE_TOOLS = {
    "行情": ["tushare_kline", "tushare_daily_basic"],
    "估值": ["tushare_daily_basic", "tushare_pe_percentile"],
    "财务": ["tushare_fina_indicator"],
    "资金": ["tushare_moneyflow"],
    "行业": ["tushare_stock_info"],
    "板块": ["tushare_concept_list"],
    "新闻": ["tushare_news", "tushare_major_news"],
    "宏观": ["tushare_cn_cpi", "tushare_cn_pmi"],
    "国际": ["web_search", "get_us_cpi"],
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
              topic_name: str = "", stock_code: str = "", company_name: str = "") -> TaskPlan:
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

    # 兜底：有股票代码/名称但无数据域匹配时，默认查询行情（纯代码查询场景）
    if not domains and (stock_code or company_name):
        domains = ["行情"]
        # L1 不自动追加估值域（L2+才追加，避免不必要的数据拉取）
        if complexity_level not in ("L0", "L1"):
            pure_code_match = re.search(r'(?<!\d)(\d{5,6})(?!\d)', question)
            if pure_code_match and not company_name:
                domains = ["行情", "估值"]

    # 主题/宏观问题自动追加新闻域和板块域（L2+ 才追加，L1 只保留核心域）
    if topic_name and complexity_level not in ("L0", "L1"):
        if "新闻" not in domains:
            domains.append("新闻")
        # 主题/板块问题自动追加板块域（查概念成分股+板块指数）
        if "板块" not in domains:
            domains.append("板块")
        macro_topics = {"黄金", "白银", "原油", "煤炭", "房地产"}
        if topic_name in macro_topics and "宏观" not in domains:
            domains.append("宏观")
        # 宏观主题总是追加国际域，获取国际商品价格、美国宏观和最新事件
        if topic_name in macro_topics and "国际" not in domains:
            domains.append("国际")

    # L1 使用精简工具集，L2+ 使用全量工具
    if complexity_level == "L1":
        tools = _get_l1_tools_for_domains(domains)
    else:
        tools = _get_tools_for_domains(domains)

    # 全部复杂度统一使用两阶段快路径（并行拉取+单次LLM），不走ReAct
    # ReAct 串行 LLM-工具迭代导致频繁超时（18工具串行需270-810s），快路径并行+缓存3-15s完成
    # L4 仍通过 model=mimo-v2.5-pro + thinking=True + deep模板 保持分析深度
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
    # 无匹配时返回空列表，由上游决定如何处理（L0/主题匹配/无数据）
    return matched


MAX_TOOLS_PER_QUERY = 22


def _get_tools_for_domains(domains: List[str]) -> List[str]:
    """获取数据域对应的工具列表（去重）。多域时优先保留核心工具，上限 MAX_TOOLS_PER_QUERY。"""
    tools: Set[str] = set()
    for domain in domains:
        if domain in DATA_DOMAINS:
            tools.update(DATA_DOMAINS[domain]["tools"])

    result = sorted(tools)
    if len(result) <= MAX_TOOLS_PER_QUERY:
        return result

    # 超限时：web_search 绝对最高优先级，其次国际/宏观工具，财务/估值最后
    priority_prefixes = [
        "web_search", "get_us_", "get_commodity", "get_dollar", "get_gold",
        "get_spot", "get_comex", "get_global", "get_sge", "get_fx",
        "tushare_cn_", "tushare_fx", "tushare_shibor",
        "tushare_eco_cal", "tushare_latest_trading_date",
    ]
    prioritized = [t for t in result if any(t.startswith(p) for p in priority_prefixes)]
    rest = [t for t in result if t not in prioritized]
    # web_search 必须排在最前面（绝对最高优先级）
    if "web_search" in prioritized:
        prioritized.remove("web_search")
        prioritized.insert(0, "web_search")
    return (prioritized + rest)[:MAX_TOOLS_PER_QUERY]


def _get_l1_tools_for_domains(domains: List[str]) -> List[str]:
    """L1 精简版：每域只取1-2个核心工具。去重后返回。
    未在 L1_LITE_TOOLS 中定义的域，回退到全量工具的第一个。"""
    tools: Set[str] = set()
    for domain in domains:
        if domain in L1_LITE_TOOLS:
            tools.update(L1_LITE_TOOLS[domain])
        elif domain in DATA_DOMAINS:
            full_tools = DATA_DOMAINS[domain]["tools"]
            if full_tools:
                tools.add(full_tools[0])
    return sorted(tools)


def extract_stock_from_question(question: str, session_stock_code: str = "",
                                 session_company_name: str = "") -> tuple:
    """
    从问题中提取股票代码和公司名称（仅处理确定性结构化输入）。
    纯数字代码、括号格式、会话引用。自然语言提取由 LLM 完成。
    """
    # 1. 括号格式：公司名(代码)（如 "嘉友国际(603871)"）
    paren_match = re.search(r'([^（(]+?)\s*[（(](\d{5,6})[)）]', question)
    if paren_match:
        name = paren_match.group(1).strip()
        code = normalize_stock_code(paren_match.group(2))
        return code, name

    # 2. 纯数字股票代码（如 "603871"）——括号内的代码已在上面处理
    code_match = re.search(r'(?<!\d)(\d{5,6})(?!\d)', question)
    if code_match:
        code = normalize_stock_code(code_match.group(1))
        return code, None

    # 3. 会话上下文引用（如 "它"、"这个股票"）
    ref_pattern = r'(?:它|这个|那个|这只|那家|刚才|上次)(?:股票|公司|股|标的)?'
    if re.search(ref_pattern, question) and session_stock_code:
        return session_stock_code, session_company_name

    return None, None


def normalize_stock_code(code: str) -> str:
    """标准化A股代码格式：为纯数字代码添加交易所前缀。
    6/688/5开头→sh., 0/3/1/4开头→sz., 430/431/8/920开头→bj.。
    已有前缀的代码原样返回。非标准格式原样返回。"""
    if not code or not isinstance(code, str):
        return code
    code = code.strip()
    if not code:
        return code
    if code.startswith(("sh.", "sz.", "bj.")):
        return code
    _is_bse = (code.startswith(("430", "431", "920")) or
               (len(code) >= 3 and code[:3] in
                ("830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
                 "870", "871", "872", "873")))
    if _is_bse:
        return f"bj.{code}"
    elif code.startswith(("6", "688", "5")):
        return f"sh.{code}"
    elif code.startswith(("0", "3", "1", "4")):
        return f"sz.{code}"
    return code  # 非标准前缀原样返回
