"""
基金类型知识库 — 各类基金的关键指标基准水平, 用于快速查询中的同类对比。
"""

# 基金类型 → 基准指标（2024-2025年A股市场公募基金中位数水平）
FUND_TYPE_BENCHMARKS = {
    "股票型": {
        "annual_return_1y": "8~15%",
        "annual_return_3y": "5~12%",
        "annual_volatility": "18~25%",
        "max_drawdown": "-25~-40%",
        "sharpe_ratio": "0.3~0.8",
        "mgmt_fee": "1.2~1.5%",
        "description": "权益仓位≥80%，高收益高波动，适合风险承受能力较强的投资者",
    },
    "混合型": {
        "annual_return_1y": "5~12%",
        "annual_return_3y": "4~10%",
        "annual_volatility": "12~20%",
        "max_drawdown": "-15~-30%",
        "sharpe_ratio": "0.3~0.7",
        "mgmt_fee": "1.0~1.5%",
        "description": "股债灵活配置，风险和收益介于股票型和债券型之间",
    },
    "债券型": {
        "annual_return_1y": "2.5~5%",
        "annual_return_3y": "2.5~4.5%",
        "annual_volatility": "1~4%",
        "max_drawdown": "-1~-5%",
        "sharpe_ratio": "0.5~1.2",
        "mgmt_fee": "0.3~0.8%",
        "description": "以债券为主要投资标的，风险较低，适合稳健型投资者",
    },
    "货币型": {
        "annual_return_1y": "1.2~2.0%",
        "annual_return_3y": "1.5~2.2%",
        "annual_volatility": "0.1~0.5%",
        "max_drawdown": "0%",
        "sharpe_ratio": "N/A（风险极低）",
        "mgmt_fee": "0.15~0.33%",
        "description": "流动性高、风险极低的现金管理工具",
    },
    "指数型": {
        "annual_return_1y": "6~12%",
        "annual_return_3y": "3~10%",
        "annual_volatility": "16~24%",
        "max_drawdown": "-20~-38%",
        "sharpe_ratio": "0.3~0.6",
        "mgmt_fee": "0.15~0.5%",
        "description": "跟踪特定指数，费率低，适合看好某一市场/行业的投资者",
    },
    "QDII": {
        "annual_return_1y": "5~18%",
        "annual_return_3y": "3~15%",
        "annual_volatility": "15~25%",
        "max_drawdown": "-20~-40%",
        "sharpe_ratio": "0.3~0.7",
        "mgmt_fee": "1.0~1.8%",
        "description": "投资海外市场，分散A股风险，但需注意汇率波动",
    },
    "ETF": {
        "annual_return_1y": "6~12%",
        "annual_return_3y": "3~10%",
        "annual_volatility": "16~24%",
        "max_drawdown": "-20~-38%",
        "sharpe_ratio": "0.3~0.6",
        "mgmt_fee": "0.15~0.5%",
        "description": "交易所交易基金，流动性好、费率最低、持仓透明",
    },
    "LOF": {
        "annual_return_1y": "5~12%",
        "annual_return_3y": "3~10%",
        "annual_volatility": "14~22%",
        "max_drawdown": "-18~-35%",
        "sharpe_ratio": "0.3~0.7",
        "mgmt_fee": "0.5~1.5%",
        "description": "上市型开放式基金，兼具场内交易和场外申赎的灵活性",
    },
}

# 默认基准（无法确定类型时使用）
DEFAULT_BENCHMARK = {
    "annual_return_1y": "视市场而定",
    "annual_return_3y": "视市场而定",
    "annual_volatility": "视类型而定",
    "max_drawdown": "视类型而定",
    "sharpe_ratio": "视类型而定",
    "mgmt_fee": "0.5~1.5%",
    "description": "基金类型未确定，基准对比仅供参考",
}


def identify_fund_type(fund_type_str: str) -> str:
    """从基金类型字符串中识别主要类型。"""
    if not fund_type_str:
        return ""
    s = fund_type_str.strip()
    for key in ["股票型", "混合型", "债券型", "货币型", "指数型", "QDII", "ETF", "LOF"]:
        if key in s:
            return key
    return ""


def get_fund_benchmark(fund_type_str: str) -> dict:
    """获取基金类型的同类基准指标。"""
    ftype = identify_fund_type(fund_type_str)
    if ftype and ftype in FUND_TYPE_BENCHMARKS:
        return FUND_TYPE_BENCHMARKS[ftype]
    return dict(DEFAULT_BENCHMARK)
