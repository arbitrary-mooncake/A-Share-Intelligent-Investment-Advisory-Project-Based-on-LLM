"""
IndustryKnowledge: A股行业分类 + 各行业估值基准 + 打分自适应规则

核心设计理念：
1. 不同行业采用不同的估值方法和打分标准
2. 通过行业相对分位（而非绝对PE/PB数值）保证跨行业可比性
3. 权重框架统一不变，但每个维度内的打分参考随行业调整

数据截至2026年4月，基于申万一级行业分类。
"""
from typing import Dict, Any, Optional

# ============================================================================
# A股行业分类与估值基准（2026年4月最新）
# ============================================================================

# 申万一级行业估值基准表
# 数据来源：东方财富估值数据中心、Wind
# PE/PB为板块中位数，历史分位基于过去5年数据
INDUSTRY_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    # ---------- 金融 ----------
    "银行": {
        "category": "金融",
        "investment_style": "价值",
        "primary_valuation": "PB-ROE",
        "secondary_valuation": "股息率",
        "pe_median": 6.5,
        "pb_median": 0.65,
        "pe_cheap_threshold": 5.5,    # 低于此值视为显著低估
        "pe_expensive_threshold": 8.0,  # 高于此值视为高估
        "pb_cheap_threshold": 0.55,
        "pb_expensive_threshold": 0.85,
        "roe_typical": 10.0,          # 典型ROE水平(%)
        "growth_typical": 5.0,         # 典型增速(%)
        "dividend_typical": 4.5,       # 典型股息率(%)
        "pe_reasonable_range": "5-8倍",
        "pb_reasonable_range": "0.5-0.9倍",
        "key_metrics": ["PB", "ROE", "不良率", "净息差", "股息率"],
        "scoring_notes": "银行股PE天然低，不能用绝对PE打分；重点看PB-ROE匹配度和股息率",
    },
    "非银金融": {
        "category": "金融",
        "investment_style": "价值",
        "primary_valuation": "PB",
        "secondary_valuation": "PE",
        "pe_median": 15.0,
        "pb_median": 1.3,
        "pe_cheap_threshold": 10.0,
        "pe_expensive_threshold": 22.0,
        "pb_cheap_threshold": 1.0,
        "pb_expensive_threshold": 1.8,
        "roe_typical": 8.0,
        "growth_typical": 8.0,
        "dividend_typical": 2.5,
        "pe_reasonable_range": "10-22倍",
        "pb_reasonable_range": "1.0-1.8倍",
        "key_metrics": ["PB", "ROE", "保费增速", "投资收益"],
        "scoring_notes": "券商看PB和资本市场活跃度，保险看EV（内含价值）和NBV",
    },

    # ---------- 科技 ----------
    "计算机": {
        "category": "科技",
        "investment_style": "成长",
        "primary_valuation": "PS",
        "secondary_valuation": "PEG",
        "pe_median": 55.0,
        "pb_median": 5.5,
        "pe_cheap_threshold": 30.0,
        "pe_expensive_threshold": 100.0,
        "pb_cheap_threshold": 3.0,
        "pb_expensive_threshold": 9.0,
        "roe_typical": 8.0,
        "growth_typical": 25.0,
        "dividend_typical": 0.5,
        "pe_reasonable_range": "30-100倍",
        "pb_reasonable_range": "3-9倍",
        "key_metrics": ["营收增速", "PS", "PEG", "研发占比", "订单增速"],
        "scoring_notes": "计算机股PE天然高，不能用绝对PE打分；重点看PS和营收增速，PE>80不一定贵",
    },
    "电子": {
        "category": "科技",
        "investment_style": "成长",
        "primary_valuation": "PE",
        "secondary_valuation": "PEG",
        "pe_median": 45.0,
        "pb_median": 4.0,
        "pe_cheap_threshold": 25.0,
        "pe_expensive_threshold": 80.0,
        "pb_cheap_threshold": 2.5,
        "pb_expensive_threshold": 6.5,
        "roe_typical": 12.0,
        "growth_typical": 20.0,
        "dividend_typical": 1.0,
        "pe_reasonable_range": "25-80倍",
        "pb_reasonable_range": "2.5-6.5倍",
        "key_metrics": ["PE", "PEG", "毛利率", "研发占比", "产能利用率"],
        "scoring_notes": "半导体PE高但增速快，看PEG<1为低估；消费电子看产品周期",
    },
    "通信": {
        "category": "科技",
        "investment_style": "成长",
        "primary_valuation": "PE",
        "secondary_valuation": "PS",
        "pe_median": 40.0,
        "pb_median": 3.5,
        "pe_cheap_threshold": 22.0,
        "pe_expensive_threshold": 70.0,
        "pb_cheap_threshold": 2.0,
        "pb_expensive_threshold": 5.5,
        "roe_typical": 10.0,
        "growth_typical": 18.0,
        "dividend_typical": 1.5,
        "pe_reasonable_range": "22-70倍",
        "pb_reasonable_range": "2.0-5.5倍",
        "key_metrics": ["PE", "营收增速", "5G订单", "ARPU值"],
        "scoring_notes": "通信设备看订单和5G渗透率，运营商看ARPU和分红",
    },
    "传媒": {
        "category": "科技",
        "investment_style": "成长",
        "primary_valuation": "PE",
        "secondary_valuation": "PS",
        "pe_median": 30.0,
        "pb_median": 3.0,
        "pe_cheap_threshold": 18.0,
        "pe_expensive_threshold": 55.0,
        "pb_cheap_threshold": 1.8,
        "pb_expensive_threshold": 4.5,
        "roe_typical": 10.0,
        "growth_typical": 15.0,
        "dividend_typical": 1.0,
        "pe_reasonable_range": "18-55倍",
        "pb_reasonable_range": "1.8-4.5倍",
        "key_metrics": ["PE", "版号审批", "游戏流水", "票房", "内容储备"],
        "scoring_notes": "游戏看版号审批和产品周期，影视看票房和IP储备，互联网广告看广告主预算",
    },

    # ---------- 消费 ----------
    "食品饮料": {
        "category": "消费",
        "investment_style": "价值+成长",
        "primary_valuation": "PE",
        "secondary_valuation": "DCF",
        "pe_median": 25.0,
        "pb_median": 6.0,
        "pe_cheap_threshold": 15.0,
        "pe_expensive_threshold": 45.0,
        "pb_cheap_threshold": 3.5,
        "pb_expensive_threshold": 10.0,
        "roe_typical": 20.0,
        "growth_typical": 12.0,
        "dividend_typical": 2.5,
        "pe_reasonable_range": "15-45倍",
        "pb_reasonable_range": "3.5-10倍",
        "key_metrics": ["PE", "ROE", "毛利率", "预收款", "渠道库存"],
        "scoring_notes": "白酒看品牌和渠道，食品看渠道扩张和品类延伸，ROE>20%为佳",
    },
    "医药生物": {
        "category": "消费",
        "investment_style": "成长",
        "primary_valuation": "PE",
        "secondary_valuation": "管线估值",
        "pe_median": 30.0,
        "pb_median": 4.0,
        "pe_cheap_threshold": 18.0,
        "pe_expensive_threshold": 55.0,
        "pb_cheap_threshold": 2.5,
        "pb_expensive_threshold": 6.5,
        "roe_typical": 12.0,
        "growth_typical": 15.0,
        "dividend_typical": 1.0,
        "pe_reasonable_range": "18-55倍",
        "pb_reasonable_range": "2.5-6.5倍",
        "key_metrics": ["PE", "研发管线", "集采影响", "医保谈判", "创新药占比"],
        "scoring_notes": "创新药看管线进展和审批，CXO看订单，中药看品牌和OTC渠道",
    },
    "家用电器": {
        "category": "消费",
        "investment_style": "价值",
        "primary_valuation": "PE",
        "secondary_valuation": "股息率",
        "pe_median": 14.0,
        "pb_median": 3.0,
        "pe_cheap_threshold": 10.0,
        "pe_expensive_threshold": 22.0,
        "pb_cheap_threshold": 2.0,
        "pb_expensive_threshold": 4.5,
        "roe_typical": 18.0,
        "growth_typical": 8.0,
        "dividend_typical": 3.0,
        "pe_reasonable_range": "10-22倍",
        "pb_reasonable_range": "2.0-4.5倍",
        "key_metrics": ["PE", "ROE", "毛利率", "出口增速", "高端化占比"],
        "scoring_notes": "家电看地产周期和出口，龙头看高端化出海",
    },
    "汽车": {
        "category": "消费",
        "investment_style": "周期+成长",
        "primary_valuation": "PE",
        "secondary_valuation": "PS",
        "pe_median": 20.0,
        "pb_median": 2.0,
        "pe_cheap_threshold": 12.0,
        "pe_expensive_threshold": 40.0,
        "pb_cheap_threshold": 1.2,
        "pb_expensive_threshold": 3.5,
        "roe_typical": 10.0,
        "growth_typical": 15.0,
        "dividend_typical": 2.0,
        "pe_reasonable_range": "12-40倍",
        "pb_reasonable_range": "1.2-3.5倍",
        "key_metrics": ["PE", "销量增速", "新能源占比", "单车利润", "出口"],
        "scoring_notes": "传统车看销量和利润，新能源看智能化和出海",
    },
    "美容护理": {
        "category": "消费",
        "investment_style": "成长",
        "primary_valuation": "PE",
        "secondary_valuation": "PS",
        "pe_median": 35.0,
        "pb_median": 7.0,
        "pe_cheap_threshold": 20.0,
        "pe_expensive_threshold": 60.0,
        "pb_cheap_threshold": 4.0,
        "pb_expensive_threshold": 11.0,
        "roe_typical": 18.0,
        "growth_typical": 20.0,
        "dividend_typical": 1.5,
        "pe_reasonable_range": "20-60倍",
        "pb_reasonable_range": "4-11倍",
        "key_metrics": ["PE", "营收增速", "毛利率", "品牌力", "渠道"],
        "scoring_notes": "美妆看品牌矩阵和线上渠道，高增长高估值",
    },
    "社会服务": {
        "category": "消费",
        "investment_style": "成长",
        "primary_valuation": "PE",
        "secondary_valuation": "EV/EBITDA",
        "pe_median": 30.0,
        "pb_median": 3.5,
        "pe_cheap_threshold": 18.0,
        "pe_expensive_threshold": 55.0,
        "pb_cheap_threshold": 2.0,
        "pb_expensive_threshold": 5.5,
        "roe_typical": 10.0,
        "growth_typical": 15.0,
        "dividend_typical": 1.0,
        "pe_reasonable_range": "18-55倍",
        "pb_reasonable_range": "2.0-5.5倍",
        "key_metrics": ["PE", "客流恢复", "单店收入", "扩张速度"],
        "scoring_notes": "旅游酒店看客流和ADR，教育看政策恢复",
    },

    # ---------- 周期 ----------
    "有色金属": {
        "category": "周期",
        "investment_style": "周期",
        "primary_valuation": "PB",
        "secondary_valuation": "正常化PE",
        "pe_median": 20.0,
        "pb_median": 2.0,
        "pe_cheap_threshold": 10.0,
        "pe_expensive_threshold": 40.0,
        "pb_cheap_threshold": 1.2,
        "pb_expensive_threshold": 3.5,
        "roe_typical": 8.0,
        "growth_typical": 10.0,
        "dividend_typical": 2.0,
        "pe_reasonable_range": "10-40倍",
        "pb_reasonable_range": "1.2-3.5倍",
        "key_metrics": ["PB", "商品价格", "产能利用率", "库存", "成本曲线"],
        "scoring_notes": "周期股PE低可能是周期顶部陷阱！重点看PB分位和商品价格周期",
    },
    "钢铁": {
        "category": "周期",
        "investment_style": "周期",
        "primary_valuation": "PB",
        "secondary_valuation": "正常化PE",
        "pe_median": 12.0,
        "pb_median": 0.9,
        "pe_cheap_threshold": 7.0,
        "pe_expensive_threshold": 25.0,
        "pb_cheap_threshold": 0.6,
        "pb_expensive_threshold": 1.3,
        "roe_typical": 7.0,
        "growth_typical": 3.0,
        "dividend_typical": 3.0,
        "pe_reasonable_range": "7-25倍",
        "pb_reasonable_range": "0.6-1.3倍",
        "key_metrics": ["PB", "钢价", "吨钢利润", "产能利用率"],
        "scoring_notes": "钢铁是典型周期股，PE低不等于便宜，看PB分位和供给侧",
    },
    "建筑材料": {
        "category": "周期",
        "investment_style": "周期",
        "primary_valuation": "PE",
        "secondary_valuation": "PB",
        "pe_median": 15.0,
        "pb_median": 1.8,
        "pe_cheap_threshold": 10.0,
        "pe_expensive_threshold": 28.0,
        "pb_cheap_threshold": 1.2,
        "pb_expensive_threshold": 2.8,
        "roe_typical": 12.0,
        "growth_typical": 5.0,
        "dividend_typical": 3.0,
        "pe_reasonable_range": "10-28倍",
        "pb_reasonable_range": "1.2-2.8倍",
        "key_metrics": ["PE", "水泥价格", "地产开工", "基建投资"],
        "scoring_notes": "建材看地产周期和基建，水泥龙头有区域垄断优势",
    },
    "基础化工": {
        "category": "周期",
        "investment_style": "周期",
        "primary_valuation": "PB",
        "secondary_valuation": "正常化PE",
        "pe_median": 18.0,
        "pb_median": 2.0,
        "pe_cheap_threshold": 10.0,
        "pe_expensive_threshold": 35.0,
        "pb_cheap_threshold": 1.2,
        "pb_expensive_threshold": 3.2,
        "roe_typical": 10.0,
        "growth_typical": 8.0,
        "dividend_typical": 2.0,
        "pe_reasonable_range": "10-35倍",
        "pb_reasonable_range": "1.2-3.2倍",
        "key_metrics": ["PB", "产品价格", "价差", "产能", "原油价格"],
        "scoring_notes": "化工细分多，看具体产品的供需格局和价差变化",
    },
    "煤炭": {
        "category": "周期",
        "investment_style": "周期+价值",
        "primary_valuation": "PE",
        "secondary_valuation": "股息率",
        "pe_median": 9.0,
        "pb_median": 1.2,
        "pe_cheap_threshold": 6.0,
        "pe_expensive_threshold": 15.0,
        "pb_cheap_threshold": 0.8,
        "pb_expensive_threshold": 1.8,
        "roe_typical": 15.0,
        "growth_typical": 0.0,
        "dividend_typical": 6.0,
        "pe_reasonable_range": "6-15倍",
        "pb_reasonable_range": "0.8-1.8倍",
        "key_metrics": ["PE", "股息率", "煤价", "ROE", "长协占比"],
        "scoring_notes": "煤炭看煤价和分红，高股息是核心逻辑",
    },
    "石油石化": {
        "category": "周期",
        "investment_style": "周期+价值",
        "primary_valuation": "PE",
        "secondary_valuation": "EV/EBITDA",
        "pe_median": 12.0,
        "pb_median": 1.1,
        "pe_cheap_threshold": 8.0,
        "pe_expensive_threshold": 20.0,
        "pb_cheap_threshold": 0.8,
        "pb_expensive_threshold": 1.5,
        "roe_typical": 10.0,
        "growth_typical": 3.0,
        "dividend_typical": 4.0,
        "pe_reasonable_range": "8-20倍",
        "pb_reasonable_range": "0.8-1.5倍",
        "key_metrics": ["PE", "油价", "炼化利润", "股息率"],
        "scoring_notes": "石油看油价周期，炼化看裂解价差，高分红是价值支撑",
    },
    "农林牧渔": {
        "category": "周期",
        "investment_style": "周期",
        "primary_valuation": "PB",
        "secondary_valuation": "正常化PE",
        "pe_median": 25.0,
        "pb_median": 2.5,
        "pe_cheap_threshold": 12.0,
        "pe_expensive_threshold": 50.0,
        "pb_cheap_threshold": 1.5,
        "pb_expensive_threshold": 4.0,
        "roe_typical": 5.0,
        "growth_typical": 10.0,
        "dividend_typical": 1.0,
        "pe_reasonable_range": "12-50倍",
        "pb_reasonable_range": "1.5-4.0倍",
        "key_metrics": ["PB", "猪价/鸡价", "存栏量", "成本", "疫病"],
        "scoring_notes": "养猪看猪周期，PE低可能是周期顶部，重点看能繁母猪存栏",
    },
    "交通运输": {
        "category": "周期",
        "investment_style": "价值+周期",
        "primary_valuation": "PE",
        "secondary_valuation": "EV/EBITDA",
        "pe_median": 15.0,
        "pb_median": 1.5,
        "pe_cheap_threshold": 10.0,
        "pe_expensive_threshold": 28.0,
        "pb_cheap_threshold": 1.0,
        "pb_expensive_threshold": 2.2,
        "roe_typical": 8.0,
        "growth_typical": 5.0,
        "dividend_typical": 2.5,
        "pe_reasonable_range": "10-28倍",
        "pb_reasonable_range": "1.0-2.2倍",
        "key_metrics": ["PE", "客货运量", "票价", "成本", "产能扩张"],
        "scoring_notes": "公路/港口看车流/吞吐量，航空看RPK和票价，快递看单量和市占率",
    },
    "房地产": {
        "category": "周期",
        "investment_style": "价值",
        "primary_valuation": "PB",
        "secondary_valuation": "NAV折价",
        "pe_median": 15.0,
        "pb_median": 0.7,
        "pe_cheap_threshold": 8.0,
        "pe_expensive_threshold": 25.0,
        "pb_cheap_threshold": 0.4,
        "pb_expensive_threshold": 1.1,
        "roe_typical": 5.0,
        "growth_typical": -5.0,
        "dividend_typical": 2.0,
        "pe_reasonable_range": "8-25倍",
        "pb_reasonable_range": "0.4-1.1倍",
        "key_metrics": ["PB", "销售面积", "拿地", "负债率", "土储"],
        "scoring_notes": "地产看政策周期和销售数据，PB<0.5可能隐含资产减值风险",
    },

    # ---------- 其他 ----------
    "电力设备": {
        "category": "制造",
        "investment_style": "成长",
        "primary_valuation": "PE",
        "secondary_valuation": "PEG",
        "pe_median": 25.0,
        "pb_median": 3.5,
        "pe_cheap_threshold": 15.0,
        "pe_expensive_threshold": 50.0,
        "pb_cheap_threshold": 2.0,
        "pb_expensive_threshold": 5.5,
        "roe_typical": 12.0,
        "growth_typical": 20.0,
        "dividend_typical": 1.5,
        "pe_reasonable_range": "15-50倍",
        "pb_reasonable_range": "2.0-5.5倍",
        "key_metrics": ["PE", "PEG", "装机量", "中标量", "毛利率"],
        "scoring_notes": "光伏/风电看装机和中标，电池看产能和成本",
    },
    "机械设备": {
        "category": "制造",
        "investment_style": "周期+成长",
        "primary_valuation": "PE",
        "secondary_valuation": "PB",
        "pe_median": 22.0,
        "pb_median": 2.5,
        "pe_cheap_threshold": 14.0,
        "pe_expensive_threshold": 40.0,
        "pb_cheap_threshold": 1.5,
        "pb_expensive_threshold": 4.0,
        "roe_typical": 10.0,
        "growth_typical": 10.0,
        "dividend_typical": 2.0,
        "pe_reasonable_range": "14-40倍",
        "pb_reasonable_range": "1.5-4.0倍",
        "key_metrics": ["PE", "订单增速", "产能利用率", "出口"],
        "scoring_notes": "工程机械看地产基建周期，专用设备看下游景气",
    },
    "轻工制造": {
        "category": "制造",
        "investment_style": "价值",
        "primary_valuation": "PE",
        "secondary_valuation": "PB",
        "pe_median": 18.0,
        "pb_median": 2.0,
        "pe_cheap_threshold": 12.0,
        "pe_expensive_threshold": 30.0,
        "pb_cheap_threshold": 1.3,
        "pb_expensive_threshold": 3.0,
        "roe_typical": 10.0,
        "growth_typical": 8.0,
        "dividend_typical": 2.0,
        "pe_reasonable_range": "12-30倍",
        "pb_reasonable_range": "1.3-3.0倍",
        "key_metrics": ["PE", "ROE", "出口占比", "原材料成本"],
        "scoring_notes": "轻工看出口和原材料成本，龙头看品类拓展",
    },
    "纺织服饰": {
        "category": "制造",
        "investment_style": "价值",
        "primary_valuation": "PE",
        "secondary_valuation": "PB",
        "pe_median": 16.0,
        "pb_median": 1.8,
        "pe_cheap_threshold": 10.0,
        "pe_expensive_threshold": 28.0,
        "pb_cheap_threshold": 1.2,
        "pb_expensive_threshold": 2.8,
        "roe_typical": 10.0,
        "growth_typical": 8.0,
        "dividend_typical": 2.5,
        "pe_reasonable_range": "10-28倍",
        "pb_reasonable_range": "1.2-2.8倍",
        "key_metrics": ["PE", "ROE", "品牌力", "渠道", "库存"],
        "scoring_notes": "服装看品牌和渠道，代工看出海和客户结构",
    },
    "公用事业": {
        "category": "公用",
        "investment_style": "价值",
        "primary_valuation": "PB",
        "secondary_valuation": "股息率",
        "pe_median": 15.0,
        "pb_median": 1.3,
        "pe_cheap_threshold": 10.0,
        "pe_expensive_threshold": 22.0,
        "pb_cheap_threshold": 0.9,
        "pb_expensive_threshold": 1.8,
        "roe_typical": 8.0,
        "growth_typical": 3.0,
        "dividend_typical": 3.5,
        "pe_reasonable_range": "10-22倍",
        "pb_reasonable_range": "0.9-1.8倍",
        "key_metrics": ["PB", "股息率", "电价", "利用小时", "装机"],
        "scoring_notes": "电力看电改和煤价，水务燃气看调价机制",
    },
    "环保": {
        "category": "公用",
        "investment_style": "价值",
        "primary_valuation": "PE",
        "secondary_valuation": "PB",
        "pe_median": 20.0,
        "pb_median": 1.5,
        "pe_cheap_threshold": 12.0,
        "pe_expensive_threshold": 35.0,
        "pb_cheap_threshold": 1.0,
        "pb_expensive_threshold": 2.2,
        "roe_typical": 7.0,
        "growth_typical": 5.0,
        "dividend_typical": 1.5,
        "pe_reasonable_range": "12-35倍",
        "pb_reasonable_range": "1.0-2.2倍",
        "key_metrics": ["PE", "PPP项目", "订单", "应收账款"],
        "scoring_notes": "环保看政策订单和应收账款回收",
    },
    "商贸零售": {
        "category": "消费",
        "investment_style": "价值",
        "primary_valuation": "PE",
        "secondary_valuation": "PS",
        "pe_median": 20.0,
        "pb_median": 2.0,
        "pe_cheap_threshold": 12.0,
        "pe_expensive_threshold": 35.0,
        "pb_cheap_threshold": 1.2,
        "pb_expensive_threshold": 3.0,
        "roe_typical": 8.0,
        "growth_typical": 5.0,
        "dividend_typical": 2.0,
        "pe_reasonable_range": "12-35倍",
        "pb_reasonable_range": "1.2-3.0倍",
        "key_metrics": ["PE", "GMV", "同店增长", "线上占比"],
        "scoring_notes": "零售看同店增长和线上渗透，电商看GMV和变现率",
    },
    "国防军工": {
        "category": "制造",
        "investment_style": "成长",
        "primary_valuation": "PE",
        "secondary_valuation": "PS",
        "pe_median": 50.0,
        "pb_median": 4.0,
        "pe_cheap_threshold": 30.0,
        "pe_expensive_threshold": 90.0,
        "pb_cheap_threshold": 2.5,
        "pb_expensive_threshold": 6.5,
        "roe_typical": 8.0,
        "growth_typical": 20.0,
        "dividend_typical": 0.5,
        "pe_reasonable_range": "30-90倍",
        "pb_reasonable_range": "2.5-6.5倍",
        "key_metrics": ["PE", "订单", "交付", "研发", "军费增速"],
        "scoring_notes": "军工看订单和交付确定性，PE高但有订单支撑",
    },
    "综合": {
        "category": "其他",
        "investment_style": "综合",
        "primary_valuation": "PE",
        "secondary_valuation": "PB",
        "pe_median": 25.0,
        "pb_median": 2.0,
        "pe_cheap_threshold": 15.0,
        "pe_expensive_threshold": 45.0,
        "pb_cheap_threshold": 1.2,
        "pb_expensive_threshold": 3.0,
        "roe_typical": 8.0,
        "growth_typical": 8.0,
        "dividend_typical": 1.5,
        "pe_reasonable_range": "15-45倍",
        "pb_reasonable_range": "1.2-3.0倍",
        "key_metrics": ["PE", "PB", "ROE"],
        "scoring_notes": "综合类企业看主业和资产质量",
    },
}

# 行业关键词映射：从分析文本中自动识别行业
# 当用户只给股票代码/名称时，通过关键词匹配推断行业
INDUSTRY_KEYWORDS: Dict[str, list] = {
    "银行": ["银行", "农商", "股份行", "城商行"],
    "非银金融": ["证券", "保险", "券商", "信托", "期货", "多元金融"],
    "计算机": ["软件", "计算机", "IT服务", "云计算", "大数据", "人工智能", "AI", "SaaS", "网络安全"],
    "电子": ["半导体", "芯片", "电子", "显示", "LED", "PCB", "被动元件", "消费电子"],
    "通信": ["通信", "5G", "光纤", "光模块", "运营商", "移动", "联通", "电信"],
    "传媒": ["传媒", "游戏", "影视", "广告", "出版", "互联网"],
    "食品饮料": ["白酒", "啤酒", "食品", "饮料", "乳制品", "调味品", "零食", "酿酒"],
    "医药生物": ["医药", "生物", "制药", "医疗器械", "中药", "创新药", "CXO", "医疗服务", "疫苗"],
    "家用电器": ["家电", "空调", "冰箱", "洗衣机", "小家电", "厨电"],
    "汽车": ["汽车", "新能源", "电动车", "零部件", "轮胎", "整车", "智能驾驶"],
    "美容护理": ["美容", "化妆品", "个护"],
    "社会服务": ["旅游", "酒店", "餐饮", "教育", "培训"],
    "有色金属": ["有色", "铜", "铝", "锂", "钴", "镍", "稀土", "黄金", "贵金属"],
    "钢铁": ["钢铁", "特钢", "钢管"],
    "建筑材料": ["水泥", "建材", "玻璃", "玻纤", "防水材料"],
    "基础化工": ["化工", "化肥", "农药", "橡胶", "塑料", "染料", "涂料"],
    "煤炭": ["煤炭", "焦炭", "煤化工"],
    "石油石化": ["石油", "石化", "炼化", "油田"],
    "农林牧渔": ["农业", "养殖", "猪", "鸡", "饲料", "种子", "种业", "渔业"],
    "交通运输": ["航空公司", "机场", "港口", "高速公路", "物流", "快递", "航运", "铁路"],
    "房地产": ["房地产", "地产", "物业"],
    "电力设备": ["电力设备", "光伏", "风电", "电池", "储能", "新能源车", "锂电", "逆变器"],
    "机械设备": ["机械", "工程机械", "机床", "机器人", "泵阀"],
    "轻工制造": ["造纸", "家具", "包装", "印刷", "文具"],
    "纺织服饰": ["纺织", "服装", "鞋类", "家纺"],
    "公用事业": ["电力", "水务", "燃气", "供热"],
    "环保": ["环保", "污水处理", "固废", "大气治理"],
    "商贸零售": ["零售", "商贸", "百货", "超市", "电商"],
    "国防军工": ["军工", "航天", "航空发动机", "兵器", "船舶", "国防", "武器装备"],
}


def identify_industry(company_name: str, analysis_text: str = "") -> Optional[str]:
    """
    从公司名称和分析文本中推断行业

    Args:
        company_name: 公司名称
        analysis_text: 基本面/新闻等分析文本

    Returns:
        行业名称（申万一级行业），如果无法识别返回None
    """
    combined = company_name + " " + analysis_text

    # 精确匹配优先（检查公司名称是否包含行业关键词）
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in combined.lower():
                return industry

    return None


def get_industry_info(industry: str) -> Optional[Dict[str, Any]]:
    """获取行业估值基准信息"""
    return INDUSTRY_BENCHMARKS.get(industry)


def get_industry_scoring_guidance(industry: str) -> str:
    """
    获取特定行业的打分指导建议

    这个字符串会被注入到打分Agent的system prompt中，
    让LLM知道当前行业应该重点关注什么、如何正确解读估值指标。
    """
    info = INDUSTRY_BENCHMARKS.get(industry)
    if not info:
        return ""

    guidance_parts = [
        f"### 行业识别：{industry}（{info['category']}）",
        f"- 投资风格：{info['investment_style']}",
        f"- 主要估值方法：{info['primary_valuation']}，辅助方法：{info['secondary_valuation']}",
        f"- 行业中位数：PE约{info['pe_median']}倍，PB约{info['pb_median']}倍",
        f"- 合理估值区间：PE {info['pe_reasonable_range']}，PB {info['pb_reasonable_range']}",
        f"- 低估参考：PE<{info['pe_cheap_threshold']}倍为显著低估，PE>{info['pe_expensive_threshold']}倍为高估",
        f"- 典型ROE：{info['roe_typical']}%，典型增速：{info['growth_typical']}%，典型股息率：{info['dividend_typical']}%",
        f"- 核心关注指标：{', '.join(info['key_metrics'])}",
        f"- **重要提醒**：{info['scoring_notes']}",
    ]

    return "\n".join(guidance_parts)


def generate_industry_context_prompt(industry: Optional[str]) -> str:
    """
    生成注入到打分Agent prompt中的行业上下文指引

    如果识别到行业，注入详细的行业估值基准和打分调整指引。
    如果未识别到行业，给出通用的行业差异提醒。

    这个方法确保评分跨行业可比：
    - 不是用绝对PE/PB打分，而是用相对于行业均值的百分位打分
    - 权重框架统一不变（保证跨行业可比）
    - 每个维度内的打分参考随行业调整
    """
    if industry and industry in INDUSTRY_BENCHMARKS:
        guidance = get_industry_scoring_guidance(industry)
        return f"""
## 行业自适应打分指引

你当前分析的股票属于**【{industry}】**行业。

{guidance}

### 跨行业可比性要求

重要：你的打分必须保证跨行业可比性。具体来说：
1. **不要使用绝对PE/PB数值打分**——银行的PE 6倍可能是合理估值，科技的PE 60倍也可能是低估
2. **使用相对行业均值的百分位打分**——重点看当前估值相对该行业历史5年分位和行业均值的偏离
3. **权重框架保持不变**——各行业维度权重相同，但每个维度内的打分参考标准随行业调整
4. 对于成长性评估，不要用通用标准——银行5%的增速已经算不错，但科技股5%的增速是严重不及格
"""
    else:
        return """
## 行业差异提醒

当前未能明确识别股票所属行业。请注意：

1. **不同行业的估值方法和合理区间天差地别**：
   - 银行：PE 5-8倍合理，PB 0.5-0.9倍合理
   - 消费：PE 15-45倍合理（白酒龙头20-30倍）
   - 科技：PE 30-100倍合理（计算机板块中位数55倍）
   - 医药：PE 18-55倍合理
   - 周期：PE低可能是周期顶部陷阱，重点看PB分位

2. **请务必使用相对行业均值的百分位进行估值打分**，不要使用绝对数值。
3. **请根据分析文本中提到的行业信息自行判断行业属性**，并相应调整打分标准。
4. 如果无法判断行业，请使用通用标准但仍需警惕PE陷阱等问题。
"""
