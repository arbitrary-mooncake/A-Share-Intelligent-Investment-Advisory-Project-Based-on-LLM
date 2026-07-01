"""
组合健康度 + 偏好匹配度评估 — 提供提示词模板和数据准备函数。

PortfolioAssessor 类负责构建供 MiMo-V2.5-Pro 使用的评估提示词模板，
包含组合健康度（五色评估）和投资偏好匹配度（四级评估）两大能力。

Usage:
    from src.advisory.portfolio_assessor import PortfolioAssessor

    assessor = PortfolioAssessor()
    ctx = assessor.prepare_health_context(pf)
    # 将 ctx["prompt"] 传入 MiMo-V2.5-Pro，解析 JSON 返回

    pctx = assessor.prepare_preference_context(pf, user_profile)
    # 将 pctx["prompt"] 传入 MiMo-V2.5-Pro，解析 JSON 返回
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.advisory.schemas import (
    AdvisoryPortfolio,
    HealthAssessment,
    HealthColor,
    MatchLevel,
    PreferenceMatch,
)
from src.advisory.recommendation import get_recommendation_engine
from src.utils.industry_knowledge import identify_industry

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# 提示词模板
# ──────────────────────────────────────────────────────────────────────

HEALTH_ASSESSMENT_PROMPT = """你是一位专业的A股投资组合诊断分析师。请根据以下投资组合数据，对组合的健康度进行五色评估。

## 五色健康度定义

- **深绿 (deep_green)**：组合非常健康。行业高度分散（覆盖 4+ 行业），单股权重均不超过 20%，无估值过高持仓，风险分散充分，个股质量普遍优秀。无需调整。
- **浅绿 (light_green)**：组合比较健康。行业较分散（覆盖 3+ 行业），单股权重大部分在 20% 以内，少量持仓估值略高但不极端，风险分散基本到位，个股质量良好。可微调。
- **黄色 (yellow)**：组合存在一定风险。可能出现行业集中度过高（1-2 个行业占比超 60%）、单股权重接近 30%、部分持仓估值偏高、风险分散不足、或个别持仓质量待验证。需要关注。
- **橙色 (orange)**：组合风险较明显。行业高度集中（单行业超 70%）、存在超重仓（单股权重超 35%）、多只持仓估值显著偏高、风险分散严重不足、或持仓中有明确的质量风险标的。建议调整。
- **红色 (red)**：组合处于危险状态。极端集中（单行业超 85% 或单只超 50%）、估值泡沫、几乎无风险分散、或持有存在重大瑕疵的个股。必须调整。

## 评估维度

请从以下五个维度进行综合分析：

1. **行业集中度**：持仓覆盖的行业数量、前三大行业权重占比。单一行业占比越高，风险越集中。
2. **单股权重**：最大权重持仓占比。单只股票占比过高会增加非系统性风险。
3. **估值合理性**：参考各持仓所属行业的典型 PE/PB 区间，判断是否整体偏高或偏低。注意不同行业的估值水平天然不同（如银行 PE 天然低，科技股 PE 天然高）。
4. **风险分散程度**：持仓数量、行业多样性、以及持仓之间的相关性。组合中是否存在同板块集中持仓。
5. **个股质量**：结合各持仓的分析缓存评分和信号包信息，评估持仓中是否有明确的质量风险或负面信号。

## 输入数据

**组合概况**：
- 组合名称：{portfolio_name}
- 总市值：{total_market_value:.2f}
- 总盈亏：{total_pnl:+.2f}（{total_pnl_pct:+.2f}%）
- 持仓数量：{holdings_count} 只
- 现金余额：{cash:.2f}

**各持仓详细数据**：
{holdings_detail}

## 输出要求

仅返回 JSON 格式，不要输出任何其他文字：

```json
{{
  "color": "deep_green",
  "summary": "用 2-3 句中文概括组合的健康状况、主要风险点和建议方向。"
}}
```

color 必须为以下之一：deep_green, light_green, yellow, orange, red。
summary 需针对具体持仓数据给出有实质内容的判断，避免泛泛而谈。
"""

PREFERENCE_MATCH_PROMPT = """你是一位专业的投资顾问，正在评估当前投资组合与用户投资偏好的匹配程度。请根据以下数据做出四级匹配判断。

## 四级匹配定义

- **high（高度匹配）**：组合配置与用户偏好高度一致。风险等级匹配、投资周期匹配、持仓行业方向符合用户偏好且无回避板块，投资风格一致。用户对组合的满意度预计很高。
- **basic（基本匹配）**：组合配置与用户偏好的核心方向一致，但存在一些次要偏差。如风险等级或投资周期匹配，但部分持仓行业不在偏好范围内、或轻微涉及回避板块，但不构成重大冲突。
- **partial（部分匹配）**：组合配置与用户偏好存在较明显的差异。如风险等级或投资周期不完全匹配、多个持仓板块与偏好不符、或较大比例持仓落在回避板块内。用户可能需要接受一定的方向性妥协。
- **deviated（偏离）**：组合配置与用户偏好显著偏离。风险等级完全相反（如保守型用户的组合却持有高波动标的）、投资周期矛盾、大量持仓在回避板块、或风格完全不符。用户很可能不满意。

## 用户投资偏好

以下为用户画像中的投资偏好设置：

```
风险承受能力：{risk_tolerance}
投资周期偏好：{investment_horizon}
投资风格：{investment_style}
偏好板块：{favorite_sectors}
回避板块：{avoid_sectors}
```

## 当前组合概况

以下为投资组合的概要信息：

```
组合名称：{portfolio_name}
总市值：{total_market_value:.2f}
持仓数量：{holdings_count} 只
现金占比：{cash_pct:.1f}%
```

**持仓板块分布**：
{sector_distribution}

**各持仓详情**：
{holdings_detail}

## 输出要求

仅返回 JSON 格式，不要输出任何其他文字：

```json
{{
  "level": "high",
  "reason": "用 2-3 句中文说明匹配/不匹配的具体原因，指出关键一致点和冲突点。"
}}
```

level 必须为以下之一：high, basic, partial, deviated。
reason 需要结合用户画像和持仓的具体数据，给出有实质内容的分析。
"""

# ──────────────────────────────────────────────────────────────────────
# PortfolioAssessor 类
# ──────────────────────────────────────────────────────────────────────


class PortfolioAssessor:
    """组合健康度 + 偏好匹配度评估器。

    职责：
    - 构建供 MiMo-V2.5-Pro 调用使用的组合健康度评估提示词。
    - 构建偏好匹配度评估提示词。
    - 通过缓存数据为 LLM 提供持仓分析辅助信息。
    - 根据股票代码/公司名称推测行业归属。
    """

    def __init__(self) -> None:
        """初始化 PortfolioAssessor，通过 get_recommendation_engine() 获取推荐引擎实例。"""
        self._engine = get_recommendation_engine()
        logger.info("PortfolioAssessor 初始化完成")

    # ── 公共方法 ──

    def prepare_health_context(
        self, pf: AdvisoryPortfolio
    ) -> Dict[str, Any]:
        """构建组合健康度评估的提示词和数据上下文。

        遍历持仓，查询生产缓存中的打分和 signal_pack，
        将数据注入 HEALTH_ASSESSMENT_PROMPT。

        Args:
            pf: 待评估的 AdvisoryPortfolio 实例。

        Returns:
            {"prompt": str, "holdings_count": int}
              - prompt: 注入数据后的完整健康度评估提示词。
              - holdings_count: 持仓数量。
        """
        holdings_list = list(pf.holdings.values())
        holdings_count = len(holdings_list)

        # 构建各持仓的详细文本
        holdings_detail_lines: List[str] = []
        for i, h in enumerate(holdings_list, 1):
            sector = self._guess_holding_sector(h.stock_code, h.company_name)
            score = self._engine.check_score_cache(h.stock_code)

            # 检查 signal_pack 缓存（查看是否有风险信号）
            risk_flags: List[str] = []
            for agent in (
                "fundamental",
                "technical",
                "value",
                "news",
                "event",
                "quality_risk",
                "moneyflow",
            ):
                sp = self._engine.check_signal_pack_cache(h.stock_code, agent)
                if sp and isinstance(sp, dict):
                    flags = sp.get("risk_flags", []) or []
                    risk_flags.extend(flags)

            lines = [
                f"  {i}. **{h.company_name}（{h.stock_code}）**",
                f"     - 市值占比：{h.weight:.1f}%",
                f"     - 持仓数量：{h.quantity} 股",
                f"     - 成本价：{h.cost_price:.2f}",
                f"     - 现价：{h.current_price:.2f}",
                f"     - 盈亏：{h.market_value - h.quantity * h.cost_price:+.2f}",
                f"     - 推测行业：{sector if sector else '未识别'}",
                f"     - 缓存评分：{f'{score:.1f}' if score is not None else '无缓存'}",
            ]
            if risk_flags:
                lines.append(f"     - 风险信号（{len(risk_flags)}项）：{'；'.join(risk_flags[:5])}")
            lines.append("")
            holdings_detail_lines.extend(lines)

        holdings_detail = "\n".join(holdings_detail_lines).strip()

        prompt = HEALTH_ASSESSMENT_PROMPT.format(
            portfolio_name=pf.name or "未命名组合",
            total_market_value=pf.total_market_value,
            total_pnl=pf.total_pnl,
            total_pnl_pct=pf.total_pnl_pct,
            holdings_count=holdings_count,
            cash=pf.cash,
            holdings_detail=holdings_detail,
        )

        return {"prompt": prompt, "holdings_count": holdings_count}

    def prepare_preference_context(
        self, pf: AdvisoryPortfolio, user_profile: Dict[str, Any]
    ) -> Dict[str, str]:
        """构建偏好匹配度评估的提示词和数据上下文。

        对比持仓概况和用户画像，将数据注入 PREFERENCE_MATCH_PROMPT。

        Args:
            pf: 待评估的 AdvisoryPortfolio 实例。
            user_profile: 用户画像字典，包含 risk_tolerance、investment_horizon、
                          favorite_sectors、avoid_sectors、investment_style 等字段。

        Returns:
            {"prompt": str}
              - prompt: 注入数据后的完整偏好匹配度评估提示词。
        """
        holdings_list = list(pf.holdings.values())
        holdings_count = len(holdings_list)

        # 板块分布统计
        sector_counts: Dict[str, int] = {}
        sector_values: Dict[str, float] = {}
        for h in holdings_list:
            sector = self._guess_holding_sector(h.stock_code, h.company_name)
            sector_name = sector or "其他"
            sector_counts[sector_name] = sector_counts.get(sector_name, 0) + 1
            sector_values[sector_name] = (
                sector_values.get(sector_name, 0.0) + h.market_value
            )

        total_mv = pf.total_market_value if pf.total_market_value > 0 else 1
        sector_lines: List[str] = []
        for sector_name in sorted(sector_values.keys(), key=lambda s: sector_values[s], reverse=True):
            pct = sector_values[sector_name] / total_mv * 100
            sector_lines.append(
                f"  - {sector_name}：{sector_counts[sector_name]} 只，市值占比 {pct:.1f}%"
            )
        sector_distribution = "\n".join(sector_lines)

        # 各持仓详情
        holdings_detail_lines: List[str] = []
        for i, h in enumerate(holdings_list, 1):
            sector = self._guess_holding_sector(h.stock_code, h.company_name)
            holdings_detail_lines.append(
                f"  {i}. {h.company_name}（{h.stock_code}）："
                f"权重 {h.weight:.1f}%，行业 {sector if sector else '未知'}"
            )
        holdings_detail = "\n".join(holdings_detail_lines)

        cash_pct = (
            pf.cash / (pf.total_market_value + pf.cash) * 100
            if (pf.total_market_value + pf.cash) > 0
            else 0.0
        )

        prompt = PREFERENCE_MATCH_PROMPT.format(
            risk_tolerance=user_profile.get("risk_tolerance", "Unknown"),
            investment_horizon=user_profile.get("investment_horizon", "Unknown"),
            investment_style=user_profile.get("investment_style", "未设置"),
            favorite_sectors="、".join(user_profile.get("favorite_sectors", [])) or "无指定",
            avoid_sectors="、".join(user_profile.get("avoid_sectors", [])) or "无指定",
            portfolio_name=pf.name or "未命名组合",
            total_market_value=pf.total_market_value,
            holdings_count=holdings_count,
            cash_pct=cash_pct,
            sector_distribution=sector_distribution,
            holdings_detail=holdings_detail,
        )

        return {"prompt": prompt}

    # ── 内部方法 ──

    def _guess_holding_sector(self, stock_code: str, company_name: str) -> Optional[str]:
        """根据股票代码和公司名称推测行业。

        内部使用 industry_knowledge.identify_industry()，
        优先根据公司名称的关键词匹配申万一级行业分类。

        Args:
            stock_code: 股票代码（如 600519.SH / sz.000498）。
            company_name: 公司名称（如 "贵州茅台"）。

        Returns:
            申万一级行业名称，无法识别时返回 None。
        """
        # 用 company_name 做关键词匹配
        sector = identify_industry(company_name)
        if sector:
            return sector

        # 补充：从 stock_code 提取前缀做简单行业映射
        # 处理 "sz.000498" 或 "600519.SH" 等格式
        clean_code = stock_code.replace(".", "").upper()
        code_num = ""
        for ch in clean_code:
            if ch.isdigit():
                code_num += ch
            else:
                break

        if code_num:
            try:
                num = int(code_num)
                # 600000-609999: 上海主板 (金融、传统行业)
                # 000000-009999: 深圳主板
                # 300000-309999: 创业板 (科技、医药、新兴)
                # 688000-689999: 科创板 (硬科技)
                # 002000-002999: 中小板
                if 600000 <= num <= 609999:
                    # 多数在上交所主板，无法精确推断行业，返回"主板"
                    return "主板"
                elif 300000 <= num <= 309999:
                    return "创业板"
                elif 688000 <= num <= 689999:
                    return "科创板"
                elif 2000 <= num <= 2999 or 20000 <= num <= 29999:
                    return "中小板"
                elif 0 <= num <= 9999:
                    return "深圳主板"
            except (ValueError, IndexError):
                pass

        return None

    def _guess_sectors(self, pf: AdvisoryPortfolio) -> List[str]:
        """遍历组合中所有持仓，返回去重的行业列表。

        供外部快速获取组合的行业分布概况。

        Args:
            pf: 待分析的投资组合。

        Returns:
            去重后的行业名称列表。
        """
        sectors: set[str] = set()
        for h in pf.holdings.values():
            sector = self._guess_holding_sector(h.stock_code, h.company_name)
            if sector:
                sectors.add(sector)
        return sorted(sectors)
