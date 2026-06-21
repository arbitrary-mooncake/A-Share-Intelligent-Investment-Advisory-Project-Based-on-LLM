# A股主线架构升级实施大纲（补充 CLAUDE.md）

## 1. 文档定位

- 本文是对当前 `CLAUDE.md` 的补充实施大纲，不替代原文。
- 原有项目结构、命令、接口、模型配置、Windows/WSL 兼容方式仍以现有 `CLAUDE.md` 为准。
- 本次改造遵循四个原则：
  1. 最小侵入
  2. 向后兼容
  3. 分阶段落地
  4. 优先复用现有 MCP / Tushare / AKShare / LangGraph 基础设施
- 如果理想设计与当前项目冲突较大，优先实现“兼容版 / 降级版”，不要为了追求完美而重写整套系统。

## 2. 本次改造的范围与非目标

## 2.1 本次改造重点（只聚焦 A 股主线）

本次只重点改造以下主线：

1. A 股单票深度分析报告  
   - 入口：`src/main.py`
   - API：`POST /api/report`、`GET /api/report/{task_id}`

2. A 股股票池评分  
   - 入口：`src/stock_pool/scoring_engine.py`
   - 相关命令：`python -m src.main_pool score ...`
   - API：`POST /api/score/{term}/{stock_code}`、`POST /api/score-all/{stock_code}`

3. A 股第一层分析 agent 的中间产物扩展  
   - 当前 4 个 agent：`fundamental / technical / value / news`
   - 目标：在保持兼容的前提下，扩展为“结构化证据 + 新增关键维度”

4. A 股打分 agent 与总结报告 agent 的消费方式升级  
   - 从“只读自然语言分析”升级为“优先读结构化证据，辅以自然语言总结”
   - 增加简单风险门控（risk gate），但不做完整的客户适当性系统

## 2.2 本次明确不作为重点改造的内容

以下模块不要作为本次主要改造对象：

- 基金分析全链路（`src/fund_main.py` 及其 7-agent pipeline）
- 智能问答（`src/qa/`）
- 快速查询接口（`POST /api/query`）
- 快筛 / 批量打分路径（`quick-screen`、`batch-score`）  
  - 允许复用新建的工具函数或 schema
  - 但不要强制把它们一起重构
- MCP 协议、FastMCP 通信方式、Windows 兼容层
- 前端 Streamlit 页面的大改版

## 3. 本次改造的硬约束

1. 不要破坏现有 CLI 与 API 的使用方式。
2. 不要删除现有 4 个 analysis key：
   - `fundamental_analysis`
   - `technical_analysis`
   - `value_analysis`
   - `news_analysis`
3. 新增能力时，优先以“新增字段”而不是“替换旧字段”的方式做。
4. `stock_pool.json`、现有 API 响应、现有前端页面，必须尽量保持可兼容。
5. 尽量不新增新的环境变量，不新增新的模型槽位；优先复用现有 `5-model architecture`。
6. 尽量不增加重型新依赖；优先使用现有依赖和标准库。
7. 如果某类数据当前 MCP / Tushare 暂时拿不到，不要阻塞整个改造，应做“降级实现 + 明确标记置信度下降”。

## 4. 总体目标架构（基于当前项目演进，而不是推翻重写）

## 4.1 目标：保留两层架构，但增强第一层

当前主线：

- 单票分析：  
  `start_node → [fundamental, technical, value, news] → summarizer`

- 股票池评分：  
  `start_node → [fundamental, technical, value, news] → [short, medium, long]`

目标主线（建议）：

- 单票分析：  
  `start_node → [fundamental, technical, value, news, event, quality_risk, moneyflow] → summarizer`

- 股票池评分：  
  `start_node → [fundamental, technical, value, news, event, quality_risk, moneyflow] → [short, medium, long] → risk_gate(可做成后处理，不一定单独成图节点)`

说明：

- `event`：公告/监管/公司重大事项/事件驱动
- `quality_risk`：财务质量 / 治理风险 / 雷点排查
- `moneyflow`：资金面 / 量价确认 / 微观结构代理
- 如果 `moneyflow` 一期难做成独立 agent，可先并入 `technical`，但要预留独立 key 和模块接口，后续便于拆分。

## 4.2 不强制增加新的 LangGraph merge node

为了降低对现有图结构的冲击：

- 第一选择：各 agent 继续并行写入 `state.data` 的独立字段
- 在 `scoring_nodes.py` 和 `summarizer` 中调用纯 Python 工具函数，例如：
  - `build_analysis_package()`
  - `apply_risk_gate()`

只有在当前代码难以维护时，才考虑增加一个纯 Python 的 merge node；不要为了“图结构更漂亮”而引入新的 LLM node。

## 5. 分阶段实施优先级

## 5.1 P0（必须完成）

1. 扩展现有 4 个 agent 的分析深度
2. 新增 `event_analyst`
3. 新增 `quality_risk_analyst`
4. 引入结构化中间产物（signal pack / analysis package）
5. 改造 short / medium / long scorer，让其优先读取结构化证据
6. 为总结报告增加“反证 / 风险 / 数据缺口”部分
7. 加入轻量级 `risk_gate` 后处理

## 5.2 P1（应该完成）

1. 新增独立 `moneyflow_analyst`
2. 在 stock pool 评分结果里追加可选字段：
   - `confidence`
   - `risk_flags`
   - `key_drivers`
   - `abstain`
3. 让报告和股票池详情页可展示更多结构化信息（如果改动较小）

## 5.3 P2（可选，不要喧宾夺主）

1. 增加市场状态 / 风格切换 helper（不是必须新增 agent）
2. 增加行业/景气辅助上下文 builder（优先复用 `industry_knowledge.py`）
3. 后续如果 MCP 工具支持更完整公告/交易数据，再升级 `event` / `moneyflow`

## 6. 对现有 4 个 agent 的具体改造要求

## 6.1 fundamental_analyst：从“基本面描述”升级为“基本面质量分析”

保留现有 `fundamental_analysis` 文本输出，但新增结构化输出。

重点增加以下内容：

1. 盈利质量  
   - 收入与利润是否匹配
   - 毛利率/净利率变化趋势
   - ROE / ROIC / 资本回报能力
   - 非经常性损益依赖度（如果数据可得）

2. 现金流质量  
   - 经营现金流与净利润匹配程度
   - 现金流是否明显弱于利润表表现

3. 资产负债表健康度  
   - 负债压力
   - 应收账款、存货、商誉等风险点（如果数据可得）
   - 偿债能力与财务弹性

4. 成长持续性  
   - 增长来源是否可持续
   - 一次性驱动 vs 持续性驱动

5. 资本配置与股东回报（能拿到就做）  
   - 分红、回购、资本开支强度等

fundamental 不要再承担以下职责：

- 不要把“市场情绪”“媒体叙事”混进来
- 不要把“估值便宜/贵”作为主结论
- 不要承担“公告事件判断”

## 6.2 technical_analyst：从“技术指标集合”升级为“技术状态 + 量价行为”

重点增加以下内容：

1. 趋势状态  
   - 上升/下降/震荡
   - 趋势延续还是衰减
   - 是否临近关键支撑/阻力

2. 量价关系  
   - 放量上涨、缩量调整、放量滞涨等
   - 成交量与价格是否共振

3. 波动与交易质量  
   - 波动率是否异常
   - 振幅、跳空、上影/下影等行为特征

4. 流动性代理  
   - 成交额、换手率、连续性
   - 是否适合短线执行

5. 为 `moneyflow_analyst` 预留边界  
   - technical 负责“图形与量价状态”
   - moneyflow 负责“资金行为与资金确认”
   - 一期如果 moneyflow 尚未独立，technical 可暂时兼任其部分逻辑，但输出时仍建议拆为不同字段

technical 不要承担以下职责：

- 不要把正式公告类事件当成“技术结论”
- 不要替代 `news` 或 `event` 的事实判断

## 6.3 value_analyst：从“估值是否便宜”升级为“定价是否合理”

重点增加以下内容：

1. 行业相对估值  
   - 结合 `industry_knowledge.py`
   - 不同行业使用不同估值容忍度

2. 历史分位  
   - PE / PB / EV/EBITDA / 股息率等的历史位置（能拿到就做）
   - 优先复用 `src/utils/tushare_client.py` 现有能力：
     - `compute_pe_percentile`
     - `compute_ev_ebitda`

3. 预期差 / 定价隐含要求  
   - 当前估值是否已经反映高增长
   - 是否属于低估值但基本面恶化，或者高估值但景气支持

4. 安全边际  
   - 对中长线打分尤其重要

value 不要承担以下职责：

- 不要重复 fundamental 的财务质量结论
- 不要把新闻情绪当估值结论
- 不要在缺乏核心数据时硬给强判断

## 6.4 news_analyst：重新定位为“新闻/舆情/叙事分析”，不再兼任事件事实层

这是本次改造的关键之一。

news_analyst 应只负责：

1. 媒体新闻情绪
2. 行业/政策舆情
3. 题材热度与叙事强度
4. 多家媒体是否在放大同一逻辑
5. 市场关注点是否集中、是否形成一致预期

news_analyst 不再负责：

- “公司正式公告发生了什么”  
- “监管事件是否成立”  
- “重大事项的事实判断”

这些交给 `event_analyst`。

换言之：

- `news` = 市场如何看、如何传、如何发酵
- `event` = 真实发生了什么、何时发生、可能影响哪个期限

## 7. 新增 agent 设计

## 7.1 event_analyst（新增，优先级最高）

目标：把当前 `news` 中混杂的“事件事实层”拆出来。

职责：

1. 识别重大事件 / 催化剂  
   - 业绩预告、业绩快报、回购、增减持、重大合同、并购重组、诉讼仲裁、处罚/问询/立案、股权质押、异常停复牌等
   - 如果当前 MCP/Tushare 没有完整公告接口，可先使用现有新闻 / 事件数据做“代理实现”，但必须增加 `source_level` 与 `confidence`，不能假装是正式公告

2. 标记事件时效  
   - 事件日期
   - 新近程度（freshness）
   - 影响短/中/长哪一个期限

3. 输出明确的事件方向  
   - 利多 / 利空 / 中性
   - 一次性 / 持续性
   - 是否已经被市场交易过

建议默认模型：
- 先映射到 Model 3（与 news/short-term 类似）
- 不新增新的模型槽位

## 7.2 quality_risk_analyst（新增，优先级第二）

目标：专门做“踩雷排查”，补足当前 fundamental 的薄弱点。

职责：

1. 财务质量  
   - 利润现金含量
   - 应收/存货/商誉/减值风险（有数据则做）
   - 非经常性损益依赖（有数据则做）

2. 治理与股东风险  
   - 质押、冻结、控制权变化、关联交易、股东减持等（有数据则做）
   - 不能确认时也要显式写“未获取到”

3. 风险标签输出  
   - 适合直接给 scorer / report 使用
   - 例如：
     - `cashflow_mismatch`
     - `high_pledge_risk`
     - `regulatory_risk`
     - `audit_risk`
     - `impairment_risk`
     - `earnings_quality_concern`

建议默认模型：
- 先映射到 Model 4（与 fundamental / value 同系）
- 如实现困难，可短期先复用 fundamental 模块的部分工具，但输出字段必须独立

## 7.3 moneyflow_analyst（新增，建议做；如果数据不足可先降级）

目标：补齐短线和波段最缺的“资金确认”维度。

职责（按数据可得性从高到低实现）：

1. 量价确认  
   - 换手率、成交额、持续性
   - 放量突破 / 缩量整理 / 异常放量回落

2. 资金代理  
   - 融资融券变化（如果当前数据有）
   - 龙虎榜 / 大宗交易 / 主力资金类代理（如果当前数据有）

3. 执行风险  
   - 流动性是否支持短线操作
   - 是否容易受单日情绪主导

降级策略：

- 如果当前 MCP / Tushare 数据不足以支撑独立 moneyflow agent：
  - 先在 `technical_analyst` 中扩展相关逻辑
  - 但仍建议生成一个 `moneyflow_signal_pack`，内容可标记为 `proxy_from_technical`
  - 这样后续数据一旦补齐，可以平滑切换成独立 agent

建议默认模型：
- Model 3

## 8. 中间产物升级：从纯文本升级为“文本 + 结构化证据”

## 8.1 总原则

当前项目中，各 agent 主要写入字符串，例如：

- `state.data.fundamental_analysis`
- `state.data.technical_analysis`
- `state.data.value_analysis`
- `state.data.news_analysis`

本次改造要求：

- 保留以上旧字段
- 同时为每个 agent 增加一个结构化字段（建议命名统一）

建议新增字段：

- `fundamental_signal_pack`
- `technical_signal_pack`
- `value_signal_pack`
- `news_signal_pack`
- `event_signal_pack`
- `quality_risk_signal_pack`
- `moneyflow_signal_pack`
- `analysis_package`
- `risk_gate_result`

## 8.2 建议的 signal_pack 结构

每个 agent 的结构化产物建议至少包含：

    {
      "agent_name": "fundamental",
      "analysis_text": "给人看的简洁总结",
      "bias": "bullish | neutral | bearish",
      "confidence": 0.0,
      "data_quality_score": 0.0,
      "key_points": ["..."],
      "signals": [
        {
          "factor": "经营现金流/净利润匹配",
          "direction": -1,
          "strength": 72,
          "confidence": 0.81,
          "time_horizon": ["medium", "long"],
          "source_level": "official_like | structured | news | derived | proxy",
          "freshness": "intraday | daily | weekly | quarterly | unknown",
          "risk_flags": ["cashflow_mismatch"],
          "note": "一句话说明"
        }
      ],
      "risk_flags": ["cashflow_mismatch"],
      "missing_data": ["未获取到审计意见", "未获取到质押数据"],
      "source_summary": "简述使用了哪些数据来源",
      "as_of_date": "YYYY-MM-DD"
    }

说明：

- `analysis_text` 继续服务旧逻辑和最终总结
- `signals` 是给 scorer / summarizer 真正消费的结构化证据
- `missing_data` 必须保留，不能装作所有数据都齐全
- `confidence` 与 `data_quality_score` 是不同概念：
  - `confidence` = 当前结论确信度
  - `data_quality_score` = 数据是否足够完整、可靠

## 8.3 新增纯 Python 工具：analysis package builder

建议新增工具模块，例如：

- `src/utils/analysis_schema.py`
- `src/utils/analysis_package_builder.py`

职责：

1. 将多个 `*_signal_pack` 合并成统一 `analysis_package`
2. 对重复因子做去重与归并
3. 给不同 source 打优先级
4. 汇总全局 `risk_flags`
5. 生成给 scorer / summarizer 的紧凑上下文

建议 `analysis_package` 至少包含：

    {
      "as_of_date": "YYYY-MM-DD",
      "executed_agents": ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"],
      "available_agents": [...],
      "missing_agents": [...],
      "global_risk_flags": [...],
      "global_missing_data": [...],
      "bullish_signals": [...],
      "bearish_signals": [...],
      "conflicting_signals": [...],
      "source_priority_summary": {...},
      "compact_prompt_context": "供 scorer / summary 直接使用的压缩文本"
    }

## 8.4 兼容策略

如果某个 agent 暂时没有产出 `*_signal_pack`：

- 不要直接报错终止
- 用旧的 `*_analysis` 文本生成一个最简 fallback pack
- fallback pack 要显式标记：
  - `source_level = derived`
  - `data_quality_score` 较低
  - `missing_data` 包含“结构化产物缺失”

## 9. 证据优先级（非常重要）

必须明确实现以下原则：

1. `event_analyst` / `quality_risk_analyst` 中来自结构化财务/事件/交易数据的证据，优先级高于纯新闻舆情
2. `news_analyst` 的结论可以增强或削弱市场预期，但不能推翻高优先级事实层证据
3. 当 `news` 与 `event` 冲突时：
   - 默认优先信任 `event`
   - 报告中要显式写出冲突
4. 当数据不足以支持强判断时：
   - 输出“观察/中性/待确认”
   - 不要强行给出高置信度方向

建议定义统一 source level：

- `official_like`：正式事件/结构化财务/可信结构化市场数据
- `structured`：可验证的数值类工具输出
- `news`：媒体与资讯
- `derived`：基于已有信息的推断
- `proxy`：临时代理实现（例如 moneyflow 暂由 technical 代理）

## 10. Scorer 改造要求

## 10.1 Scorer 仍保留 3 个，不新建新的终端打分 agent

继续保留：

- `short_term_scorer`
- `medium_term_scorer`
- `long_term_scorer`

不要再新增第四个“总评分 agent”。当前结构已经足够。

## 10.2 依赖关系调整

建议改为：

- `short_term_scorer`
  - 主要读取：`technical + news + event + moneyflow`
  - 如果 `moneyflow` 缺失，可继续运行，但降低 `data_quality_score`

- `medium_term_scorer`
  - 主要读取：`fundamental + value + quality_risk + event + technical + news`
  - `moneyflow` 可作为附加确认，不必成为强依赖

- `long_term_scorer`
  - 主要读取：`fundamental + value + quality_risk + event + news`
  - `technical / moneyflow` 只作辅助，不要过度放大短期价格行为

实现层面：

- 如果 LangGraph 边依赖改动不方便，可让所有 scorer 等待所有 agent 完成
- 但 scorer 内部的权重要体现不同期限的主次差异

## 10.3 Scorer 输入方式升级

当前 scorer 主要读取字符串分析结果。改造后要求：

1. 优先读取 `analysis_package`
2. 其次读取各 `*_signal_pack`
3. 最后才回退到原始 `*_analysis` 文本

Scorer 不要直接把所有 raw text 全量拼给 LLM。  
应先用 Python 做一次压缩整理，再交给 LLM 生成分数与解释。

## 10.4 建议的评分权重方向（不要求绝对照抄，可做近似实现）

### short_term（1-5 天）

建议重点：

- 技术状态：25
- 量价 / 流动性：20
- 资金确认：20
- 事件催化：20
- 新闻叙事 / 情绪：15
- 风险扣分：单独后处理

### medium_term（1-3 个月）

建议重点：

- 基本面质量：20
- 估值：15
- 财务质量 / 治理风险：20
- 事件持续性：15
- 技术与量价确认：10
- 行业/估值适配（沿用现有 `industry_knowledge.py`）：10
- 新闻叙事：10
- 风险扣分：单独后处理

### long_term（1-3 年）

建议重点：

- 基本面与资本回报：25
- 财务质量 / 治理风险：20
- 估值安全边际：15
- 行业地位 / 商业质量 / moat：15
- 资本配置 / 股东回报：10
- 事件与政策风险：10
- 技术确认：5
- 风险扣分：单独后处理

## 10.5 继续保留现有输出 shape，但允许追加字段

保持当前核心字段：

    {
      "score": int,
      "sub_scores": {...},
      "rating": str,
      "reasoning": str,
      "risk_warning": str,
      "suggested_action": str,
      "time_horizon": str
    }

允许新增可选字段：

    {
      "confidence": float,
      "key_drivers": [...],
      "risk_flags": [...],
      "abstain": bool,
      "abstain_reason": str,
      "data_quality_score": float,
      "evidence_summary": {...}
    }

要求：

- 旧代码若不读取这些字段，不应报错
- 前端若暂时不展示，也不应受影响

## 11. 风险门控（risk gate）要求

## 11.1 本次只做“分析层面的风险门控”，不做完整适当性系统

本次不要扩展成“客户风险偏好/适当性画像系统”。  
只做一个轻量的分析后处理模块，例如：

- `src/utils/risk_gate.py`

## 11.2 risk gate 的职责

1. 根据 `global_risk_flags` 和 `data_quality_score` 做后处理
2. 对明显不应高分的情形做“分数封顶 / 建议降级 / 改为观察”
3. 在报告中显式输出“为什么不能给更高建议”

## 11.3 建议的门控规则（可简化实现）

建议至少实现以下规则：

1. 如果存在高严重度风险标签，例如：
   - `audit_risk`
   - `regulatory_risk`
   - `high_pledge_risk`
   - `cashflow_mismatch`
   - `major_event_negative`
   
   则：
   - `medium_term_score`、`long_term_score` 不能直接给很高评级
   - 建议设置分数上限或建议动作降级到“观察/谨慎”

2. 如果只有新闻叙事强，但没有事件/基本面/质量支持：
   - `long_term_scorer` 不能给强推荐
   - `medium_term_scorer` 也应保守

3. 如果数据缺失较多，例如：
   - 核心 agent 缺失 >= 2 个
   - `data_quality_score` 很低
   
   则：
   - `abstain = true` 或至少建议“观察”
   - 报告必须说明“由于数据不足，不做强判断”

4. 如果短线流动性差、波动异常、量价不确认：
   - `short_term_scorer` 不应高分

## 12. Summary / Report Agent 改造要求

## 12.1 继续保留当前总结 agent，不强制改名

保持现有 summarizer / report agent 的总体角色，但输入改成：

- 优先：`analysis_package`
- 辅助：各 `*_signal_pack`
- 兼容：各 `*_analysis`

## 12.2 报告结构必须升级

建议输出以下结构（Markdown）：

1. `## 核心结论`
2. `## 多维信号总览`
3. `## 关键利多因素`
4. `## 关键利空与反证`
5. `## 事件与催化剂时间线`
6. `## 短线 / 中线 / 长线判断`
7. `## 主要风险与需要继续核验的数据`
8. `## 结论的置信度与适用边界`
9. `## 风险提示`

## 12.3 报告写作原则

1. 明确区分：
   - 事实
   - 推断
   - 建议

2. 当 `news` 与 `event` 冲突时：
   - 必须写出冲突
   - 不要只保留一种声音

3. 必须有“反证”部分  
   不要只写看多理由

4. 当数据不足时：
   - 必须承认不确定性
   - 不要过度自信

5. 报告要尽量使用简洁专业语言，不要空泛套话

## 13. 代码层面的建议改动清单

## 13.1 `src/utils/state_definition.py`

保留当前 `AgentState` 三大键不变：

- `messages`
- `data`
- `metadata`

但扩展 `data` 中允许出现的新字段（至少）：

- `fundamental_signal_pack`
- `technical_signal_pack`
- `value_signal_pack`
- `news_signal_pack`
- `event_signal_pack`
- `quality_risk_signal_pack`
- `moneyflow_signal_pack`
- `analysis_package`
- `risk_gate_result`

扩展 `metadata` 中建议新增：

- `analysis_version`（例如 `a_share_v2`）
- `executed_agents`
- `missing_agents`
- `data_quality_summary`
- `warnings`

## 13.2 新增 schema / builder / gate 工具模块

建议新增以下工具文件（文件名可微调，但职责要保留）：

- `src/utils/analysis_schema.py`
- `src/utils/analysis_package_builder.py`
- `src/utils/risk_gate.py`

职责：

- 定义 TypedDict / dataclass / schema（优先轻量实现，不要引入重型依赖）
- 合并多个 signal pack
- 做 source priority、risk flag 归并
- 输出 scorer / summarizer 用的 compact context
- 执行 score cap / abstain / recommendation downgrade

## 13.3 `src/utils/model_config.py`

在不新增模型槽位的前提下，增加新 agent 的模型映射：

- `event_analyst` -> Model 3
- `quality_risk_analyst` -> Model 4
- `moneyflow_analyst` -> Model 3

已有映射尽量保持不变：

- `fundamental_agent` -> Model 4
- `value_agent` -> Model 4
- `technical_agent` -> Model 3
- `news_agent` -> Model 3
- `summary_agent` / `medium/long scorer` -> Model 1
- `short_term_scorer` -> Model 3

## 13.4 A 股 agents 文件

请在现有 A 股 agent 所在目录中：

1. 修改现有 4 个 agent
2. 新增 2~3 个 agent 文件：
   - `event_analyst_agent.py`
   - `quality_risk_analyst_agent.py`
   - `moneyflow_analyst_agent.py`（可选但推荐）

如果当前实际文件命名不同，请按概念对应，不必强行对齐本文文件名。

要求：

- 每个 agent 返回：
  - 旧文本字段
  - 新结构化字段
- 每个 agent 输出要简洁：
  - `key_points` 建议不超过 6 条
  - `signals` 建议不超过 8 条
  - `analysis_text` 尽量控制在较短长度，避免后续 prompt 过长

## 13.5 `src/main.py`

改造单票分析图：

- 在现有 4 个并行 agent 基础上，加入 `event`、`quality_risk`、`moneyflow`
- `summarizer` 最终读取合并后的 `analysis_package`

注意：

- 不要影响原命令用法
- 如有必要可给单票分析增加一个轻量“构建 analysis_package”的纯 Python 步骤，但优先放在 summarizer 内部调用

## 13.6 `src/stock_pool/scoring_engine.py`

改造股票池评分图：

- 在现有 4 个 agent 基础上，加入 `event`、`quality_risk`、`moneyflow`
- 调整 scorer 依赖关系（见上文）
- 保持现有 `short / medium / long` 三池逻辑和 CLI/API 逻辑不变

注意：

- 为了兼容，允许所有 scorer 等所有 agent 完成后再运行
- 真正的期限区分通过 scorer 内部权重体现

## 13.7 `src/stock_pool/scoring_nodes.py` 及相关 scorer 模块

这是本次改造的核心落点之一。

要求：

1. 不再只读取 4 段纯文本
2. 优先调用 `build_analysis_package()`
3. 基于 `analysis_package` 组织 scorer prompt
4. scorer 出结果后，调用 `apply_risk_gate()`
5. 最终写回：
   - `short_term_score`
   - `medium_term_score`
   - `long_term_score`

如果当前 scorer 是纯 LLM 评分，不要求这次彻底改成纯规则引擎；  
但至少要做成“Python 预处理 + LLM 评分 + Python 风险后处理”的混合模式。

## 13.8 `src/stock_pool/stock_pool_manager.py`

可选增强，注意兼容：

- 如果风险较低，可在 stock entry 中增加可选字段：
  - `confidence`
  - `risk_flags`
  - `key_drivers`
  - `abstain`
  - `analysis_version`

要求：

- 老的 `stock_pool.json` 必须仍能被读取
- 缺失这些字段时用默认值回填
- 如果感觉改动风险偏大，可以先只在详细 report / task result 中暴露，不急着落盘

## 13.9 API 层（只做必要适配）

仅在必要时对以下接口返回结构做“追加字段”式增强：

- `/api/report`
- `/api/score/{term}/{stock_code}`
- `/api/score-all/{stock_code}`

原则：

- 不删除旧字段
- 不破坏现有前端
- 新字段前端可忽略

## 13.10 数据访问层（尽量少改）

优先复用现有能力：

- `src/utils/tushare_client.py`
- 现有 MCP server 工具
- `industry_knowledge.py`
- `fetch_utils.py`

如果确实缺少关键数据，只允许做“窄接口增强”：

- 例如新增某个财务质量快照函数
- 或某个事件/公告代理抓取函数
- 或某个量价/资金代理函数

不要为了本次任务大幅重构 MCP server。

## 14. Prompt / 输出格式要求

## 14.1 Agent prompt 要求

每个分析 agent 的 prompt 应明确要求输出以下信息：

1. 简洁结论
2. 关键依据
3. 风险点
4. 缺失数据
5. 结构化 signals

如果模型不稳定，允许采用：

- “先让模型输出 JSON”
- 若 JSON 解析失败，则回退到“从文本中提取最小结构化结果”

不要假设所有 OpenAI-compatible 模型都能 100% 严格遵守 JSON schema。

## 14.2 Scorer prompt 要求

Scorer prompt 应明确：

1. 不要重复照抄上游分析
2. 必须基于不同期限的目标做取舍
3. 必须考虑反证与风险
4. 必须输出当前保留的标准 shape
5. 在数据不足或高风险时，允许给出“观察 / 谨慎 / abstain”

## 14.3 Summary prompt 要求

Summary prompt 应明确：

1. 要基于 `analysis_package`
2. 要有“核心结论 + 反证 + 风险 + 数据缺口”
3. 要区分短/中/长逻辑
4. 要承认不确定性
5. 不要过度营销化表达

## 15. 最低可接受实现（如果时间有限，先做到这里）

如果一次性改太多有风险，最低可接受版本是：

1. 保留原 4 agent，但都扩展为“文本 + signal_pack”
2. 新增 `event_analyst`
3. 新增 `quality_risk_analyst`
4. `moneyflow` 先由 `technical` 代理
5. 加 `analysis_package_builder`
6. 改造 3 个 scorer 读取结构化数据
7. 加 `risk_gate`
8. 升级总结报告结构

这已经足以把系统从“4 段分析文本拼接打分”升级为“更专业、更稳健的多维证据系统”。

## 16. 测试与验收要求

## 16.1 必做测试

至少补充以下测试：

1. `analysis_package_builder` 合并测试
2. `risk_gate` 规则测试
3. scorer 在新旧字段并存时的兼容测试
4. 缺失 `event` / `moneyflow` 时的降级测试
5. `news` 与 `event` 冲突时的处理测试
6. stock pool 旧数据文件兼容测试

## 16.2 建议 smoke test

至少跑通以下路径：

1. `python -m src.main --command "分析嘉友国际"`
2. `python -m src.main_pool score 603871`
3. `python -m src.main_pool report 603871`
4. 现有 pytest 测试不应大面积回归失败
5. `/api/report` 与 `/api/score-all` 至少能完成一次真实任务

## 16.3 验收标准

满足以下条件可视为改造完成：

1. 现有 CLI / API 仍可正常运行
2. A 股单票报告能显式展示：
   - 核心结论
   - 关键利多
   - 关键利空/反证
   - 事件驱动
   - 风险标签
   - 数据缺口
3. 股票池 short / medium / long 评分明显不再只是基于 4 段自然语言总结
4. 当高风险标签出现时，系统会自动保守处理
5. 数据不足时，系统会承认不确定性，而不是强行给高置信度推荐

## 17. 最后的实施风格要求（给 Claude Code）

1. 优先做“增量式重构”，不要大面积推翻。
2. 优先保证主线可运行，再逐步增强。
3. 优先复用现有模型配置、现有工具、现有状态结构。
4. 遇到数据源缺口，优先做 graceful fallback，不要把整个任务卡死。
5. 对外接口尽量不变，对内逐步升级为结构化证据流。
6. 如果某一步无法完美实现，请先做兼容版本，并在代码注释或 TODO 中说明后续升级点。
7. 这次任务的中心不是“多加几个花哨 agent”，而是把 A 股主线改成：
   - 更清晰的职责边界
   - 更强的结构化中间产物
   - 更合理的风险约束
   - 更专业的最终报告和评分解释