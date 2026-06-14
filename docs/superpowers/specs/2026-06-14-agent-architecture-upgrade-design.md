# A股主线架构升级设计文档

## 1. 概述

本设计文档是对 `子agent架构改进方案.md`（以下简称"大纲"）的实施方案确认。大纲本身即为完整设计规范，本文仅记录实施决策和关键技术选型。

**目标**：将A股主线从"4段分析文本拼接打分"升级为"7维结构化证据 + 风险门控 + 专业报告"系统。

**范围**：仅A股主线（单票分析 + 股票池评分），不涉及基金分析、智能问答、快筛/批量打分。

## 2. 实施决策汇总

| 决策项 | 选择 | 理由 |
|--------|------|------|
| moneyflow_analyst | 独立agent | Tushare MCP有moneyflow/margin等数据源，足以支撑 |
| JSON输出策略 | JSON解析 + 详细文本解析fallback | 兼容不同LLM的JSON能力，fallback确保永不崩溃 |
| 短期scorer依赖 | technical+news+event+moneyflow（4个） | 语义正确，短线不依赖基本面 |
| 中长期scorer依赖 | 全部7个agent | 中长线需要全面信息 |
| 报告结构 | 替换为大纲的9段式 | 更专业，覆盖反证/置信度/数据缺口 |
| fallback粒度 | 详细文本解析（正则/关键词） | 最大化提取结构化信息 |
| 实施节奏 | 一次性全部完成 | 各部分有依赖，分期增加集成风险 |
| 实施策略 | 自底向上 | 每步可验证，保证现有功能不被破坏 |

## 3. 架构变更

### 3.1 新增agent

| Agent | 模型 | Thinking | 输出字段 |
|-------|------|----------|---------|
| event_analyst | M3 (Qwen3.7-Plus) | enabled | `event_analysis` + `event_signal_pack` |
| quality_risk_analyst | M4 (Kimi K2.6) | enabled | `quality_risk_analysis` + `quality_risk_signal_pack` |
| moneyflow_analyst | M3 (Qwen3.7-Plus) | enabled | `moneyflow_analysis` + `moneyflow_signal_pack` |

### 3.2 图结构变更

**单票分析 (main.py)**：
```
start_node → [fundamental, technical, value, news, event, quality_risk, moneyflow] (7并行) → summarizer → END
```

**股票池评分 (scoring_engine.py)**：
```
start_node → [fundamental, technical, value, news, event, quality_risk, moneyflow] (7并行)
  short_term_scorer ← [technical, news, event, moneyflow]
  medium_term_scorer ← [all 7]
  long_term_scorer ← [all 7] → END
```

### 3.3 中间产物

每个agent输出两个字段：
- `*_analysis`：保留旧文本字段（向后兼容）
- `*_signal_pack`：新增结构化JSON（大纲8.2节格式）

新增纯Python工具：
- `src/utils/analysis_schema.py` — TypedDict定义
- `src/utils/analysis_package_builder.py` — 合并signal_pack为analysis_package
- `src/utils/risk_gate.py` — 风险门控后处理

### 3.4 Scorer改造

Scorer输入优先级：`analysis_package` > `*_signal_pack` > `*_analysis`文本

Scorer输出保留现有shape，新增可选字段：`confidence`, `key_drivers`, `risk_flags`, `abstain`, `abstain_reason`, `data_quality_score`

### 3.5 Summary agent改造

输入：优先读`analysis_package`，兼容读`*_analysis`
输出：大纲12.2节的9段式报告结构

## 4. 实施顺序（自底向上）

### Phase 1: 基础层
1. `src/utils/analysis_schema.py` — signal_pack和analysis_package的TypedDict定义
2. `src/utils/analysis_package_builder.py` — 合并、去重、source priority、compact context生成
3. `src/utils/risk_gate.py` — 风险门控规则
4. `src/utils/state_definition.py` — 扩展metadata字段
5. `src/utils/model_config.py` — 新增3个agent的模型映射

### Phase 2: Agent层
6. 改造 `fundamental_agent.py` — 增加signal_pack输出
7. 改造 `technical_agent.py` — 增加signal_pack输出，明确moneyflow边界
8. 改造 `value_agent.py` — 增加signal_pack输出
9. 改造 `news_agent.py` — 重新定位为舆情分析，移除事件事实层
10. 新增 `event_analyst_agent.py`
11. 新增 `quality_risk_analyst_agent.py`
12. 新增 `moneyflow_analyst_agent.py`

### Phase 3: 集成层
13. 改造 `scoring_nodes.py` — 读取analysis_package，调用risk_gate
14. 改造 `short_term_scorer.py` — 依赖调整 + 权重调整 + 读取结构化数据
15. 改造 `medium_term_scorer.py` — 读取结构化数据 + 权重调整
16. 改造 `long_term_scorer.py` — 读取结构化数据 + 权重调整
17. 改造 `summary_agent.py` — 9段式报告 + 读取analysis_package

### Phase 4: 入口层
18. 改造 `src/main.py` — 图结构扩展为7 agent + summarizer读取analysis_package
19. 改造 `src/stock_pool/scoring_engine.py` — 图结构扩展 + scorer依赖调整

### Phase 5: 测试层
20. `tests/test_analysis_schema.py` — schema定义和builder测试
21. `tests/test_risk_gate.py` — 风险门控规则测试
22. `tests/test_signal_pack_fallback.py` — JSON解析失败fallback测试
23. `tests/test_backward_compat.py` — 新旧字段并存兼容测试

## 5. 向后兼容保证

- 所有现有`*_analysis`文本字段保留不变
- `stock_pool.json`格式兼容（新字段为可选）
- 现有API响应不删除旧字段
- 现有CLI命令用法不变
- 不新增环境变量，不新增模型槽位
- 现有pytest测试不应大面积回归

## 6. 降级策略

- 若某agent未产出`*_signal_pack`：从`*_analysis`文本生成fallback pack（source_level=derived, data_quality_score较低）
- 若JSON解析失败：用详细文本解析提取bias/key_points/risk_flags等
- 若MCP/Tushare某数据不可得：标记`missing_data`，不阻塞流程
- moneyflow一期数据不足时：标记`proxy_from_technical`

## 7. 不在范围内

- 基金分析全链路
- 智能问答（QA）
- 快速查询接口
- 快筛/批量打分路径（但可复用新工具函数）
- MCP协议层改造
- Streamlit前端大改版
- 客户适当性/风险偏好系统
