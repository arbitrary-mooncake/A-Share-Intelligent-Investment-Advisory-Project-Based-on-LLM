# 🤖 A股智能投顾Agent助手

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/Protocol-MCP-green?style=flat-square)](https://modelcontextprotocol.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)
[![Tushare](https://img.shields.io/badge/Data-Tushare%20Pro-blue?style=flat-square)](https://tushare.pro/)

</div>

> 🧠 **专业级多智能体投研平台** — 基于 LangGraph 的 7 智能体并行分析 + 短/中/长三线评分 + 14 条模拟盘线路 + 23 条回测线路 + 消融实验贡献度评估系统。覆盖 A 股个股、基金/ETF 两大资产类别。

---

## ✨ 核心亮点

- **🕸️ 多智能体协作** — 7 个分析 Agent 并行运行（基本面 / 技术面 / 估值 / 新闻 / 事件 / 质量风控 / 资金流向），分工明确，交叉验证。
- **📊 结构化证据体系** — 每个 Agent 输出 SignalPack（bias / confidence / signals / risk_flags / missing_data），支持来源优先级排序与冲突检测。
- **🛡️ 反幻觉设计** — 数据事实区 / 分析判断区分离，`[数据]` `[判断]` 标签，分级缺失数据语言，杜绝 LLM 编造数据。
- **📈 三维度评分** — 短线 / 中线 / 长线独立评分 + 风险门控（4 道硬规则），输出可执行的买入/持有/回避建议。
- **🧪 评估控制塔** — 14 条模拟盘线路 + 23 条回测线路，消融实验量化每个 Agent 的真实贡献（Bootstrap CI + Permutation Test）。
- **🔍 四层精筛股票池** — 全市场 ~5000 只股票 → 硬筛 → 批量打分 → 快筛 → 正式评分，冷启动可寻。
- **💰 A股 + 基金双管线** — 基金分析包含 7 并行 Agent + 10 道基金风险门控。
- **⚡ 双缓存架构** — 生产 / 评估缓存物理隔离，每 Agent 独立 TTL，热缓存命中时全链路跳过 LLM。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   Web UI (Streamlit)                 │
├─────────────────────────────────────────────────────┤
│  A股分析管线                 基金分析管线              │
│  ┌──────────────────┐    ┌──────────────────┐       │
│  │ 7 并行分析 Agent  │    │ 7 并行分析 Agent  │       │
│  │ · 基本面          │    │ · 产品文档        │       │
│  │ · 技术面          │    │ · 业绩风险        │       │
│  │ · 估值            │    │ · 持仓分析        │       │
│  │ · 新闻舆情        │    │ · 基金经理        │       │
│  │ · 事件催化剂      │    │ · 基准比较        │       │
│  │ · 质量风控        │    │ · 费率评估        │       │
│  │ · 资金流向        │    │ · 基金事件        │       │
│  └──────┬───────────┘    └──────┬───────────┘       │
│         ↓ 信号包合并             ↓ 合并节点          │
│  ┌──────────────────┐    ┌──────────────────┐       │
│  │ 短/中/长三维评分   │    │ 基金评分 + 报告    │       │
│  │ + 风险门控        │    │ + 10道风险门控     │       │
│  └──────┬───────────┘    └──────────────────┘       │
│         ↓                                            │
│  Markdown + PDF 分析报告                              │
├─────────────────────────────────────────────────────┤
│              评估控制塔 (Eval System)                  │
│  · 14 条模拟盘线路 · 23 条回测线路                     │
│  · 多维 Loss · 消融贡献度 · Bootstrap 统计检验         │
│  · 长期趋势存储 · Streamlit 趋势面板                   │
├─────────────────────────────────────────────────────┤
│                 MCP 数据服务层                         │
│  · Tushare MCP Server · AKShare MCP Server           │
│  · Web Search MCP Server · YFinance MCP Server       │
└─────────────────────────────────────────────────────┘
```

---

## 📦 安装

> ⚠️ 本项目复杂度较高，需要多步骤环境配置。未来将封装为可安装包简化流程。

### 前置要求

| 依赖 | 说明 |
|------|------|
| Python 3.10+ | 推荐 3.11 |
| Tushare Pro Token | [注册获取](https://tushare.pro/)（需积分 ≥2000） |
| LLM API Key | MiMo / DeepSeek / Qwen 等（见模型配置） |
| Git | 克隆仓库 |

### 安装步骤

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd A股智能投顾Agent助手

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Linux/WSL
# 或 venv\Scripts\activate  # Windows CMD

# 3. 安装依赖
cd Finance
pip install -r requirements.txt

# 4. 配置环境变量
cd Financial-MCP-Agent
cp .env.example .env
# 编辑 .env，填入 API Key 和模型配置
```

### 启动 Web UI

```bash
# Linux / WSL
./run.sh start

# Windows CMD
run.bat start

# Windows PowerShell
.\run.ps1 start
```

启动后在浏览器打开 `http://localhost:8501`。

---

## ⚙️ 配置说明

### 多模型架构

本系统使用 **6 个独立的 LLM 模型**，按任务复杂度分配合适的模型以平衡成本与质量：

| 代号 | 默认模型 | 适用角色 |
|------|---------|---------|
| **M1** | MiMo-V2.5-Pro | 总结、中/长线评分、基本面、估值、质量风控、基金评分/报告 |
| **M2** | Qwen3.6-Flash | 快速查询、批量初筛 |
| **M3** | Qwen3.7-Plus | 技术面、新闻、短线评分、事件、资金流向、基金经理 |
| **M5** | MiMo-V2.5 | 评估系统 Agent 分析（成本优化） |
| **M6** | DeepSeek V4 Pro | 评估编排、归因诊断、报告撰写、LLM Free 策略 |

> 📌 **未来计划**：模型将统一为 MiMo / DeepSeek / Qwen 三家，简化配置。

### 环境变量

`.env` 文件中按以下格式配置：

```bash
# 模型 M1（基础模型）
OPENAI_COMPATIBLE_API_KEY=sk-xxxx
OPENAI_COMPATIBLE_BASE_URL=https://api.mimo.com/v1
OPENAI_COMPATIBLE_MODEL=mimo-v2.5-pro

# 模型 M2（快速模型）
OPENAI_COMPATIBLE_API_KEY_2=sk-xxxx
OPENAI_COMPATIBLE_BASE_URL_2=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_COMPATIBLE_MODEL_2=qwen3.6-flash

# 模型 M3
OPENAI_COMPATIBLE_API_KEY_3=sk-xxxx
OPENAI_COMPATIBLE_BASE_URL_3=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_COMPATIBLE_MODEL_3=qwen3.7-plus

# 模型 M5（评估）
OPENAI_COMPATIBLE_API_KEY_5=sk-xxxx
OPENAI_COMPATIBLE_BASE_URL_5=https://api.mimo.com/v1
OPENAI_COMPATIBLE_MODEL_5=mimo-v2.5

# 模型 M6（编排）
OPENAI_COMPATIBLE_API_KEY_6=sk-xxxx
OPENAI_COMPATIBLE_BASE_URL_6=https://api.deepseek.com
OPENAI_COMPATIBLE_MODEL_6=deepseek-v4-pro

# Tushare
TUSHARE_TOKEN=your_tushare_token_here
```

---

## 💻 CLI 命令速查

所有命令从 `Finance/Financial-MCP-Agent/` 目录运行。

### 个股分析

```bash
# 单股完整分析（生成 Markdown + PDF 报告）
python -m src.main --command "分析嘉友国际"

# 添加股票到精筛池
python -m src.main_pool add 603871 嘉友国际

# 三维度评分
python -m src.main_pool score 603871

# 查看评分详情
python -m src.main_pool report 603871
```

### 基金分析

```bash
# 基金完整分析
python -m src.fund_main --command "分析华夏上证50ETF"
```

### 评估系统

```bash
python -m src.eval check              # 日常检查（调仓 + 结算）
python -m src.eval status             # 查看所有线路状态
python -m src.eval pool status --term short  # 精筛池状态
python -m src.eval pool update --term medium --mode full  # 全量池更新（四层管线）
python -m src.eval backtest --term medium --start 2024-01-01 --end 2025-12-31  # 历史回测
python -m src.eval report --latest    # 最新评估报告
python -m src.eval trends --metric score --term medium --days 90  # 趋势图
python -m src.eval agent-contribution --term medium --source backtest  # Agent 贡献度
python -m src.eval optimize --analyze  # 优化建议
```

---

## 💬 使用场景

### 1. 单股深度分析
> **用户**："分析嘉友国际"
>
> **系统**：启动 7 个并行 Agent，分别分析基本面（盈利质量/现金流/成长性）、技术面（趋势/量价/指标形态）、估值（行业相对PE/PB/历史分位）、新闻舆情、事件催化剂、质量风控（商誉/担保/质押）、资金流向（融资融券/大宗交易）。→ 信号包合并 → 短/中/长三维评分 → 风险门控 → 9 段式 Markdown + PDF 报告。

### 2. 三维度评分
> **用户**："给 603871 打个分"
>
> **系统**：短线评分（4 代理权重：技术25/量价20/资金20/事件20/舆情15）→ 中线评分（7 代理全量）→ 长线评分（7 代理全量，含护城河/政策因子）→ 三道风险门 → 输出：短线68分（谨慎买入）、中线72分（买入）、长线80分（强烈推荐）。

### 3. 基金深度分析
> **用户**："分析华商优势行业混合"
>
> **系统**：7 个并行基金 Agent → 合并节点（signal_pack 优先 + regex 回退 + 冲突检测）→ 6 维基金评分 + 10 道基金风险门控 → 基金分析报告。

### 4. 全市场精筛
> **用户**：后台定时触发四层精筛管道
>
> **系统**：Layer 0 硬筛（去 ST/新股/僵尸股，~5000→~4500）→ Layer 1 批量打分（M1/M3，~900批×5只，输出短/中/长三维分数）→ Layer 2 快筛（M2 轻量复核，~2000只）→ Layer 3 正式评分（Top~100只，全 7 Agent 管线逐只分析）。

### 5. 消融实验
> **用户**："看看技术面 Agent 对中线评分到底有没有贡献"
>
> **系统**：跑中线 8 条回测线路（MB-L0 全 Agent 基线 → MB-L1 去基本面 → ... → MB-L7 去资金流向）→ 计算各线路 Loss → ΔLoss 归因 → Bootstrap CI 显著性 → 输出："技术面 Agent 在中线维度贡献 12.3% (±2.1%, p<0.01)，**显著正向**。"

### 6. LLM Free 策略验证
> **用户**："不用 Agent 打分，让 LLM 直接看数据做决策，效果差多少？"
>
> **系统**：LLM Free 线路（M6/DeepSeek V4 Pro）仅使用原始 Tushare 行情数据做自主决策 → 与全 Agent 评分线路对比 → 量化 Agent 体系相对于裸 LLM 的超额收益。

---

## 🧠 Agent 体系详解

### A股 7 大分析 Agent

| Agent | 模型 | Thinking | 核心能力 |
|-------|------|----------|---------|
| **基本面** | M1 (MiMo) | ✅ | 盈利质量、现金流、资产负债表健康度、成长性 |
| **技术面** | M3 (Qwen) | ❌ | 价格趋势、量价关系、MACD/RSI/MA 指标形态 |
| **估值** | M1 (MiMo) | ✅ | 行业相对 PE/PB、历史分位数、安全边际 |
| **新闻舆情** | M3 (Qwen) | ✅ | 媒体情绪、叙事强度（不分析事实性事件） |
| **事件催化剂** | M3 (Qwen) | ✅ | 财报发布、回购、并购、处罚、质押、增减持 |
| **质量风控** | M1 (MiMo) | ✅ | 现金流质量、商誉/减值、担保、治理风险 |
| **资金流向** | M3 (Qwen) | ✅ | 融资融券、大宗交易、龙虎榜、量价确认 |

### 信号包结构 (SignalPack)

每个 Agent 输出的结构化证据：

```json
{
  "bias": "bullish",
  "confidence": 0.78,
  "signals": [
    {"factor": "profit_quality", "direction": 1, "strength": 3, "source_level": "structured"}
  ],
  "risk_flags": [
    {"flag": "high_goodwill", "severity": "medium", "detail": "商誉占净资产 35%"}
  ],
  "missing_data": ["operating_cash_flow_q3"],
  "data_quality_score": 0.85
}
```

### 风险门控规则

| # | 规则 | 效果 |
|---|------|------|
| 1 | 关键风险标记触发 | 分数上限 + 操作降级（审计风险→60, 退市风险→50） |
| 2 | 纯新闻叙事无事实支撑 | 中/长线分数上限 55 |
| 3 | ≥2 Agent 数据缺失 + 质量 <0.4 | 弃权 |
| 4 | 短线流动性风险 | 分数上限 50 |

---

## 🧪 评估控制塔

评估系统是环绕主投顾系统的独立评估层，不参与生产推荐，仅用于系统迭代验证。

### 线路体系

| 类型 | 数量 | 说明 |
|------|------|------|
| 短线模拟盘 | 10 条 | S-L0（全 Agent 基线）→ S-L7（消融线），S-L8（持仓策略），S-L9（LLM Free） |
| 中线模拟盘 | 2 条 | M-L0（全 Agent），M-L1（LLM Free） |
| 长线模拟盘 | 2 条 | L-L0（全 Agent），L-L1（LLM Free） |
| 短线回测 | 7 条 | SB-L0~SB-L6（含参考线 SB-L6） |
| 中线回测 | 8 条 | MB-L0~MB-L7（含退化一致性检查线） |
| 长线回测 | 8 条 | LB-L0~LB-L7（同上） |

### 多维 Loss

```
L_total = w_effect × L_effect + w_stability × L_stability + w_efficiency × L_efficiency
```

- **L_effect**（效果 Loss）：超额收益、胜率、盈亏比
- **L_stability**（稳定性 Loss）：最大回撤、波动率、连续亏损次数
- **L_efficiency**（效率 Loss）：换手率、仓位利用率

### 统计检验

- **Bootstrap CI**：Agent 贡献度的 95% 置信区间
- **Cluster Bootstrap**：短线的时序自相关修正
- **Permutation Test**：非参数显著性检验

---

## ⚡ 缓存系统

| | 生产缓存 | 评估缓存 |
|---|---|---|
| 目录 | `data/intermediate_cache/` | `data/eval/cache/` |
| 命名空间后缀 | （无） | `_eval` |
| 模型 | M1/M3 | M5 |
| 隔离性 | 与评估互不干扰 | 与生产互不干扰 |

**每 Agent 独立 TTL**：基本面 15 天 / 估值 7 天 / 质量风控 7 天 / 技术面 1 天 / 新闻 1 天 / 事件 1 天 / 资金流向 1 天。

**热缓存命中时**，全部 7 个 Agent 缓存新鲜 → 跳过整个 LLM 管线，直接从缓存重建结果。

---

## 📊 性能参考

冷启动全市场精筛（无缓存，首次运行）：
- Layer 0（硬筛）：~22min（~4500 次 Tushare `daily` API，限速 200 次/分）
- Layer 1（批量打分）：~2.5h
- Layer 2（快筛）：~1h
- Layer 3（正式评分）：~5.5h（~100 只 × ~200s/只）
- **冷启动总耗时：约 9-10 小时**（一个期限维度）

热缓存后续运行：Agent 缓存命中时跳过 LLM 管线，主要瓶颈在 scorer LLM。

---

## 🗂️ 项目结构

```
A股智能投顾Agent助手/
├── Finance/
│   ├── a-share-mcp-server/          # MCP 数据服务（Tushare / AKShare / WebSearch / YFinance）
│   └── Financial-MCP-Agent/         # 主应用
│       ├── src/
│       │   ├── agents/              # 7 个 A 股分析 Agent + 7 个基金 Agent + 评分 Agent
│       │   ├── stock_pool/          # 精筛股票池 & 四层筛选管线
│       │   ├── qa/                  # 问答引擎
│       │   ├── eval/                # 评估控制塔（模拟盘、回测、消融、趋势）
│       │   ├── tools/               # MCP 客户端 & 工具缓存
│       │   ├── utils/               # 缓存、模型配置、行业知识、风控门
│       │   ├── api/                 # FastAPI + Streamlit Web 服务
│       │   └── fund_pool/           # 基金池管理
│       ├── data/                    # 缓存 & 持久化数据
│       ├── reports/                 # 分析报告输出
│       ├── tests/                   # 单元测试
│       └── run.sh / run.bat / run.ps1  # 启动脚本
├── docs/                            # 文档
├── CLAUDE.md                        # 开发者指南
├── 评分智能体开发总纲.md                # 评估系统设计规范
└── README.md
```

---

## 🚧 路线图

- [ ] **安装包化** — 封装为 `pip install` 可安装包，简化部署流程
- [ ] **模型统一** — 精简为 MiMo / DeepSeek / Qwen 三家，降低配置复杂度
- [ ] **一键安装脚本** — 自动化环境检测 + 依赖安装 + 配置生成
- [ ] **回测加速** — Layer 0/3 并行化，缩短冷启动时间
- [ ] **桌面应用** — Electron 桌面版，降低非技术用户门槛
- [ ] **策略市场** — 可插拔交易策略注册框架

---

## 📊 数据来源

| 数据 | 来源 |
|------|------|
| A股行情 / 财务 / 估值 | [Tushare Pro](https://tushare.pro/) |
| 国际市场 / 宏观 / 商品补充 | [AKShare](https://akshare.akfamily.xyz/) + [Yahoo Finance](https://finance.yahoo.com/) |
| 联网搜索 | Web Search MCP Server |

---

## 🤝 贡献

本项目正在积极开发中。欢迎提交 Issue / Pull Request。

---

## 📝 许可证

MIT License

---

<div align="center">

⭐ **如果这个项目对你有帮助，请给一个 Star！**

</div>
