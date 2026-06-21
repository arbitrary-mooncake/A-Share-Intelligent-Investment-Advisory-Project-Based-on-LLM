"""
评测系统核心数据结构 — 与 analysis_schema.py 互补, 不替代。
analysis_schema.py 定义分析中间产物(Signal/SignalPack/AnalysisPackage/DecisionPack),
本文件定义评测系统自身的持久化与运行态数据结构。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class EvalBatch:
    """一次完整的检查批次"""
    batch_id: str = ""                          # 唯一ID
    status: str = "queued"                      # queued/running/completed/failed/optimizing
    trigger_source: str = "ui"                  # ui/cli/api
    started_at: str = ""                        # ISO datetime
    finished_at: str = ""                       # ISO datetime
    market_session: str = "post_close"          # post_close/pre_open/intraday
    data_cutoff_time: str = ""                  # 数据截止时间
    stable_version: str = ""                    # 当前 stable commit/tag
    candidate_version: str = ""                 # 当前 candidate commit/tag
    run_profile: str = ""                       # 评测运行配置名
    summary_metrics_json: str = ""              # 批次总览指标JSON
    report_md_path: str = ""                    # 报告路径
    report_pdf_path: str = ""                   # PDF报告路径
    optimize_ready: bool = False                # 是否允许点击优化
    error_message: str = ""                     # 失败时错误信息

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "trigger_source": self.trigger_source,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "market_session": self.market_session,
            "data_cutoff_time": self.data_cutoff_time,
            "stable_version": self.stable_version,
            "candidate_version": self.candidate_version,
            "run_profile": self.run_profile,
            "summary_metrics_json": self.summary_metrics_json,
            "report_md_path": self.report_md_path,
            "report_pdf_path": self.report_pdf_path,
            "optimize_ready": self.optimize_ready,
            "error_message": self.error_message,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'EvalBatch':
        return EvalBatch(
            batch_id=str(data.get("batch_id", "")),
            status=str(data.get("status", "queued")),
            trigger_source=str(data.get("trigger_source", "ui")),
            started_at=str(data.get("started_at", "")),
            finished_at=str(data.get("finished_at", "")),
            market_session=str(data.get("market_session", "post_close")),
            data_cutoff_time=str(data.get("data_cutoff_time", "")),
            stable_version=str(data.get("stable_version", "")),
            candidate_version=str(data.get("candidate_version", "")),
            run_profile=str(data.get("run_profile", "")),
            summary_metrics_json=str(data.get("summary_metrics_json", "")),
            report_md_path=str(data.get("report_md_path", "")),
            report_pdf_path=str(data.get("report_pdf_path", "")),
            optimize_ready=bool(data.get("optimize_ready", False)),
            error_message=str(data.get("error_message", "")),
        )


@dataclass
class PredictionSnapshot:
    """某标的在某次检查中的结构化快照"""
    snapshot_id: str = ""
    batch_id: str = ""
    line_id: str = ""                       # S-L0, S-L1, ..., M-L0, L-L0, etc.
    asset_type: str = "stock"               # stock/fund
    symbol: str = ""                        # 股票代码
    name: str = ""                          # 公司名称
    term: str = ""                          # short/medium/long
    as_of_date: str = ""                    # YYYY-MM-DD
    pit_mode: str = "exact"                 # exact/best_effort/partial/unsupported
    eval_mode: str = "real"                 # real/backtest
    score: float = 0.0                      # 评分
    action: str = ""                        # buy/sell/hold
    signal_pack_bundle_json: str = ""       # 所有agent的signal_pack JSON
    analysis_package_json: str = ""         # AnalysisPackage JSON
    decision_pack_json: str = ""            # DecisionPack JSON
    model_profile: str = ""
    version_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        d["snapshot_id"] = self.snapshot_id
        d["batch_id"] = self.batch_id
        d["line_id"] = self.line_id
        d["asset_type"] = self.asset_type
        d["symbol"] = self.symbol
        d["name"] = self.name
        d["term"] = self.term
        d["as_of_date"] = self.as_of_date
        d["pit_mode"] = self.pit_mode
        d["eval_mode"] = self.eval_mode
        d["score"] = self.score
        d["action"] = self.action
        d["signal_pack_bundle_json"] = self.signal_pack_bundle_json
        d["analysis_package_json"] = self.analysis_package_json
        d["decision_pack_json"] = self.decision_pack_json
        d["model_profile"] = self.model_profile
        d["version_hash"] = self.version_hash
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'PredictionSnapshot':
        def _f(v, d=0.0):
            try: return float(v)
            except: return d
        return PredictionSnapshot(
            snapshot_id=str(data.get("snapshot_id", "")),
            batch_id=str(data.get("batch_id", "")),
            line_id=str(data.get("line_id", "")),
            asset_type=str(data.get("asset_type", "stock")),
            symbol=str(data.get("symbol", "")),
            name=str(data.get("name", "")),
            term=str(data.get("term", "")),
            as_of_date=str(data.get("as_of_date", "")),
            pit_mode=str(data.get("pit_mode", "exact")),
            eval_mode=str(data.get("eval_mode", "real")),
            score=_f(data.get("score")),
            action=str(data.get("action", "")),
            signal_pack_bundle_json=str(data.get("signal_pack_bundle_json", "")),
            analysis_package_json=str(data.get("analysis_package_json", "")),
            decision_pack_json=str(data.get("decision_pack_json", "")),
            model_profile=str(data.get("model_profile", "")),
            version_hash=str(data.get("version_hash", "")),
        )


@dataclass
class RealizedLabel:
    """未来真实市场结果"""
    snapshot_id: str = ""
    line_id: str = ""
    term: str = ""
    horizon_days: int = 1
    outcome_date: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    asset_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    excess_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility_pct: float = 0.0
    is_valid: bool = True
    settlement_notes: str = ""
    meta_json: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "line_id": self.line_id,
            "term": self.term,
            "horizon_days": self.horizon_days,
            "outcome_date": self.outcome_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "asset_return_pct": self.asset_return_pct,
            "benchmark_return_pct": self.benchmark_return_pct,
            "excess_return_pct": self.excess_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "volatility_pct": self.volatility_pct,
            "is_valid": self.is_valid,
            "settlement_notes": self.settlement_notes,
            "meta_json": self.meta_json,
        }


@dataclass
class ExperimentRun:
    """一次控制变量实验"""
    experiment_id: str = ""
    batch_id: str = ""
    experiment_type: str = ""       # ablation/gate_on_off/consistency/fidelity/stable_vs_candidate
    variant_key: str = ""           # 实验变体标识
    asset_type: str = "stock"
    symbol: str = ""
    term: str = ""
    metrics_json: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "batch_id": self.batch_id,
            "experiment_type": self.experiment_type,
            "variant_key": self.variant_key,
            "asset_type": self.asset_type,
            "symbol": self.symbol,
            "term": self.term,
            "metrics_json": self.metrics_json,
        }


@dataclass
class ModuleLoss:
    """某功能模块在某批次的loss"""
    batch_id: str = ""
    module_name: str = ""           # stock_short_term/stock_medium_term/stock_long_term
    line_id: str = ""
    L_return: float = 0.0
    L_risk: float = 0.0
    L_structure: float = 0.0
    L_total: float = 0.0
    score_total: float = 0.0
    sub_breakdown_json: str = ""
    sample_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "module_name": self.module_name,
            "line_id": self.line_id,
            "L_return": self.L_return,
            "L_risk": self.L_risk,
            "L_structure": self.L_structure,
            "L_total": self.L_total,
            "score_total": self.score_total,
            "sub_breakdown_json": self.sub_breakdown_json,
            "sample_size": self.sample_size,
        }


@dataclass
class AgentContribution:
    """某Agent的消融实验结果"""
    batch_id: str = ""
    term: str = ""
    agent_name: str = ""
    delta_L_total: float = 0.0
    delta_L_return: float = 0.0
    delta_L_risk: float = 0.0
    delta_L_structure: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    significance: str = ""          # significant_positive/significant_negative/not_significant
    stars: str = ""                 # ★★★/★★/☆/↓/↓↓↓
    sample_size: int = 0
    eval_mode: str = ""             # real/backtest
    market_regime_breakdown_json: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "term": self.term,
            "agent_name": self.agent_name,
            "delta_L_total": self.delta_L_total,
            "delta_L_return": self.delta_L_return,
            "delta_L_risk": self.delta_L_risk,
            "delta_L_structure": self.delta_L_structure,
            "ci_95_lower": self.ci_95_lower,
            "ci_95_upper": self.ci_95_upper,
            "significance": self.significance,
            "stars": self.stars,
            "sample_size": self.sample_size,
            "eval_mode": self.eval_mode,
            "market_regime_breakdown_json": self.market_regime_breakdown_json,
        }


@dataclass
class OptimizationTicket:
    """优化建议项"""
    ticket_id: str = ""
    batch_id: str = ""
    ticket_type: str = ""           # PARAM_TUNE/PROMPT_PATCH/LOGIC_FIX/ARCH_CHANGE/RESEARCH
    severity: str = "medium"        # high/medium/low
    title: str = ""
    summary: str = ""
    evidence_json: str = ""         # 触发证据JSON
    route: str = ""                 # auto/semi_auto/manual
    status: str = "pending"         # pending/accepted/rejected/implemented/rolled_back
    patch_path: str = ""
    manual_package_path: str = ""
    before_loss: float = 0.0
    after_loss: float = 0.0         # 实施后回填

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "batch_id": self.batch_id,
            "ticket_type": self.ticket_type,
            "severity": self.severity,
            "title": self.title,
            "summary": self.summary,
            "evidence_json": self.evidence_json,
            "route": self.route,
            "status": self.status,
            "patch_path": self.patch_path,
            "manual_package_path": self.manual_package_path,
            "before_loss": self.before_loss,
            "after_loss": self.after_loss,
        }
