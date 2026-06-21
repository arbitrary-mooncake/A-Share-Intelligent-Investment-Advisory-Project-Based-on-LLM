"""
评测系统配置管理 — 集中管理所有可配置参数。
优先级: 环境变量 > config JSON > 默认值
"""
import os
import json
from typing import Dict, Any, Optional


# 配置目录
_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "eval"
)

# ── 默认配置 ──

DEFAULTS: Dict[str, Any] = {
    # 存储
    "storage_dir": "data/eval",
    "db_path": "data/eval/eval.db",

    # 模型配置
    "eval_fast_model_profile": "eval_analysis",      # 日常分析Agent
    "eval_report_model_profile": "eval_orchestrator", # 报告撰写
    "eval_optimizer_model_profile": "eval_orchestrator",  # 优化路由

    # 精筛池大小
    "stock_pool_short_size": 100,
    "stock_pool_medium_size": 80,
    "stock_pool_long_size": 60,

    # 初筛
    "hard_screen_min_daily_amount": 20000000,  # 2000万
    "hard_screen_min_list_days": 60,
    "quick_screen_threshold_short": 50,
    "quick_screen_threshold_medium": 50,
    "quick_screen_threshold_long": 50,
    "whitelist_pass_ratio": 1.2,  # 白名单:初筛通过 = 1:1.2

    # 黑名单
    "blacklist_expiry_days": 120,

    # 分级打分频率
    "grading_enabled": True,
    "monday_full_run": True,
    "score_change_upgrade_threshold": 10.0,    # 评分变化超过此值→升频
    "score_stable_downgrade_threshold": 3.0,   # 评分变化低于此值→降频
    "score_high_zone": 75,                     # 稳定高分区
    "score_low_zone": 45,                      # 稳定低分区

    # 策略参数
    "max_positions_short_ablation": 10,
    "max_positions_short_longhold": 12,
    "max_positions_medium": 8,
    "max_positions_long": 5,

    "single_weight_limit_short_ablation": 0.10,
    "single_weight_limit_short_longhold": 0.12,
    "single_weight_limit_medium": 0.18,
    "single_weight_limit_long": 0.25,

    "min_cash_ratio_short": 0.10,
    "min_cash_ratio_medium": 0.15,
    "min_cash_ratio_long": 0.20,

    "score_buy_threshold_short": 60,
    "score_buy_threshold_medium": 65,
    "score_buy_threshold_long": 70,

    "score_sell_hard_short": 35,
    "score_sell_soft_low_short": 35,
    "score_sell_soft_high_short": 45,
    "score_sell_hard_medium": 45,
    "score_sell_hard_long": 35,

    # 市场仿真
    "commission_rate": 0.00025,
    "min_commission": 5.0,
    "transfer_fee_rate": 0.00002,
    "stamp_tax_rate": 0.001,
    "slippage_base": 0.0005,
    "slippage_hs300": 0.0002,
    "slippage_small_cap": 0.001,
    "max_single_order_ratio": 0.05,
    "max_daily_buy_ratio": 0.10,
    "suspension_force_days": 20,
    "suspension_discount": 0.95,

    # 回测
    "backtest_medium_start": "2022-01-01",
    "backtest_medium_end": "2026-05-31",
    "backtest_long_start": "2020-01-01",
    "backtest_long_end": "2025-06-30",
    "backtest_train_split": "2023-12-31",
    "backtest_validation_split": "2025-06-30",

    # Loss权重
    "loss_effect_weight": 0.75,
    "loss_stability_weight": 0.15,
    "loss_efficiency_weight": 0.10,

    "loss_return_weight_short": 0.45,
    "loss_risk_weight_short": 0.40,
    "loss_structure_weight_short": 0.15,

    "loss_return_weight_medium": 0.50,
    "loss_risk_weight_medium": 0.35,
    "loss_structure_weight_medium": 0.15,

    "loss_return_weight_long": 0.55,
    "loss_risk_weight_long": 0.30,
    "loss_structure_weight_long": 0.15,

    # 统计
    "bootstrap_iterations": 10000,
    "significance_level": 0.95,
    "permutation_iterations": 10000,

    # 自动优化
    "autofix_enabled": True,
    "autofix_path_whitelist": [
        "config/eval/",
        "src/utils/model_config.py",
    ],
    "autofix_max_files": 3,
    "autofix_cooldown_days": 7,
    "autofix_max_consecutive_failures": 3,

    # 系统
    "max_concurrent_jobs": 4,
    "initial_capital_per_line": 1000000.0,  # 每条线100万
}


def _load_json_config(filename: str) -> Dict[str, Any]:
    """加载JSON配置文件"""
    path = os.path.join(_CONFIG_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_config() -> Dict[str, Any]:
    """
    获取完整配置。
    优先级: 环境变量 > config/eval/defaults.json > 代码默认值
    """
    config = dict(DEFAULTS)

    # 加载JSON配置覆盖
    json_config = _load_json_config("defaults.json")
    config.update(json_config)

    # 环境变量覆盖（EVAL_前缀）
    for key in config:
        env_key = f"EVAL_{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            # 尝试类型转换
            default_val = config[key]
            if isinstance(default_val, bool):
                config[key] = env_val.lower() in ("true", "1", "yes")
            elif isinstance(default_val, int):
                try:
                    config[key] = int(env_val)
                except ValueError:
                    pass
            elif isinstance(default_val, float):
                try:
                    config[key] = float(env_val)
                except ValueError:
                    pass
            elif isinstance(default_val, list):
                config[key] = [x.strip() for x in env_val.split(",")]
            else:
                config[key] = env_val

    return config


def get(key: str, default: Any = None) -> Any:
    """获取单个配置项"""
    return get_config().get(key, default)
