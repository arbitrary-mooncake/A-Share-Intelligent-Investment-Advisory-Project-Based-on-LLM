"""
模型策略配置表（4.4 定稿：ModelPolicyRegistry 降级为纯配置表）。

每个场景同时声明：模型路由 agent 名、输出预算、thinking、超时。
缓存键带模型与 prompt 版本的要求由确定性 scorer 的 weights_version 体现。
"""
from typing import Any, Dict

MODEL_POLICIES: Dict[str, Dict[str, Any]] = {
    # 冲突仲裁：快速档、不开 thinking、有界输出
    "conflict_arbitration": {
        "agent_name": "conflict_arbiter",
        "max_tokens": 1000,
        "thinking": False,
        "timeout": 60,
        "temperature": 0.2,
    },
    # 打分解释：快速档、不开 thinking、懒加载
    "score_explanation": {
        "agent_name": "score_explainer",
        "max_tokens": 2000,
        "thinking": False,
        "timeout": 90,
        "temperature": 0.4,
    },
    # L2 快筛：不开 thinking（已在 pool_screening 落地，此处备案）
    "layer2_quick_screen": {
        "thinking": False,
        "max_tokens": 1200,
    },
}


def get_policy(scene: str) -> Dict[str, Any]:
    return MODEL_POLICIES.get(scene, {})
