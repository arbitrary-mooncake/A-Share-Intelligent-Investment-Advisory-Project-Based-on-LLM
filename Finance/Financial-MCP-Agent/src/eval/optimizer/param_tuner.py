"""
贝叶斯参数优化器 — 自动搜索最优参数组合（总纲 §11.3）。

实现思路：
  - 首轮：在参数空间内均匀网格采样（latin hypercube 近似）
  - 迭代轮：基于历史最优附近进行自适应细化（局部搜索 + 随机扰动）
  - 损失函数由调用方注入（通常来自 LossEngine），本模块只负责搜索调度
  - 纯 Python 实现，不依赖外部优化库

使用示例：
    tuner = ParamTuner()
    result = tuner.tune(
        param_space={"threshold": (30, 80), "weight": (0.1, 0.5)},
        loss_fn=lambda p: compute_loss(p["threshold"], p["weight"]),
        n_iterations=50
    )
"""

import copy
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple


def _uniform_sample_bounds(bounds: Tuple[float, float], step: Optional[float] = None) -> float:
    """在区间 [lo, hi] 内采样；若 step 提供则按 step 离散化。"""
    lo, hi = bounds
    val = lo + random.random() * (hi - lo)
    if step is not None and step > 0:
        val = round(val / step) * step
        val = max(lo, min(hi, val))
    return val


def _int_bounds_cast(bounds: Tuple[float, float]) -> Tuple[int, int]:
    """将浮点 bounds 转换为整数区间。"""
    return (int(math.ceil(bounds[0])), int(math.floor(bounds[1])))


class ParamTuner:
    """贝叶斯参数优化器 — 自动搜索最优参数组合。

    策略：先做均匀网格搜索（grid），再在最优解附近做自适应细化（refine）。
    这模拟了贝叶斯优化的探索-利用思想，但实现简洁可控。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化调参器。

        Args:
            config: 可选配置字典，支持键：
                - random_seed (int): 随机种子，默认 42
                - grid_fraction (float): 网格搜索占总迭代数的比例，默认 0.4
                - local_search_radius (float): 局部搜索半径（占参数区间的比例），默认 0.15
                - local_search_decay (float): 每轮局部搜索半径衰减因子，默认 0.92
                - n_top_candidates (int): 用于细化的历史最优候选数，默认 3
        """
        cfg = config or {}
        self.random_seed = cfg.get("random_seed", 42)
        self.grid_fraction = cfg.get("grid_fraction", 0.4)
        self.local_search_radius = cfg.get("local_search_radius", 0.15)
        self.local_search_decay = cfg.get("local_search_decay", 0.92)
        self.n_top_candidates = cfg.get("n_top_candidates", 3)

    # ── Public API ────────────────────────────────────────────────

    def tune(
        self,
        param_space: Dict[str, Any],
        loss_fn: Callable[[Dict[str, Any]], float],
        n_iterations: int = 50,
    ) -> Dict[str, Any]:
        """运行贝叶斯优化搜索。

        Args:
            param_space: 参数空间定义。每个键的值可以是：
                - (lo, hi) 元组：连续参数区间
                - [val1, val2, ...] 列表：离散候选值
                - {"type": "int", "bounds": (lo, hi)}：整数参数
            loss_fn: 损失函数 callable(params_dict) -> float（越小越好）
            n_iterations: 总迭代次数

        Returns:
            {
                "best_params": {param_name: best_value, ...},
                "best_loss": float,
                "optimization_history": [
                    {"params": ..., "loss": ..., "phase": "grid"/"refine"},
                    ...
                ]
            }
        """
        random.seed(self.random_seed)

        n_grid = max(4, int(n_iterations * self.grid_fraction))
        n_refine = n_iterations - n_grid

        history: List[Dict[str, Any]] = []

        # Phase 1: Grid search (uniform sampling)
        for i in range(n_grid):
            params = self._sample_uniform(param_space)
            loss = self._evaluate_params(params, loss_fn)
            history.append({"params": copy.deepcopy(params), "loss": loss, "phase": "grid"})

        # Phase 2: Adaptive refinement around best candidates
        radius = self.local_search_radius
        for i in range(n_refine):
            params = self._suggest_next_params(param_space, history, radius)
            loss = self._evaluate_params(params, loss_fn)
            history.append({"params": copy.deepcopy(params), "loss": loss, "phase": "refine"})
            radius *= self.local_search_decay  # gradually narrow

        # Find best
        best_entry = min(history, key=lambda e: e["loss"])
        return {
            "best_params": best_entry["params"],
            "best_loss": best_entry["loss"],
            "optimization_history": history,
            "n_iterations": n_iterations,
            "n_grid": n_grid,
            "n_refine": n_refine,
        }

    def _evaluate_params(
        self, params: Dict[str, Any], loss_fn: Callable[[Dict[str, Any]], float]
    ) -> float:
        """评估一组参数并返回 loss。封装为独立方法便于子类重写（如加噪声/缓存）。"""
        try:
            return float(loss_fn(params))
        except Exception:
            # 非法参数组合 → 返回一个大 loss
            return 1e9

    # ── 采样逻辑 ──────────────────────────────────────────────────

    def _sample_uniform(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """从参数空间均匀采样一组参数。"""
        params = {}
        for name, spec in param_space.items():
            params[name] = self._sample_one(name, spec, "uniform")
        return params

    def _sample_one(self, name: str, spec: Any, mode: str = "uniform") -> Any:
        """对单个参数采样。"""
        if isinstance(spec, (list, tuple)):
            if len(spec) == 2 and isinstance(spec[0], (int, float)) and isinstance(spec[1], (int, float)):
                # (lo, hi) continuous range
                if mode == "uniform":
                    return _uniform_sample_bounds((float(spec[0]), float(spec[1])))
                else:
                    return _uniform_sample_bounds((float(spec[0]), float(spec[1])))
            else:
                # Discrete choice list
                return random.choice(spec)

        if isinstance(spec, dict):
            stype = spec.get("type", "float")
            bounds = spec.get("bounds", (0.0, 1.0))
            if stype == "int":
                lo, hi = _int_bounds_cast((float(bounds[0]), float(bounds[1])))
                if lo > hi:
                    lo, hi = hi, lo
                return random.randint(lo, hi)
            else:
                step = spec.get("step")
                return _uniform_sample_bounds((float(bounds[0]), float(bounds[1])), step)

        # Fallback: treat as single fixed value
        return spec

    # ── 自适应建议 ────────────────────────────────────────────────

    def _suggest_next_params(
        self,
        param_space: Dict[str, Any],
        history: List[Dict[str, Any]],
        radius: float = 0.15,
    ) -> Dict[str, Any]:
        """基于历史结果建议下一组参数（local-refinement 策略）。

        步骤：
        1. 找出历史上 loss 最低的 n_top_candidates 组参数
        2. 随机选其中一组作为锚点
        3. 每个参数在锚点附近按 radius 做随机扰动
        4. 20% 概率跳出局部区域，全局重新采样（避免陷入局部最优）
        """
        # 20% 概率全局探索
        if random.random() < 0.20:
            return self._sample_uniform(param_space)

        # 排序历史，取 top N
        sorted_history = sorted(history, key=lambda e: e["loss"])
        top_candidates = sorted_history[: min(self.n_top_candidates, len(sorted_history))]

        # 随机选一个锚点
        anchor = random.choice(top_candidates)["params"]

        # 在每个参数维度上扰动
        new_params = {}
        for name, spec in param_space.items():
            base_val = anchor.get(name)
            new_params[name] = self._perturb_param(name, spec, base_val, radius)

        return new_params

    def _perturb_param(
        self, name: str, spec: Any, base_val: Any, radius: float
    ) -> Any:
        """对单个参数在 base_val 附近做 radius 比例的扰动。"""
        if base_val is None:
            return self._sample_one(name, spec, "uniform")

        # 获取参数区间
        lo, hi = self._get_bounds(name, spec)

        if isinstance(spec, (list, tuple)) and not (
            len(spec) == 2
            and isinstance(spec[0], (int, float))
            and isinstance(spec[1], (int, float))
        ):
            # Discrete list — with some probability, pick a neighbor
            if random.random() < 0.5:
                return base_val
            return random.choice(spec)

        if isinstance(spec, dict) and spec.get("type") == "int":
            span = hi - lo
            delta = max(1, int(span * radius / 2))
            new_val = int(base_val) + random.randint(-delta, delta)
            return max(lo, min(hi, new_val))

        # Continuous parameter
        span = hi - lo
        delta = random.gauss(0, span * radius)
        new_val = float(base_val) + delta
        return max(lo, min(hi, new_val))

    def _get_bounds(self, name: str, spec: Any) -> Tuple[float, float]:
        """从 spec 中提取参数的上下界。"""
        if isinstance(spec, (list, tuple)):
            if len(spec) == 2 and isinstance(spec[0], (int, float)) and isinstance(spec[1], (int, float)):
                return (float(spec[0]), float(spec[1]))
            else:
                return (0.0, float(len(spec) - 1))
        if isinstance(spec, dict):
            bounds = spec.get("bounds", (0.0, 1.0))
            return (float(bounds[0]), float(bounds[1]))
        return (0.0, 0.0)
