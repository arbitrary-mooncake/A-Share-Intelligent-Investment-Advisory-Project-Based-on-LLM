"""
贝叶斯参数优化器 — 自动搜索最优参数组合（总纲 §11.3）。

实现思路：
  - 首轮：Latin Hypercube Sampling 在参数空间内均匀采样
  - 迭代轮：基于历史最优附近进行自适应细化（Dirichlet + 高斯扰动）
  - Σ=1.0约束通过Dirichlet分布自动保持
  - 损失函数由调用方注入（通常来自 LossEngine），本模块只负责搜索调度
  - 支持 param_space.json 加载

使用示例：
    tuner = ParamTuner()
    # 方式1: 从JSON加载参数空间
    tuner.load_param_space("config/eval/param_space.json")
    result = tuner.tune(
        param_space=tuner.param_space_dict,
        loss_fn=lambda p: compute_loss(p),
        n_iterations=50
    )
    # 方式2: 直接传入参数字典（向后兼容）
    result = tuner.tune(
        param_space={"threshold": (30, 80), "weight": (0.1, 0.5)},
        loss_fn=lambda p: compute_loss(p["threshold"], p["weight"]),
        n_iterations=50
    )
"""

import copy
import json
import math
import os
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


# ── Dirichlet 采样 (纯Python实现，不依赖numpy) ────────────────

def _sample_dirichlet(alpha_list: List[float]) -> List[float]:
    """从 Dirichlet(alpha) 分布采样，返回归一化的向量 (Σ=1.0)。

    使用Gamma分布方法：X_i ~ Gamma(alpha_i, 1)，然后 Y_i = X_i / ΣX_j。
    这保证 Y 的每个分量在 (0,1) 且 ΣY = 1.0。

    当 alpha 不可用时（如已 import numpy），自动回退到 numpy 实现。
    """
    n = len(alpha_list)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # 尝试使用 numpy（如果可用，精度更高）
    try:
        import numpy as np
        samples = np.random.dirichlet(alpha_list)
        return list(samples)
    except ImportError:
        pass

    # 纯Python Gamma采样回退
    gamma_samples = []
    for alpha_i in alpha_list:
        if alpha_i <= 0:
            # alpha <= 0 无效，使用很小的默认值
            alpha_i = 1e-6
        # random.gammavariate(alpha, beta) where beta = 1/scale
        # Gamma(alpha, 1) 相当于 gammavariate(alpha, 1.0)
        gamma_samples.append(random.gammavariate(alpha_i, 1.0))

    total = sum(gamma_samples)
    if total <= 0:
        # 退化情况：均匀分布
        return [1.0 / n] * n

    return [g / total for g in gamma_samples]


def _dirichlet_from_base(base_values: List[float], concentration: float = 10.0) -> List[float]:
    """基于当前值(base_values)构建Dirichlet alpha并采样。

    alpha_i = base_i * concentration，浓度越大越集中，越小越分散。
    默认 concentration=10 使采样围绕当前值有适度波动。

    Args:
        base_values: 当前参数值列表（必须全为正, Σ≈1.0）
        concentration: 浓度参数（越高越集中）

    Returns:
        新采样值列表，Σ=1.0
    """
    n = len(base_values)
    # 确保所有值为正
    min_positive = 1.0 / (n * 100.0)
    safe_base = [max(v, min_positive) for v in base_values]
    # 归一化到 Σ=1
    base_sum = sum(safe_base)
    safe_base = [v / base_sum for v in safe_base]
    # alpha_i = base_i * concentration
    alpha = [v * concentration for v in safe_base]
    return _sample_dirichlet(alpha)


# ── Latin Hypercube Sampling (纯Python实现) ───────────────────

def _lhs_sample(bounds_list: List[Tuple[float, float]], n_samples: int,
                random_seed: Optional[int] = None) -> List[List[float]]:
    """Latin Hypercube Sampling — 在n维参数空间中均匀分层采样。

    将每个维度划分成n_samples个等概率区间，每个区间随机采1个点，
    然后随机排列各维度的样本顺序，保证低差异序列特性。

    如果 scipy.stats.qmc 可用则使用它（更高质量），否则回退到手工分层。

    Args:
        bounds_list: [(lo_1, hi_1), ..., (lo_d, hi_d)]，d维
        n_samples: 样本数
        random_seed: 随机种子

    Returns:
        n_samples × d 的样本矩阵
    """
    n_dims = len(bounds_list)
    if n_dims == 0 or n_samples == 0:
        return []

    # 尝试使用 scipy.stats.qmc
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_dims, seed=random_seed or 0)
        samples_unit = sampler.random(n=n_samples)
        # 缩放到各维度的实际区间
        result = []
        for i in range(n_samples):
            row = []
            for d in range(n_dims):
                lo, hi = bounds_list[d]
                row.append(float(lo + samples_unit[i][d] * (hi - lo)))
            result.append(row)
        return result
    except ImportError:
        pass

    # 手工分层采样回退
    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    # 对每个维度，在[0,1]上分n_samples个区间采样
    stratified = []
    for d in range(n_dims):
        intervals = []
        for i in range(n_samples):
            lo_frac = i / n_samples
            hi_frac = (i + 1) / n_samples
            val = lo_frac + rng.random() * (hi_frac - lo_frac)
            intervals.append(val)
        rng.shuffle(intervals)
        stratified.append(intervals)

    # 转置并缩放到实际区间
    result = []
    for i in range(n_samples):
        row = []
        for d in range(n_dims):
            lo, hi = bounds_list[d]
            row.append(float(lo + stratified[d][i] * (hi - lo)))
        result.append(row)

    return result


class ParamTuner:
    """贝叶斯参数优化器 — 自动搜索最优参数组合。

    策略：Stage 1 LHS + Stage 2 Dirichlet局部搜索。自动处理 Σ=1.0 约束。
    支持从 param_space.json 加载结构化参数空间定义。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化调参器。

        Args:
            config: 可选配置字典，支持键：
                - random_seed (int): 随机种子，默认 42
                - grid_fraction (float): Stage 1 采样占总迭代数的比例，默认 0.4
                - dirichlet_concentration (float): Dirichlet 浓度参数，默认 10.0
                - local_search_radius (float): 局部搜索半径（占参数区间的比例），默认 0.15
                - local_search_decay (float): 每轮局部搜索半径衰减因子，默认 0.92
                - n_top_candidates (int): 用于细化的历史最优候选数，默认 3
                - use_lhs (bool): Stage 1 是否使用Latin Hypercube Sampling，默认 True
                - param_space_path (str): param_space.json 路径
        """
        cfg = config or {}
        self.random_seed = cfg.get("random_seed", 42)
        self.grid_fraction = cfg.get("grid_fraction", 0.4)
        self.dirichlet_concentration = cfg.get("dirichlet_concentration", 10.0)
        self.local_search_radius = cfg.get("local_search_radius", 0.15)
        self.local_search_decay = cfg.get("local_search_decay", 0.92)
        self.n_top_candidates = cfg.get("n_top_candidates", 3)
        self.use_lhs = cfg.get("use_lhs", True)

        # Σ=1.0 约束组索引：{group_key: [param_name, ...]}
        self._sum_to_one_groups: Dict[str, List[str]] = {}
        # 参数→约束组的反向索引：{param_name: group_key}
        self._param_to_group: Dict[str, str] = {}
        # 从 JSON 加载的完整参数空间定义
        self.param_space_definition: Dict[str, Any] = {}
        # 扁平化的参数字典（供 backward-compatible tune() 使用）
        self.param_space_dict: Dict[str, Any] = {}

        # 自动加载 param_space.json（如果存在）
        param_space_path = cfg.get("param_space_path")
        if param_space_path:
            self.load_param_space(param_space_path)

    # ── 参数空间加载 ──────────────────────────────────────────

    def load_param_space(self, json_path: str) -> Dict[str, Any]:
        """从 JSON 文件加载结构化参数空间定义。

        解析 sum_to_one 约束组并构建参数索引。
        同时构建扁平的 param_space_dict 供 backward-compatible tune() 使用。

        Args:
            json_path: param_space.json 的路径

        Returns:
            完整的参数空间定义字典
        """
        if not os.path.isabs(json_path):
            # 相对于项目根目录
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            )
            json_path = os.path.join(project_root, json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            self.param_space_definition = json.load(f)

        # 解析预定义的 sum_to_one 组索引
        predefined_groups = self.param_space_definition.get("_sum_to_one_groups", {})
        for group_key, param_names in predefined_groups.items():
            if group_key.startswith("_"):
                continue
            self._sum_to_one_groups[group_key] = list(param_names)
            for pname in param_names:
                self._param_to_group[pname] = group_key

        # 扫描所有类别，自动发现 sum_to_one 组
        self._discover_sum_to_one_groups(self.param_space_definition)

        # 构建扁平化的参数字典
        self._build_flat_param_dict(self.param_space_definition)

        return self.param_space_definition

    def _discover_sum_to_one_groups(self, node: Dict[str, Any], prefix: str = ""):
        """递归扫描参数空间，自动发现标记了 constraint: sum_to_one 的参数组。"""
        for key, value in node.items():
            if key.startswith("_"):
                continue

            if isinstance(value, dict):
                if value.get("constraint") == "sum_to_one" and "parameters" in value:
                    # 这是一个 Σ=1.0 参数组
                    group_key = f"{prefix}{key}".replace(".", ".")
                    if group_key not in self._sum_to_one_groups:
                        params = list(value["parameters"].keys())
                        self._sum_to_one_groups[group_key] = params
                        for pname in params:
                            self._param_to_group[pname] = group_key
                elif "parameters" in value:
                    # 有 parameters 但没有 constraint → 检查内部
                    sub_params = value.get("parameters", {})
                    for pname, pspec in sub_params.items():
                        if group_key := self._param_to_group.get(pname):
                            pass  # 已在索引中
                else:
                    # 递归扫描
                    new_prefix = f"{prefix}{key}."
                    self._discover_sum_to_one_groups(value, new_prefix)

    def _build_flat_param_dict(self, node: Dict[str, Any]):
        """从参数空间定义构建扁平的 {param_name: spec} 字典。"""
        for key, value in node.items():
            if key.startswith("_"):
                continue

            if isinstance(value, dict):
                if "parameters" in value:
                    params = value["parameters"]
                    for pname, pspec in params.items():
                        # 确保 spec 是 ParamTuner 可识别的格式
                        if isinstance(pspec, dict):
                            ptype = pspec.get("type", "float")
                            bounds = pspec.get("bounds")
                            if bounds is None:
                                # 布尔值等：无 bounds
                                self.param_space_dict[pname] = [True, False] if ptype == "bool" else pspec.get("default")
                            elif ptype == "int":
                                self.param_space_dict[pname] = {
                                    "type": "int",
                                    "bounds": tuple(bounds),
                                }
                            elif ptype == "float":
                                self.param_space_dict[pname] = tuple(bounds)
                            else:
                                self.param_space_dict[pname] = tuple(bounds)
                        else:
                            self.param_space_dict[pname] = pspec
                else:
                    self._build_flat_param_dict(value)

    def _load_param_space(self, json_path: Optional[str] = None) -> None:
        """[向后兼容别名] 加载参数空间。"""
        if json_path:
            self.load_param_space(json_path)

    # ── 约束验证 ──────────────────────────────────────────────

    def validate_param_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证并修正参数的 Σ=1.0 约束。

        对每个 sum_to_one 组，检查其参数之和是否等于 1.0。
        如果漂移超过容忍度 (1e-6)，则归一化修正。

        Args:
            params: 待验证的参数字典

        Returns:
            {"valid": bool, "violations": [...], "corrected_params": dict}
        """
        violations = []
        corrected = dict(params)
        tolerance = 1e-6

        for group_key, param_names in self._sum_to_one_groups.items():
            # 获取该组所有参数的实际值
            group_values = []
            missing_params = []
            for pname in param_names:
                if pname in corrected:
                    group_values.append(corrected[pname])
                else:
                    missing_params.append(pname)

            if missing_params:
                violations.append({
                    "group": group_key,
                    "type": "missing_params",
                    "missing": missing_params,
                    "message": f"组 '{group_key}' 缺少参数: {missing_params}",
                })
                continue

            if len(group_values) == 0:
                continue

            total = sum(group_values)
            if abs(total - 1.0) > tolerance:
                violations.append({
                    "group": group_key,
                    "type": "sum_drift",
                    "actual_sum": total,
                    "drift": total - 1.0,
                    "message": (
                        f"组 '{group_key}' 参数之和={total:.6f}，"
                        f"漂移={total - 1.0:.2e}，已自动归一化"
                    ),
                })
                # 归一化
                if total > 0:
                    for pname in param_names:
                        if pname in corrected:
                            corrected[pname] = corrected[pname] / total

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "corrected_params": corrected,
        }

    def _normalize_sum_to_one_groups(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """对 params 中所有 sum_to_one 组执行归一化（原地修改 + 返回）。"""
        result = self.validate_param_constraints(params)
        return result["corrected_params"]

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
                    {"params": ..., "loss": ..., "phase": "lhs"/"refine"},
                    ...
                ]
            }
        """
        random.seed(self.random_seed)

        n_lhs = max(1, min(n_iterations, int(n_iterations * self.grid_fraction)))
        n_refine = n_iterations - n_lhs

        history: List[Dict[str, Any]] = []

        # Stage 1: Latin Hypercube Sampling (or uniform grid fallback)
        if self.use_lhs:
            lhs_samples = self._generate_lhs_samples(param_space, n_lhs)
            for sample_params in lhs_samples:
                # 验证并修正约束
                sample_params = self._normalize_sum_to_one_groups(sample_params)
                loss = self._evaluate_params(sample_params, loss_fn)
                history.append({"params": copy.deepcopy(sample_params), "loss": loss, "phase": "lhs"})
        else:
            for i in range(n_lhs):
                params = self._sample_uniform(param_space)
                params = self._normalize_sum_to_one_groups(params)
                loss = self._evaluate_params(params, loss_fn)
                history.append({"params": copy.deepcopy(params), "loss": loss, "phase": "grid"})

        # Stage 2: Adaptive refinement with Dirichlet perturbation
        radius = self.local_search_radius
        for i in range(n_refine):
            params = self._suggest_next_params(param_space, history, radius)
            # 验证并修正 Σ=1.0 约束
            params = self._normalize_sum_to_one_groups(params)
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
            "n_lhs": n_lhs,
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

    # ── LHS 采样 ──────────────────────────────────────────────

    def _generate_lhs_samples(
        self, param_space: Dict[str, Any], n_samples: int
    ) -> List[Dict[str, Any]]:
        """使用 LHS 在参数空间中生成 n_samples 组参数。

        Args:
            param_space: 参数空间定义
            n_samples: 样本数

        Returns:
            参数字典列表
        """
        if n_samples <= 0:
            return []

        # 收集所有连续参数及其 bounds
        param_names = []
        bounds_list = []
        param_specs = []

        for name, spec in param_space.items():
            param_type = self._classify_spec(spec)
            if param_type == "continuous":
                lo, hi = self._get_bounds(name, spec)
                param_names.append(name)
                bounds_list.append((lo, hi))
                param_specs.append(("continuous", spec))
            elif param_type == "integer":
                lo, hi = self._get_bounds(name, spec)
                # 转换为连续采样再取整
                param_names.append(name)
                bounds_list.append((lo - 0.499, hi + 0.499))
                param_specs.append(("integer", spec))
            elif param_type == "discrete":
                # 离散参数使用独立均匀采样（LHS 不适用）
                pass

        if len(param_names) == 0:
            # 无连续参数，回退到均匀采样
            return [self._sample_uniform(param_space) for _ in range(n_samples)]

        # LHS 生成样本
        lhs_matrix = _lhs_sample(bounds_list, n_samples, self.random_seed)

        # 组装参数字典
        results = []
        for row_idx in range(n_samples):
            params = {}
            # 填充 LHS 采样的连续/整数参数
            for j, pname in enumerate(param_names):
                raw_val = lhs_matrix[row_idx][j]
                ptype, pspec = param_specs[j]
                if ptype == "integer":
                    params[pname] = int(round(raw_val))
                else:
                    step = self._get_step(pspec)
                    if step is not None:
                        raw_val = round(raw_val / step) * step
                    lo, hi = self._get_bounds(pname, pspec)
                    params[pname] = max(lo, min(hi, raw_val))

            # 填充离散参数（独立均匀采样）
            for name, spec in param_space.items():
                if name not in params:
                    params[name] = self._sample_one(name, spec, "uniform")

            results.append(params)

        return results

    def _classify_spec(self, spec: Any) -> str:
        """分类参数规格类型：continuous / integer / discrete / fixed。"""
        if isinstance(spec, (list, tuple)):
            if len(spec) == 2 and isinstance(spec[0], (int, float)) and isinstance(spec[1], (int, float)):
                return "continuous"
            else:
                return "discrete"
        if isinstance(spec, dict):
            stype = spec.get("type", "float")
            if stype == "int":
                return "integer"
            else:
                return "continuous"
        return "fixed"

    def _get_step(self, spec: Any) -> Optional[float]:
        """提取 step 值。"""
        if isinstance(spec, dict):
            return spec.get("step")
        return None

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
        3. 对 sum_to_one 组使用 Dirichlet 采样扰动
        4. 对独立参数使用高斯扰动
        5. 20% 概率跳出局部区域，全局重新采样（避免陷入局部最优）
        6. 返回前验证并修正 Σ=1.0 约束
        """
        # 20% 概率全局探索
        if random.random() < 0.20:
            return self._sample_uniform(param_space)

        # 排序历史，取 top N
        sorted_history = sorted(history, key=lambda e: e["loss"])
        top_candidates = sorted_history[: min(self.n_top_candidates, len(sorted_history))]

        # 随机选一个锚点
        anchor = random.choice(top_candidates)["params"]

        # 收集已被 Dirichlet 组处理的参数（防止重复扰动）
        perturbed_params: set = set()

        # 先在每个参数维度上扰动
        new_params = {}
        for name, spec in param_space.items():
            # 如果此参数属于 sum_to_one 组，跳过（组处理在下面）
            if name in self._param_to_group and name not in perturbed_params:
                continue

            base_val = anchor.get(name)
            new_params[name] = self._perturb_param(name, spec, base_val, radius)

        # 对 sum_to_one 组做 Dirichlet 扰动
        processed_groups: set = set()
        for pname in list(self._param_to_group.keys()):
            group_key = self._param_to_group[pname]
            if group_key in processed_groups:
                continue
            processed_groups.add(group_key)

            group_params = self._sum_to_one_groups.get(group_key, [])
            if not group_params:
                continue

            # 获取该组所有参数的锚点值
            base_values = []
            for gp_name in group_params:
                if gp_name in anchor:
                    base_values.append(anchor[gp_name])
                elif gp_name in param_space:
                    # 参数不在锚点中，采样默认值
                    default_val = self._sample_one(gp_name, param_space[gp_name], "uniform")
                    base_values.append(default_val)
                else:
                    base_values.append(0.0)

            if len(base_values) == 0:
                continue

            # 确保 base_values 中每个值都 > 0
            safe_base = [max(v, 1e-6) for v in base_values]

            # Dirichlet 采样
            concentration = self.dirichlet_concentration
            new_group_values = _dirichlet_from_base(safe_base, concentration)

            # 赋值
            for i, gp_name in enumerate(group_params):
                if i < len(new_group_values):
                    new_params[gp_name] = new_group_values[i]
                    perturbed_params.add(gp_name)

        return new_params

    def _perturb_param(
        self, name: str, spec: Any, base_val: Any, radius: float
    ) -> Any:
        """对单个参数在 base_val 附近做 radius 比例的扰动。

        如果参数属于 sum_to_one 组，则由 _suggest_next_params 统一用 Dirichlet 处理，
        此处只在非组参数上做独立高斯扰动。
        """
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
