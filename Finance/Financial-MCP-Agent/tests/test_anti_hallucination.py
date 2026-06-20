"""
Tests for the 5-layer anti-hallucination system (总纲 §12).

Covers:
  Layer 1: Input Structuring — strips narrative text, converts numerics
  Layer 2: Output Validation — catches hallucinated fields, schema violations
  Layer 3: Code Verification — numerical traceability, entity existence,
           logical consistency, comparison validity
  Layer 4: Self-Consistency — cross-run claim comparison
  Layer 5: Confidence Labeling — HIGH/MEDIUM/LOW assignment
"""
import json
import os
import sys

import pytest

from src.eval.optimizer.anti_hallucination import (
    # Layer 1
    structure_input,
    # Layer 2
    validate_output,
    ValidationError,
    # Layer 3
    verify_numerical_traceability,
    verify_entity_existence,
    verify_logical_consistency,
    verify_comparison_validity,
    # Layer 4
    check_self_consistency,
    # Layer 5
    label_confidence,
    ConfidenceLabel,
    VerificationResult,
    # Pipeline
    AntiHallucinationPipeline,
    quick_verify,
    verify_with_consistency,
    # Helpers
    _extract_json_from_response,
    _extract_stock_codes,
    _try_parse_numeric,
    _is_narrative_field,
)


# ═══════════════════════════════════════════════════════════════════
# Layer 1: Input Structuring
# ═══════════════════════════════════════════════════════════════════

class TestLayer1InputStructuring:

    def test_strips_analysis_text_fields(self):
        """输入结构化应剥离 LLM 生成的叙事文本字段。"""
        data = {
            "batch_id": "eval_20260619_001",
            "term": "medium",
            "fundamental_analysis": "该公司基本面良好，盈利能力强，建议买入...",  # 叙事文本应被剥离
            "agents": {
                "fundamental": {
                    "delta_L_total": 0.052,
                    "ci_95": [0.018, 0.086],
                    "analysis": "长文本" * 200,  # >500字叙事文本应被剥离
                }
            }
        }
        result = structure_input(data)
        # 叙事字段不应存在
        assert "fundamental_analysis" not in result
        # 短数值字段应保留
        assert "batch_id" in result
        assert result["batch_id"] == "eval_20260619_001"
        # agents子结构中的超长analysis应被剥离
        assert "agents" in result
        assert "analysis" not in result["agents"]["fundamental"]

    def test_preserves_numeric_fields(self):
        """数值字段应保留并保持类型。"""
        data = {
            "delta": 0.052,
            "sample_size": 230,
            "ci_95": [0.018, 0.086],
            "significance": "significant_positive",
        }
        result = structure_input(data)
        assert result["delta"] == 0.052
        assert result["sample_size"] == 230
        assert result["ci_95"] == [0.018, 0.086]

    def test_converts_string_numerics(self):
        """字符串形式的数值应转为原生数值类型。"""
        data = {
            "value_str_int": "42",
            "value_str_float": "3.14159",
            "value_str_pure_text": "hello world",
        }
        result = structure_input(data)
        assert result["value_str_int"] == 42
        assert isinstance(result["value_str_int"], int)
        assert result["value_str_float"] == pytest.approx(3.14159)
        assert isinstance(result["value_str_float"], float)
        assert result["value_str_pure_text"] == "hello world"

    def test_adds_meta_information(self):
        """顶层应添加 _meta 元信息。"""
        data = {"key": "value"}
        result = structure_input(data)
        assert "_meta" in result
        assert "structured_at" in result["_meta"]
        assert "total_fields" in result["_meta"]

    def test_handles_empty_input(self):
        """空输入不应崩溃。"""
        result = structure_input({})
        assert "_meta" in result
        assert "primary_keys" in result["_meta"]

    def test_handles_non_dict_input_safely(self):
        """非dict输入应安全处理。"""
        result = structure_input(None)
        assert isinstance(result, dict)
        assert "_meta" in result

    def test_strips_deep_narrative_fields(self):
        """深层嵌套的叙事字段也应剥离。"""
        data = {
            "level1": {
                "level2": {
                    "narrative_summary": "这是一段很长的分析文本...",
                    "value": 123.45,
                }
            }
        }
        result = structure_input(data)
        assert "level1" in result
        assert "level2" in result["level1"]
        assert "narrative_summary" not in result["level1"]["level2"]
        assert result["level1"]["level2"]["value"] == 123.45

    def test_max_depth_protection(self):
        """超深嵌套不应无限递归。"""
        deep = {}
        current = deep
        for i in range(20):
            current["next"] = {"val": i}
            current = current["next"]
        result = structure_input(deep)
        assert isinstance(result, dict)
        assert "_meta" in result


# ═══════════════════════════════════════════════════════════════════
# Layer 2: Output Validation
# ═══════════════════════════════════════════════════════════════════

class TestLayer2OutputValidation:

    def test_validates_correct_json(self):
        """正确的输出应通过验证。"""
        output = json.dumps({
            "diagnosis": {
                "top_findings": [
                    {
                        "claim": "fundamental agent正向贡献",
                        "confidence": "HIGH",
                        "supporting_evidence_ids": ["abl_fund_deltaL"],
                        "counter_evidence": "震荡市贡献略低"
                    }
                ],
                "agent_ranking": [],
                "market_regime_insights": {}
            },
            "optimization_suggestions": [
                {
                    "type": "PARAM_TUNE",
                    "priority": "high",
                    "target_file": "config/eval/defaults.json",
                    "target_param": "medium_term.scorer_weights.moneyflow",
                    "current_value": 0.10,
                    "suggested_range": [0.03, 0.07],
                    "rationale": "基于证据调整",
                    "expected_impact": {},
                    "verification_method": "回测",
                    "risk_level": "low"
                }
            ],
            "narrative_summary": "测试摘要"
        }, ensure_ascii=False)
        source = {"abl_fund_deltaL": 0.052}
        result = validate_output(output, source_data=source)
        assert result["diagnosis"]["top_findings"][0]["claim"] == "fundamental agent正向贡献"

    def test_rejects_missing_required_fields(self):
        """缺少必要字段应抛出ValidationError。"""
        output = json.dumps({"optimization_suggestions": []})
        with pytest.raises(ValidationError) as exc_info:
            validate_output(output)
        assert "diagnosis" in str(exc_info.value) or "failed_fields" in str(exc_info.value.__dict__)

    def test_rejects_invalid_confidence(self):
        """非法置信度值应被检测。"""
        output = json.dumps({
            "diagnosis": {
                "top_findings": [
                    {"claim": "test", "confidence": "SUPER_HIGH",
                     "supporting_evidence_ids": [], "counter_evidence": ""}
                ],
                "agent_ranking": [],
                "market_regime_insights": {}
            },
            "optimization_suggestions": []
        }, ensure_ascii=False)
        with pytest.raises(ValidationError) as exc_info:
            validate_output(output)
        assert len(exc_info.value.hallucinated_fields) > 0

    def test_rejects_invalid_ticket_type(self):
        """非法ticket类型应被检测。"""
        output = json.dumps({
            "diagnosis": {
                "top_findings": [],
                "agent_ranking": [],
                "market_regime_insights": {}
            },
            "optimization_suggestions": [
                {"type": "INVALID_TYPE", "priority": "low"}
            ]
        }, ensure_ascii=False)
        with pytest.raises(ValidationError) as exc_info:
            validate_output(output)
        assert len(exc_info.value.hallucinated_fields) > 0

    def test_extracts_json_from_markdown_block(self):
        """应能从markdown code block中提取JSON。"""
        output = '```json\n{"diagnosis": {"top_findings": [], "agent_ranking": [], "market_regime_insights": {}}, "optimization_suggestions": []}\n```'
        result = validate_output(output)
        assert "diagnosis" in result

    def test_extracts_json_with_surrounding_text(self):
        """应能从含前后文本的响应中提取JSON。"""
        output = '一些前置描述...\n{"diagnosis": {"top_findings": [], "agent_ranking": [], "market_regime_insights": {}}, "optimization_suggestions": []}\n一些后续文本...'
        result = validate_output(output)
        assert "diagnosis" in result

    def test_validation_error_on_non_json(self):
        """纯文本无JSON应抛出ValidationError。"""
        with pytest.raises(ValidationError):
            validate_output("这是纯文本，没有JSON。")

    def test_data_id_reference_validation(self):
        """引用的数据ID必须在输入数据中存在。"""
        source = {"known_id_1": 123, "known_id_2": 456}
        output = json.dumps({
            "diagnosis": {
                "top_findings": [
                    {"claim": "test", "confidence": "HIGH",
                     "supporting_evidence_ids": ["unknown_id_fake"],
                     "counter_evidence": ""}
                ],
                "agent_ranking": [],
                "market_regime_insights": {}
            },
            "optimization_suggestions": []
        }, ensure_ascii=False)
        with pytest.raises(ValidationError) as exc_info:
            validate_output(output, source_data=source)
        assert any("unknown_id_fake" in h for h in exc_info.value.hallucinated_fields)

    def test_stock_code_hallucination_detected(self):
        """输出中的股票代码若不在输入数据中应被标记。"""
        source = {"known_codes": "sh.603871"}
        output = json.dumps({
            "diagnosis": {
                "top_findings": [
                    {"claim": "测试股票 sh.999999", "confidence": "HIGH",
                     "supporting_evidence_ids": [], "counter_evidence": ""}
                ],
                "agent_ranking": [],
                "market_regime_insights": {}
            },
            "optimization_suggestions": []
        }, ensure_ascii=False)
        with pytest.raises(ValidationError) as exc_info:
            validate_output(output, source_data=source)
        assert any("999999" in h for h in exc_info.value.hallucinated_fields)

    def test_top_level_not_object_raises(self):
        """顶层不是JSON对象应失败。"""
        with pytest.raises(ValidationError):
            validate_output('[1, 2, 3]')


# ═══════════════════════════════════════════════════════════════════
# Layer 3: Code Verification
# ═══════════════════════════════════════════════════════════════════

class TestLayer3NumericalTraceability:

    def test_all_numbers_traceable(self):
        """所有数字都能在源数据中找到对应项。"""
        source = {
            "agents": {
                "fundamental": {"delta_L_total": 0.052, "ci_95": [0.018, 0.086]}
            }
        }
        changes = {
            "rationale": "fundamental agent的delta为0.052，ci范围0.018-0.086"
        }
        passed, issues = verify_numerical_traceability(changes, source)
        assert passed
        assert len(issues) == 0

    def test_flags_untraceable_numbers(self):
        """找不到对应项的数字应被标记。"""
        source = {"agents": {"fundamental": {"delta_L_total": 0.052}}}
        changes = {
            "rationale": "fundamental贡献度为0.99999，远超预期"  # 0.99999不在源数据中
        }
        passed, issues = verify_numerical_traceability(changes, source)
        assert not passed
        assert len(issues) > 0
        assert any("0.99999" in issue for issue in issues) or any(
            str(0.99999) in issue for issue in issues)

    def test_traceable_in_nested_dicts(self):
        """深层嵌套的数值也应能被追踪。"""
        source = {
            "deep": {"nested": {"values": [0.001, 0.002, 0.003]}}
        }
        changes = {"text": "最小值为0.001"}
        passed, issues = verify_numerical_traceability(changes, source)
        assert passed


class TestLayer3EntityExistence:

    def test_known_agents_pass(self):
        """已知agent应通过验证。"""
        entities = [{"type": "agent", "name": "fundamental_agent"},
                    {"type": "agent", "name": "short_term_scorer"}]
        passed, issues = verify_entity_existence(entities, {})
        assert passed
        assert len(issues) == 0

    def test_unknown_agent_fails(self):
        """未知agent应被检出。"""
        entities = [{"type": "agent", "name": "super_agent_v99"}]
        passed, issues = verify_entity_existence(entities, {})
        assert not passed
        assert len(issues) > 0

    def test_file_existence_with_known_paths(self):
        """已知文件路径应通过。"""
        entities = [{"type": "path", "name": "config/eval/strategy_defaults.json"}]
        ref = {"known_paths": {"config/eval/strategy_defaults.json"}}
        passed, issues = verify_entity_existence(entities, ref)
        assert passed

    def test_absolute_path_that_exists(self):
        """绝对路径存在时应通过。"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"x = 1\n")
            tmp = f.name
        try:
            entities = [{"type": "path", "name": tmp}]
            passed, _ = verify_entity_existence(entities, {"known_paths": set()})
            assert passed
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


class TestLayer3LogicalConsistency:

    def test_detects_claim_vs_negative_delta(self):
        """claim说正贡献但delta为负 → 应标记矛盾。"""
        before = {
            "agents": {
                "fundamental": {"delta_L_total": -0.052}
            }
        }
        after = {
            "diagnosis": {
                "top_findings": [
                    {
                        "claim": "fundamental agent在短期维度有显著正贡献",
                        "confidence": "HIGH",
                        "supporting_evidence_ids": ["abl_fundamental"],
                    }
                ],
                "agent_ranking": [],
                "market_regime_insights": {}
            },
            "optimization_suggestions": []
        }
        contradictions = verify_logical_consistency(before, after)
        assert len(contradictions) > 0
        assert any("负值" in c or "negative" in c.lower() for c in contradictions)

    def test_no_false_positives_on_consistent_data(self):
        """一致的数据不应产生误报。"""
        before = {
            "agents": {
                "fundamental": {"delta_L_total": 0.052}
            }
        }
        after = {
            "diagnosis": {
                "top_findings": [
                    {
                        "claim": "fundamental agent有正贡献",
                        "confidence": "HIGH",
                        "supporting_evidence_ids": ["abl_fundamental"],
                    }
                ],
                "agent_ranking": [],
                "market_regime_insights": {}
            },
            "optimization_suggestions": []
        }
        contradictions = verify_logical_consistency(before, after)
        assert len(contradictions) == 0

    def test_detects_ranking_direction_inconsistency(self):
        """agent排名的delta方向应与输入数据一致。"""
        before = {
            "agents": {
                "fundamental": {"delta_L_total": -0.052}
            }
        }
        after = {
            "diagnosis": {
                "top_findings": [],
                "agent_ranking": [
                    {"agent": "fundamental", "delta": 0.05}
                ],
                "market_regime_insights": {}
            },
            "optimization_suggestions": []
        }
        contradictions = verify_logical_consistency(before, after)
        assert len(contradictions) > 0

    def test_empty_inputs_no_crash(self):
        """空输入不应崩溃。"""
        contradictions = verify_logical_consistency({}, {})
        assert contradictions == []


class TestLayer3ComparisonValidity:

    def test_valid_comparison_passes(self):
        """实际值满足比较条件应通过。"""
        baseline = {"agents": {"fundamental": {"delta_L_total": 0.052}}}
        comparisons = [{
            "left": "agents.fundamental.delta_L_total",
            "operator": "gt",
            "right": 0.0,
            "claim_text": "fundamental delta > 0"
        }]
        passed, issues = verify_comparison_validity(comparisons, baseline)
        assert passed
        assert len(issues) == 0

    def test_invalid_comparison_fails(self):
        """实际值不满足比较条件应失败。"""
        baseline = {"agents": {"fundamental": {"delta_L_total": -0.052}}}
        comparisons = [{
            "left": "agents.fundamental.delta_L_total",
            "operator": "gt",
            "right": 0.0,
            "claim_text": "fundamental delta > 0"
        }]
        passed, issues = verify_comparison_validity(comparisons, baseline)
        assert not passed
        assert len(issues) > 0

    def test_missing_path_reported(self):
        """不存在的路径应被报告。"""
        baseline = {"agents": {}}
        comparisons = [{
            "left": "agents.nonexistent.delta",
            "operator": "gt",
            "right": 0.0,
            "claim_text": "test"
        }]
        passed, issues = verify_comparison_validity(comparisons, baseline)
        assert not passed

    def test_nonnumeric_field_reported(self):
        """非数值字段用于比较应被报告。"""
        baseline = {"agents": {"name": "string_value"}}
        comparisons = [{
            "left": "agents.name",
            "operator": "gt",
            "right": 0.0,
            "claim_text": "compare string"
        }]
        passed, issues = verify_comparison_validity(comparisons, baseline)
        assert not passed

    def test_lt_operator(self):
        """小于运算符应正确验证。"""
        baseline = {"val": -5.0}
        comparisons = [{"left": "val", "operator": "lt", "right": 0.0,
                        "claim_text": "val < 0"}]
        passed, _ = verify_comparison_validity(comparisons, baseline)
        assert passed

    def test_eq_operator(self):
        """等于运算符应正确验证。"""
        baseline = {"val": 3.14}
        comparisons = [{"left": "val", "operator": "eq", "right": 3.14,
                        "claim_text": "val == 3.14"}]
        passed, _ = verify_comparison_validity(comparisons, baseline)
        assert passed


# ═══════════════════════════════════════════════════════════════════
# Layer 4: Self-Consistency
# ═══════════════════════════════════════════════════════════════════

class TestLayer4SelfConsistency:

    def test_identical_results_are_consistent(self):
        """相同结果应返回高一致性。"""
        r1 = {
            "diagnosis": {
                "top_findings": [
                    {"claim": "fundamental正贡献", "confidence": "HIGH"}
                ]
            },
            "optimization_suggestions": [
                {"type": "PARAM_TUNE"}
            ]
        }
        r2 = copy_deep(r1)
        result = check_self_consistency([r1, r2])
        assert result["is_consistent"]
        assert result["consistency_score"] >= 0.9

    def test_different_claims_flagged(self):
        """不同声明应被标记。"""
        r1 = {
            "diagnosis": {
                "top_findings": [
                    {"claim": "fundamental正贡献", "confidence": "HIGH"}
                ]
            },
            "optimization_suggestions": [
                {"type": "PARAM_TUNE"}
            ]
        }
        r2 = {
            "diagnosis": {
                "top_findings": [
                    {"claim": "fundamental负贡献", "confidence": "LOW"}
                ]
            },
            "optimization_suggestions": [
                {"type": "PROMPT_PATCH"}
            ]
        }
        result = check_self_consistency([r1, r2])
        assert not result["is_consistent"]
        assert len(result["discrepancies"]) > 0

    def test_single_result_skips_check(self):
        """单个结果跳过自洽性校验。"""
        r1 = {"diagnosis": {"top_findings": []}, "optimization_suggestions": []}
        result = check_self_consistency([r1])
        assert result["is_consistent"]
        assert "少于2个结果" in result.get("note", "")

    def test_different_suggestion_types_flagged(self):
        """不同优化建议类型应被标记。"""
        r1 = {
            "diagnosis": {"top_findings": []},
            "optimization_suggestions": [
                {"type": "PARAM_TUNE"},
                {"type": "RESEARCH"}
            ]
        }
        r2 = {
            "diagnosis": {"top_findings": []},
            "optimization_suggestions": [
                {"type": "ARCH_CHANGE"},
                {"type": "LOGIC_FIX"}
            ]
        }
        result = check_self_consistency([r1, r2])
        assert not result["is_consistent"]

    def test_empty_results_safe(self):
        """空结果列表不应崩溃。"""
        result = check_self_consistency([])
        assert result["is_consistent"]
        assert "少于2个结果" in result.get("note", "")


# ═══════════════════════════════════════════════════════════════════
# Layer 5: Confidence Labeling
# ═══════════════════════════════════════════════════════════════════

class TestLayer5ConfidenceLabeling:

    def test_all_pass_yields_high(self):
        """所有验证通过 → HIGH。"""
        results = [
            VerificationResult(layer="L1", passed=True, score=1.0),
            VerificationResult(layer="L2", passed=True, score=1.0),
            VerificationResult(layer="L3", passed=True, score=1.0),
            VerificationResult(layer="L4", passed=True, score=1.0),
        ]
        label = label_confidence(results)
        assert label.level == "HIGH"
        assert label.can_auto_act

    def test_minor_issues_yields_medium(self):
        """1-2层有minor问题 → MEDIUM。"""
        results = [
            VerificationResult(layer="L1", passed=True, score=1.0),
            VerificationResult(layer="L2", passed=False, score=0.8,
                               issues=["minor: optional field missing"],
                               is_critical=False),
            VerificationResult(layer="L3", passed=False, score=0.5,
                               issues=["minor: 1 value untraceable"],
                               is_critical=False),
            VerificationResult(layer="L4", passed=True, score=1.0),
        ]
        label = label_confidence(results)
        assert label.level == "MEDIUM"
        assert label.can_auto_act  # MEDIUM仍可自动操作

    def test_critical_failure_yields_low(self):
        """有critical失败 → LOW。"""
        results = [
            VerificationResult(layer="L1", passed=True, score=1.0),
            VerificationResult(layer="L2", passed=False, score=0.0,
                               issues=["critical: 3 hallucinated fields"],
                               is_critical=True),
            VerificationResult(layer="L3", passed=True, score=1.0),
            VerificationResult(layer="L4", passed=True, score=1.0),
        ]
        label = label_confidence(results)
        assert label.level == "LOW"
        assert not label.can_auto_act  # LOW不触发自动修改

    def test_many_layer_failures_yields_low(self):
        """>=3层失败 → LOW，即使没有critical标记。"""
        results = [
            VerificationResult(layer="L1", passed=False, score=0.5,
                               issues=["issue"], is_critical=False),
            VerificationResult(layer="L2", passed=False, score=0.5,
                               issues=["issue"], is_critical=False),
            VerificationResult(layer="L3", passed=False, score=0.5,
                               issues=["issue"], is_critical=False),
            VerificationResult(layer="L4", passed=True, score=1.0),
        ]
        label = label_confidence(results)
        assert label.level == "LOW"
        assert not label.can_auto_act

    def test_no_results_defaults_high(self):
        """无验证结果 → HIGH (安全默认)。"""
        label = label_confidence([])
        assert label.level == "HIGH"

    def test_reasons_include_layer_info(self):
        """返回的reasons应包含layer信息。"""
        results = [
            VerificationResult(layer="L2", passed=False, score=0.5,
                               issues=["test issue"], is_critical=False),
            VerificationResult(layer="L1", passed=True, score=1.0),
        ]
        label = label_confidence(results)
        assert any("L2" in r or "test issue" in r for r in label.reasons)


# ═══════════════════════════════════════════════════════════════════
# Pipeline Integration
# ═══════════════════════════════════════════════════════════════════

class TestPipeline:

    def test_pipeline_runs_all_layers(self):
        """流水线应执行所有层级并返回完整结果。"""
        source = {
            "batch_id": "eval_test_001",
            "term": "medium",
            "agents": {
                "fundamental": {"delta_L_total": 0.052, "ci_95": [0.018, 0.086]}
            }
        }
        llm_output = json.dumps({
            "diagnosis": {
                "top_findings": [
                    {"claim": "fundamental正贡献", "confidence": "HIGH",
                     "supporting_evidence_ids": [], "counter_evidence": ""}
                ],
                "agent_ranking": [
                    {"agent": "fundamental", "delta": 0.05}
                ],
                "market_regime_insights": {}
            },
            "optimization_suggestions": [
                {"type": "PARAM_TUNE", "priority": "high",
                 "target_file": "config/eval/defaults.json",
                 "target_param": "weight", "current_value": 0.10,
                 "suggested_range": [0.03, 0.07],
                 "rationale": "fundamental delta=0.052",
                 "expected_impact": {}, "verification_method": "test",
                 "risk_level": "low"}
            ],
            "narrative_summary": "test"
        }, ensure_ascii=False)

        result = quick_verify(llm_output, source)
        assert "structured_input" in result
        assert "validated_output" in result
        assert "verifications" in result
        assert "confidence" in result
        assert "overall_pass" in result
        assert "summary" in result

    def test_pipeline_handles_bad_llm_output(self):
        """流水线处理坏LLM输出不应崩溃。"""
        source = {"batch_id": "test"}
        result = quick_verify("这不是JSON格式的输出...", source)
        assert not result["overall_pass"]
        assert result["confidence"]["level"] in ("MEDIUM", "LOW")

    def test_verify_with_consistency_multiple_outputs(self):
        """带自洽性校验的验证应处理多个输出。"""
        source = {"batch_id": "test"}
        output1 = json.dumps({
            "diagnosis": {"top_findings": [], "agent_ranking": [],
                          "market_regime_insights": {}},
            "optimization_suggestions": []
        })
        output2 = output1  # 相同
        result = verify_with_consistency([output1, output2], source)
        assert "confidence" in result

    def test_pipeline_graceful_on_missing_known_entities(self):
        """known_entities缺失时流水线不崩溃。"""
        source = {"batch_id": "test"}
        output = json.dumps({
            "diagnosis": {"top_findings": [], "agent_ranking": [],
                          "market_regime_insights": {}},
            "optimization_suggestions": [
                {"type": "PARAM_TUNE"}
            ]
        })
        pipeline = AntiHallucinationPipeline()
        result = pipeline.run(output, source, known_entities=None)
        assert "confidence" in result


# ═══════════════════════════════════════════════════════════════════
# Helper Unit Tests
# ═══════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_extract_json_from_plain(self):
        result = _extract_json_from_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_extract_json_from_markdown(self):
        result = _extract_json_from_response(
            '```json\n{"key": "value"}\n```'
        )
        assert result == {"key": "value"}

    def test_extract_json_from_embedded(self):
        result = _extract_json_from_response(
            '一些文本 {"key": "value"} 更多文本'
        )
        assert result == {"key": "value"}

    def test_extract_json_invalid_returns_none(self):
        result = _extract_json_from_response("纯文本无JSON")
        assert result is None

    def test_extract_stock_codes_shanghai(self):
        codes = _extract_stock_codes("分析股票 sh.600000 和 601166")
        assert "sh.600000" in codes or "601166" in codes

    def test_extract_stock_codes_shenzhen(self):
        codes = _extract_stock_codes("关注 sz.000001")
        assert "sz.000001" in codes

    def test_try_parse_numeric_int(self):
        assert _try_parse_numeric("42") == 42
        assert isinstance(_try_parse_numeric("42"), int)

    def test_try_parse_numeric_float(self):
        result = _try_parse_numeric("3.14")
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_try_parse_numeric_text(self):
        assert _try_parse_numeric("hello") == "hello"

    def test_is_narrative_field_true(self):
        assert _is_narrative_field("fundamental_analysis")
        assert _is_narrative_field("narrative_summary")
        assert _is_narrative_field("analysis")

    def test_is_narrative_field_false(self):
        assert not _is_narrative_field("delta_L_total")
        assert not _is_narrative_field("batch_id")
        assert not _is_narrative_field("ci_95")


# ═══════════════════════════════════════════════════════════════════
# Edge Cases & Robustness
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_deeply_nested_source_traceability(self):
        """深层嵌套的源数据中的数值仍应可追踪。"""
        source = {"a": {"b": {"c": {"d": {"e": 0.00001}}}}}
        changes = {"text": "最小值为0.00001"}
        passed, issues = verify_numerical_traceability(changes, source)
        assert passed

    def test_boolean_not_treated_as_numeric(self):
        """布尔值不应被当作数值追踪。"""
        source = {"flag": True}
        changes = {"text": "flag为1.0"}  # 1.0在source中不应该被追踪到（True != 1.0在语义上）
        passed, issues = verify_numerical_traceability(changes, source)
        # True被序列化为1.0但verify使用的是_deep_search_value，它不会把True当作数值
        # 所以1.0应该被认为是不可追踪的
        assert not passed

    def test_unicode_in_validation(self):
        """中文内容应在验证中正确处理。"""
        source = {"abl_测试": 123}
        output = json.dumps({
            "diagnosis": {
                "top_findings": [
                    {"claim": "这是一个中文测试", "confidence": "HIGH",
                     "supporting_evidence_ids": ["abl_测试"],
                     "counter_evidence": "没问题"}
                ],
                "agent_ranking": [],
                "market_regime_insights": {}
            },
            "optimization_suggestions": []
        }, ensure_ascii=False)
        result = validate_output(output, source_data=source)
        assert result["diagnosis"]["top_findings"][0]["claim"] == "这是一个中文测试"

    def test_structure_input_idempotent(self):
        """已结构化的输入再结构化应保持稳定。"""
        data = {"batch_id": "test", "value": 42, "extra_analysis": "text"}
        r1 = structure_input(data)
        r2 = structure_input(r1)  # 二次结构化
        assert r2["batch_id"] == "test"
        assert r2["value"] == 42

    def test_empty_validated_output_for_verification(self):
        """空的validated输出用于代码验证不应崩溃。"""
        passed, issues = verify_numerical_traceability({}, {"test": 1})
        assert passed
        assert len(issues) == 0

    def test_verification_result_dataclass(self):
        """VerificationResult数据类应有正确的默认值。"""
        vr = VerificationResult(layer="test", passed=True, score=1.0)
        assert vr.layer == "test"
        assert vr.issues == []
        assert not vr.is_critical

    def test_confidence_label_dataclass(self):
        """ConfidenceLabel应正确设置can_auto_act。"""
        cl = ConfidenceLabel(level="LOW", reasons=["test"])
        assert not cl.can_auto_act

        cl2 = ConfidenceLabel(level="HIGH", reasons=["test"])
        assert cl2.can_auto_act


# ── Helper for deep copy in tests ──

def copy_deep(obj):
    """深度复制用于测试。"""
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
