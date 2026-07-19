"""Tests for the strict score validity/LLM repair boundary."""
import pytest

from src.eval.score_assessment import (
    ScoreAssessmentSchemaError,
    ScoreValidity,
    assess_score_payload,
)


def test_valid_score_is_usable_and_strictly_bounded():
    result = assess_score_payload({
        "validity": "valid",
        "score": 72.5,
        "action": "buy",
        "coverage": 0.8,
        "missing_optional_fields": ["news"],
    })

    assert result.validity is ScoreValidity.VALID
    assert result.usable
    assert result.require_valid() == 72.5
    assert result.coverage == 0.8
    assert result.missing_optional_fields == ("news",)


@pytest.mark.parametrize("bad_score", [-0.01, 100.01, float("nan"), float("inf"), True, "50"])
def test_invalid_score_types_or_ranges_raise_schema_error(bad_score):
    with pytest.raises(ScoreAssessmentSchemaError):
        assess_score_payload({
            "validity": "valid", "score": bad_score, "coverage": 1.0,
        })


def test_legacy_score_without_validity_is_invalid_not_neutral():
    result = assess_score_payload({"score": 50, "action": "hold"})

    assert result.validity is ScoreValidity.INVALID
    assert result.score is None
    assert result.failure_category == "legacy_missing_validity"


def test_missing_core_field_or_risk_gate_abstain_removes_score():
    missing = assess_score_payload(
        {"validity": "valid", "score": 80, "coverage": 0.5},
        core_fields=("financial_evidence",),
    )
    gated = assess_score_payload({
        "validity": "valid",
        "score": 80,
        "coverage": 1.0,
        "risk_gate": {"abstain": True},
    })

    assert missing.validity is ScoreValidity.ABSTAIN
    assert missing.score is None
    assert missing.missing_core_fields == ("financial_evidence",)
    assert gated.validity is ScoreValidity.ABSTAIN
    assert gated.failure_category == "risk_gate_abstain"


def test_explicit_invalid_with_missing_core_remains_invalid():
    result = assess_score_payload({
        "validity": "invalid",
        "coverage": 0.0,
        "missing_core_fields": ["financial_data"],
        "error_type": "provider_failure",
        "error_message": "upstream unavailable",
    })

    assert result.validity is ScoreValidity.INVALID
    assert result.missing_core_fields == ("financial_data",)
    assert result.failure_category == "provider_failure"


def test_raw_json_syntax_failure_gets_exactly_one_repair():
    calls = []

    def repair(raw):
        calls.append(raw)
        return '{"validity":"valid","score":67,"coverage":1}'

    result = assess_score_payload("not json", repair=repair)

    assert len(calls) == 1
    assert result.validity is ScoreValidity.VALID
    assert result.score == 67
    assert result.repaired is True


def test_failed_repair_abstains_without_a_second_attempt():
    calls = []

    def repair(raw):
        calls.append(raw)
        return "still not json"

    result = assess_score_payload("bad", repair=repair)

    assert len(calls) == 1
    assert result.validity is ScoreValidity.ABSTAIN
    assert result.score is None
    assert result.failure_category == "llm_output_invalid"


def test_parsed_schema_error_does_not_call_repair():
    calls = []

    def repair(raw):
        calls.append(raw)
        return {"validity": "valid", "score": 50, "coverage": 1}

    with pytest.raises(ScoreAssessmentSchemaError):
        assess_score_payload(
            '{"validity":"valid","score":"50","coverage":1}', repair=repair
        )

    assert calls == []


def test_non_valid_assessment_cannot_smuggle_a_score():
    with pytest.raises(ScoreAssessmentSchemaError):
        assess_score_payload({
            "validity": "abstain", "score": 50, "coverage": 0.2,
        })
