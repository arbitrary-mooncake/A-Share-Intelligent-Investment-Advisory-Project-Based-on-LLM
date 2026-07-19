"""Strict score-result boundary for evaluation and advisory workflows.

This module deliberately does not guess a score when a scorer fails.  It is a
small, dependency-free boundary that can be reused before legacy dictionaries
are allowed into ranking, simulation, or contribution calculations.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union


class ScoreValidity(str, Enum):
    """Whether a score is safe to use as an investment assessment."""

    VALID = "valid"
    ABSTAIN = "abstain"
    INVALID = "invalid"


class ScoreAssessmentError(ValueError):
    """Base error for the score boundary."""


class ScoreAssessmentSchemaError(ScoreAssessmentError):
    """The producer returned parsed data that violates the score schema.

    Schema/programming errors are raised instead of being converted to a
    neutral score.  Only a JSON *syntax* failure is eligible for one repair.
    """


RepairCallable = Callable[[str], Union[str, Mapping[str, Any]]]


def _string_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ScoreAssessmentSchemaError(f"{field_name} must be an array of strings")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ScoreAssessmentSchemaError(
                f"{field_name} must contain only non-empty strings"
            )
        normalized = item.strip()
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def _strict_float(value: Any, field_name: str) -> float:
    # bool is an int subclass, but accepting True as score=1 is a schema bug.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreAssessmentSchemaError(f"{field_name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ScoreAssessmentSchemaError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True)
class ScoreAssessment:
    """Validated scoring outcome.

    ``score`` exists only for ``valid`` assessments.  This invariant prevents
    operational failures from silently becoming a 40/50/hold conclusion.
    """

    validity: ScoreValidity
    score: Optional[float] = None
    action: Optional[str] = None
    coverage: float = 0.0
    missing_core_fields: Tuple[str, ...] = field(default_factory=tuple)
    missing_optional_fields: Tuple[str, ...] = field(default_factory=tuple)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    repaired: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.validity, ScoreValidity):
            raise ScoreAssessmentSchemaError("validity must be a ScoreValidity")
        if isinstance(self.coverage, bool) or not isinstance(self.coverage, (int, float)):
            raise ScoreAssessmentSchemaError("coverage must be a JSON number")
        if not math.isfinite(float(self.coverage)) or not 0.0 <= float(self.coverage) <= 1.0:
            raise ScoreAssessmentSchemaError("coverage must be finite and between 0 and 1")
        object.__setattr__(self, "coverage", float(self.coverage))
        object.__setattr__(
            self,
            "missing_core_fields",
            _string_tuple(self.missing_core_fields, "missing_core_fields"),
        )
        object.__setattr__(
            self,
            "missing_optional_fields",
            _string_tuple(self.missing_optional_fields, "missing_optional_fields"),
        )

        if self.validity is ScoreValidity.VALID:
            if self.score is None:
                raise ScoreAssessmentSchemaError("valid assessment requires score")
            score = _strict_float(self.score, "score")
            if not 0.0 <= score <= 100.0:
                raise ScoreAssessmentSchemaError("score must be between 0 and 100")
            if self.missing_core_fields:
                raise ScoreAssessmentSchemaError(
                    "valid assessment cannot have missing_core_fields"
                )
            object.__setattr__(self, "score", score)
        elif self.score is not None:
            raise ScoreAssessmentSchemaError(
                "abstain/invalid assessment must not carry an actionable score"
            )

        if self.action is not None and (
            not isinstance(self.action, str) or not self.action.strip()
        ):
            raise ScoreAssessmentSchemaError("action must be a non-empty string or null")

    @property
    def usable(self) -> bool:
        """True only when the result may enter ranking/simulation."""

        return self.validity is ScoreValidity.VALID

    @property
    def failure_category(self) -> Optional[str]:
        """Compatibility/readability alias for ``error_type``."""

        return self.error_type

    @property
    def reason(self) -> Optional[str]:
        """Compatibility/readability alias for ``error_message``."""

        return self.error_message

    def require_valid(self) -> float:
        """Return the score or fail closed at a downstream usage boundary."""

        if not self.usable or self.score is None:
            raise ScoreAssessmentError(
                f"score is not usable: {self.validity.value} ({self.error_type or 'no_error_type'})"
            )
        return self.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "validity": self.validity.value,
            "score": self.score,
            "action": self.action,
            "coverage": self.coverage,
            "missing_core_fields": list(self.missing_core_fields),
            "missing_optional_fields": list(self.missing_optional_fields),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "failure_category": self.error_type,
            "reason": self.error_message,
            "repaired": self.repaired,
        }


def _failure(
    validity: ScoreValidity,
    error_type: str,
    error_message: str,
    *,
    coverage: float = 0.0,
    missing_core_fields: Iterable[str] = (),
    missing_optional_fields: Iterable[str] = (),
    repaired: bool = False,
) -> ScoreAssessment:
    return ScoreAssessment(
        validity=validity,
        score=None,
        action=None,
        coverage=coverage,
        missing_core_fields=tuple(missing_core_fields),
        missing_optional_fields=tuple(missing_optional_fields),
        error_type=error_type,
        error_message=error_message,
        repaired=repaired,
    )


def _parse_mapping(
    payload: Mapping[str, Any],
    *,
    core_fields: Tuple[str, ...],
    legacy_is_invalid: bool,
    repaired: bool,
) -> ScoreAssessment:
    if "validity" not in payload:
        if legacy_is_invalid:
            return _failure(
                ScoreValidity.INVALID,
                "legacy_missing_validity",
                "legacy score payload has no explicit validity contract",
                missing_core_fields=core_fields,
                repaired=repaired,
            )
        raise ScoreAssessmentSchemaError("validity is required")

    raw_validity = payload["validity"]
    if not isinstance(raw_validity, str):
        raise ScoreAssessmentSchemaError("validity must be a string")
    try:
        validity = ScoreValidity(raw_validity.strip().lower())
    except ValueError as exc:
        raise ScoreAssessmentSchemaError(
            "validity must be one of valid/abstain/invalid"
        ) from exc

    if validity is ScoreValidity.VALID and "coverage" not in payload:
        raise ScoreAssessmentSchemaError("valid assessment requires coverage")
    coverage = _strict_float(payload.get("coverage", 0.0), "coverage")
    if not 0.0 <= coverage <= 1.0:
        raise ScoreAssessmentSchemaError("coverage must be between 0 and 1")

    missing_core = _string_tuple(
        payload.get("missing_core_fields", ()), "missing_core_fields"
    )
    missing_optional = _string_tuple(
        payload.get("missing_optional_fields", ()), "missing_optional_fields"
    )
    missing_from_payload = tuple(
        name for name in core_fields if payload.get(name) in (None, "")
    )
    missing_core = tuple(dict.fromkeys((*missing_core, *missing_from_payload)))

    risk_gate = payload.get("risk_gate")
    if risk_gate is not None and not isinstance(risk_gate, Mapping):
        raise ScoreAssessmentSchemaError("risk_gate must be an object")
    if risk_gate and "abstain" in risk_gate and not isinstance(risk_gate["abstain"], bool):
        raise ScoreAssessmentSchemaError("risk_gate.abstain must be a boolean")
    gate_abstain = bool(risk_gate and risk_gate.get("abstain") is True)

    # An explicitly invalid producer result is an operational failure, not a
    # controlled abstention.  Preserve that classification even when the
    # producer also lists missing core fields.  Only a producer that *claims*
    # to be valid may be downgraded to abstain by evidence/risk gates.
    if validity is not ScoreValidity.VALID:
        if payload.get("score") is not None:
            raise ScoreAssessmentSchemaError(
                "abstain/invalid payload must not carry score"
            )
        error_type = payload.get("error_type", payload.get("failure_category"))
        error_message = payload.get("error_message", payload.get("reason"))
        if error_type is not None and not isinstance(error_type, str):
            raise ScoreAssessmentSchemaError("error_type must be a string or null")
        if error_message is not None and not isinstance(error_message, str):
            raise ScoreAssessmentSchemaError("error_message must be a string or null")
        return ScoreAssessment(
            validity=validity,
            coverage=coverage,
            missing_core_fields=missing_core,
            missing_optional_fields=missing_optional,
            error_type=error_type,
            error_message=error_message,
            repaired=repaired,
        )

    # Runtime missing data and an explicit risk-gate abstention are controlled
    # non-results only for an otherwise claimed-valid result.
    if missing_core or gate_abstain:
        reason = "risk_gate_abstain" if gate_abstain else "missing_core_data"
        return _failure(
            ScoreValidity.ABSTAIN,
            reason,
            str(payload.get("error_message") or reason),
            coverage=coverage,
            missing_core_fields=missing_core,
            missing_optional_fields=missing_optional,
            repaired=repaired,
        )

    action = payload.get("action")
    if action is not None and not isinstance(action, str):
        raise ScoreAssessmentSchemaError("action must be a string or null")
    return ScoreAssessment(
        validity=ScoreValidity.VALID,
        score=_strict_float(payload.get("score"), "score"),
        action=action,
        coverage=coverage,
        missing_optional_fields=missing_optional,
        repaired=repaired,
    )


def assess_score_payload(
    payload: Union[str, Mapping[str, Any]],
    *,
    core_fields: Iterable[str] = (),
    legacy_is_invalid: bool = True,
    repair: Optional[RepairCallable] = None,
) -> ScoreAssessment:
    """Parse one score payload and enforce fail-closed semantics.

    A repair callable is invoked at most once, and only if a raw string is not
    valid JSON.  Once JSON has parsed, any schema/type mismatch is a producer
    programming error and is raised immediately.
    """

    normalized_core = _string_tuple(tuple(core_fields), "core_fields")
    repaired = False
    parsed: Any = payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as original_error:
            if repair is None:
                return _failure(
                    ScoreValidity.ABSTAIN,
                    "llm_output_invalid",
                    f"LLM output is not valid JSON: {original_error.msg}",
                    missing_core_fields=normalized_core,
                )
            repaired = True
            try:
                repaired_payload = repair(payload)
            except Exception as repair_error:
                return _failure(
                    ScoreValidity.ABSTAIN,
                    "llm_output_invalid",
                    f"single JSON repair failed: {repair_error}",
                    missing_core_fields=normalized_core,
                    repaired=True,
                )
            if isinstance(repaired_payload, str):
                try:
                    parsed = json.loads(repaired_payload)
                except json.JSONDecodeError as repair_parse_error:
                    return _failure(
                        ScoreValidity.ABSTAIN,
                        "llm_output_invalid",
                        f"repaired output is not valid JSON: {repair_parse_error.msg}",
                        missing_core_fields=normalized_core,
                        repaired=True,
                    )
            else:
                parsed = repaired_payload

    if not isinstance(parsed, Mapping):
        raise ScoreAssessmentSchemaError("score payload must be a JSON object")
    return _parse_mapping(
        parsed,
        core_fields=normalized_core,
        legacy_is_invalid=legacy_is_invalid,
        repaired=repaired,
    )


# Stable alias for callers that prefer parser terminology.
parse_score_assessment = assess_score_payload


__all__ = [
    "ScoreAssessment",
    "ScoreAssessmentError",
    "ScoreAssessmentSchemaError",
    "ScoreValidity",
    "assess_score_payload",
    "parse_score_assessment",
]
