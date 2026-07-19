"""Point-in-time data boundary for trustworthy evaluation.

The production stock pipeline intentionally remains untouched.  Evaluation
code can inject a Tushare-compatible requester here and receives a read-only,
content-addressed bundle whose feature timestamps are provably no later than
the requested knowledge cutoff.  Future outcome labels use a separate channel.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union


PIT_SCHEMA_VERSION = "pit_v1"
# China has used a fixed UTC+08:00 offset since 1991.  A fixed offset avoids a
# hidden dependency on the optional ``tzdata`` wheel on Windows deployments.
SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")
UTC = timezone.utc


class PITBoundaryError(RuntimeError):
    """Base error for PIT contract violations."""


class PITDataUnavailableError(PITBoundaryError):
    """Core point-in-time data is unavailable."""


class PITTemporalViolation(PITBoundaryError):
    """Feature data crossed the knowledge cutoff."""


class PITSnapshotCorruptError(PITBoundaryError):
    """A persisted content-addressed snapshot failed verification."""


class PITMode(str, Enum):
    EXACT = "exact"
    BEST_EFFORT = "best_effort"


class EvidenceChannel(str, Enum):
    FEATURE = "feature"
    OUTCOME_LABEL = "outcome_label"


def _parse_date(value: Union[str, date, datetime], field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be date/datetime/string")
    normalized = value.strip()
    try:
        if len(normalized) == 8 and normalized.isdigit():
            return datetime.strptime(normalized, "%Y%m%d").date()
        return date.fromisoformat(normalized[:10])
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD or YYYYMMDD") from exc


def _parse_datetime(
    value: Union[str, date, datetime], field_name: str, *, date_at_end: bool = True
) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, time.max if date_at_end else time.min)
    elif isinstance(value, str):
        normalized = value.strip()
        if len(normalized) == 8 and normalized.isdigit():
            parsed_date = datetime.strptime(normalized, "%Y%m%d").date()
            result = datetime.combine(parsed_date, time.max if date_at_end else time.min)
        else:
            try:
                if len(normalized) == 10:
                    parsed_date = date.fromisoformat(normalized)
                    result = datetime.combine(
                        parsed_date, time.max if date_at_end else time.min
                    )
                else:
                    result = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{field_name} must be an ISO date/time") from exc
    else:
        raise TypeError(f"{field_name} must be date/datetime/string")
    if result.tzinfo is None:
        result = result.replace(tzinfo=SHANGHAI)
    return result.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _deep_thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_thaw(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            _deep_thaw(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PITBoundaryError(f"snapshot payload is not canonical JSON: {exc}") from exc


@dataclass(frozen=True)
class AsOfContext:
    """Immutable evaluation context included in every cache/snapshot identity."""

    trade_date: Union[str, date, datetime]
    knowledge_cutoff: Union[str, date, datetime]
    model_profile: str
    prompt_version: str
    data_snapshot_id: str = ""
    cache_namespace: str = "eval"
    schema_version: str = PIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        trade_date = _parse_date(self.trade_date, "trade_date")
        cutoff = _parse_datetime(self.knowledge_cutoff, "knowledge_cutoff")
        if trade_date > cutoff.astimezone(SHANGHAI).date():
            raise PITTemporalViolation("trade_date cannot be later than knowledge_cutoff")
        for name in ("model_profile", "prompt_version", "cache_namespace", "schema_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.data_snapshot_id, str):
            raise TypeError("data_snapshot_id must be a string")
        object.__setattr__(self, "trade_date", trade_date)
        object.__setattr__(self, "knowledge_cutoff", cutoff)

    @property
    def fingerprint(self) -> str:
        """Fingerprint covering all model, prompt, time, snapshot and cache axes."""

        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    def with_snapshot(self, snapshot_id: str) -> "AsOfContext":
        if not snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        return replace(self, data_snapshot_id=snapshot_id)

    def to_dict(self, *, include_snapshot: bool = True) -> dict[str, Any]:
        result = {
            "trade_date": self.trade_date.isoformat(),
            "knowledge_cutoff": _iso_utc(self.knowledge_cutoff),
            "model_profile": self.model_profile,
            "prompt_version": self.prompt_version,
            "cache_namespace": self.cache_namespace,
            "schema_version": self.schema_version,
        }
        if include_snapshot:
            result["data_snapshot_id"] = self.data_snapshot_id
        return result


@dataclass(frozen=True)
class TimedEvidence:
    """One normalized evidence row with explicit temporal semantics."""

    domain: str
    source: str
    available_at: Union[str, date, datetime]
    fetched_at: Union[str, date, datetime]
    payload: Mapping[str, Any]
    query_params: Mapping[str, Any] = field(default_factory=dict)
    channel: EvidenceChannel = EvidenceChannel.FEATURE
    pit_mode: PITMode = PITMode.EXACT
    row_hash: str = ""

    def __post_init__(self) -> None:
        if not self.domain or not self.source:
            raise ValueError("evidence domain/source must be non-empty")
        available_at = _parse_datetime(self.available_at, "available_at")
        fetched_at = _parse_datetime(self.fetched_at, "fetched_at", date_at_end=False)
        payload = _deep_freeze(self.payload)
        query_params = _deep_freeze(self.query_params)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "fetched_at", fetched_at)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "query_params", query_params)
        if not isinstance(self.channel, EvidenceChannel):
            object.__setattr__(self, "channel", EvidenceChannel(self.channel))
        if not isinstance(self.pit_mode, PITMode):
            object.__setattr__(self, "pit_mode", PITMode(self.pit_mode))
        expected = hashlib.sha256(_canonical_bytes(_deep_thaw(payload))).hexdigest()
        if self.row_hash and self.row_hash != expected:
            raise PITSnapshotCorruptError("evidence row_hash does not match payload")
        object.__setattr__(self, "row_hash", expected)

    def assert_allowed(self, context: AsOfContext) -> None:
        if self.channel is EvidenceChannel.FEATURE and self.available_at > context.knowledge_cutoff:
            raise PITTemporalViolation(
                f"feature {self.domain} available at {_iso_utc(self.available_at)} "
                f"after cutoff {_iso_utc(context.knowledge_cutoff)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "source": self.source,
            "available_at": _iso_utc(self.available_at),
            "fetched_at": _iso_utc(self.fetched_at),
            "payload": _deep_thaw(self.payload),
            "query_params": _deep_thaw(self.query_params),
            "channel": self.channel.value,
            "pit_mode": self.pit_mode.value,
            "row_hash": self.row_hash,
        }


@dataclass(frozen=True)
class FeatureLabelBoundary:
    """Keeps future realized outcomes out of scorer inputs by construction."""

    context: AsOfContext
    features: Tuple[TimedEvidence, ...] = field(default_factory=tuple)
    outcome_labels: Tuple[TimedEvidence, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "outcome_labels", tuple(self.outcome_labels))
        for evidence in self.features:
            if evidence.channel is not EvidenceChannel.FEATURE:
                raise PITBoundaryError("features may contain only feature evidence")
            evidence.assert_allowed(self.context)
        for label in self.outcome_labels:
            if label.channel is not EvidenceChannel.OUTCOME_LABEL:
                raise PITBoundaryError("outcome_labels require outcome_label channel")

    def scorer_payload(self) -> Tuple[Mapping[str, Any], ...]:
        """Return feature values only; future labels are never exposed."""

        return tuple(evidence.payload for evidence in self.features)


class ContentAddressedSnapshotStore:
    """Atomic, replayable storage keyed by canonical snapshot content."""

    def __init__(self, root: Optional[Union[str, Path]] = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.root = Path(root) if root is not None else (
            project_root / "data" / "eval" / "pit_snapshots" / PIT_SCHEMA_VERSION
        )
        self._lock = threading.RLock()

    @staticmethod
    def _digest(payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()

    def put(self, payload: Mapping[str, Any]) -> str:
        canonical = _canonical_bytes(payload)
        digest = hashlib.sha256(canonical).hexdigest()
        snapshot_id = f"sha256:{digest}"
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{digest}.json"
        with self._lock:
            if target.exists():
                existing = target.read_bytes()
                try:
                    existing_payload = json.loads(existing.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise PITSnapshotCorruptError(f"corrupt snapshot {snapshot_id}") from exc
                if self._digest(existing_payload) != digest:
                    raise PITSnapshotCorruptError(f"snapshot hash mismatch: {snapshot_id}")
                return snapshot_id
            fd, temp_name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=self.root)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(canonical)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, target)
            except Exception:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
                raise
        return snapshot_id

    def get(self, snapshot_id: str) -> dict[str, Any]:
        prefix = "sha256:"
        digest = snapshot_id[len(prefix):] if snapshot_id.startswith(prefix) else snapshot_id
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError("snapshot_id must be a SHA-256 digest")
        path = self.root / f"{digest}.json"
        if not path.exists():
            raise PITDataUnavailableError(f"snapshot not found: {snapshot_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PITSnapshotCorruptError(f"corrupt snapshot: {snapshot_id}") from exc
        if not isinstance(payload, dict) or self._digest(payload) != digest:
            raise PITSnapshotCorruptError(f"snapshot hash mismatch: {snapshot_id}")
        return payload


@dataclass(frozen=True)
class PITStockDataBundle:
    """Read-only normalized stock evidence used by a PIT scorer."""

    stock_code: str
    context: AsOfContext
    daily: Tuple[Mapping[str, Any], ...]
    daily_basic: Tuple[Mapping[str, Any], ...]
    fina_indicator: Tuple[Mapping[str, Any], ...]
    moneyflow: Tuple[Mapping[str, Any], ...]
    provenance: Tuple[TimedEvidence, ...]
    snapshot_id: str
    pit_mode: PITMode = PITMode.EXACT
    unsupported_domains: Tuple[str, ...] = ("news", "event")
    missing_optional_fields: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("daily", "daily_basic", "fina_indicator", "moneyflow"):
            object.__setattr__(
                self, name, tuple(_deep_freeze(row) for row in getattr(self, name))
            )
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "unsupported_domains", tuple(self.unsupported_domains))
        object.__setattr__(
            self, "missing_optional_fields", tuple(self.missing_optional_fields)
        )
        if not isinstance(self.pit_mode, PITMode):
            object.__setattr__(self, "pit_mode", PITMode(self.pit_mode))
        if self.context.data_snapshot_id != self.snapshot_id:
            raise PITBoundaryError("context data_snapshot_id must match bundle snapshot_id")
        for evidence in self.provenance:
            evidence.assert_allowed(self.context)
        if self.pit_mode is PITMode.EXACT and any(
            evidence.pit_mode is PITMode.BEST_EFFORT for evidence in self.provenance
        ):
            raise PITBoundaryError("best_effort evidence cannot be labelled exact")

    @property
    def exact(self) -> bool:
        return self.pit_mode is PITMode.EXACT

    def scorer_inputs(self) -> Mapping[str, Any]:
        """Immutable feature-only view; outcome labels cannot enter this bundle."""

        return MappingProxyType({
            "daily": self.daily,
            "daily_basic": self.daily_basic,
            "fina_indicator": self.fina_indicator,
            "moneyflow": self.moneyflow,
        })


Requester = Callable[[str, Mapping[str, Any], str], Optional[Mapping[str, Any]]]


def _response_rows(response: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not response:
        return []
    fields = response.get("fields", [])
    items = response.get("items", [])
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise PITBoundaryError("data source items must be an array")
    result = []
    for raw in items:
        if isinstance(raw, Mapping):
            row = {str(key): value for key, value in raw.items()}
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
                raise PITBoundaryError("data source fields must be an array")
            row = dict(zip((str(field) for field in fields), raw))
        else:
            raise PITBoundaryError("data source row must be an object or array")
        result.append(row)
    return result


def select_fina_indicator_rows(
    rows: Iterable[Mapping[str, Any]],
    context: AsOfContext,
    *,
    require_exact: bool = True,
) -> Tuple[Tuple[dict[str, Any], ...], PITMode]:
    """Select visible financial rows using each row's own vintage metadata.

    ``ann_date``, ``end_date`` and ``update_flag`` must come from the exact row
    whose financial values are retained.  A revised row (update_flag != 0)
    lacks a separate revision timestamp in this API and therefore can only be
    labelled ``best_effort``; it is rejected when exact PIT is required.
    """

    visible: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        missing = [key for key in ("ann_date", "end_date", "update_flag") if row.get(key) in (None, "")]
        if missing:
            # Missing row-level vintage cannot be repaired from another API or
            # from report period heuristics.
            continue
        ann_date = _parse_date(str(row["ann_date"]), "ann_date")
        end_date = _parse_date(str(row["end_date"]), "end_date")
        ann_available = _parse_datetime(ann_date, "ann_date")
        if ann_available > context.knowledge_cutoff:
            continue
        if end_date > context.trade_date:
            continue
        row["ann_date"] = ann_date.strftime("%Y%m%d")
        row["end_date"] = end_date.strftime("%Y%m%d")
        row["update_flag"] = str(row["update_flag"])
        visible.append(row)

    if not visible:
        raise PITDataUnavailableError(
            "no fina_indicator row has row-level ann_date/end_date/update_flag visible at cutoff"
        )

    # Keep the latest visible version for each reporting period, based solely
    # on that same row's ann_date.  No disclosure-map joins are allowed.
    by_period: dict[str, dict[str, Any]] = {}
    for row in visible:
        period = row["end_date"]
        previous = by_period.get(period)
        if previous is None or row["ann_date"] > previous["ann_date"]:
            by_period[period] = row
    selected = tuple(
        sorted(by_period.values(), key=lambda item: (item["end_date"], item["ann_date"]), reverse=True)
    )
    mode = (
        PITMode.EXACT
        if all(str(row["update_flag"]).strip() in {"0", "0.0"} for row in selected)
        else PITMode.BEST_EFFORT
    )
    if require_exact and mode is PITMode.BEST_EFFORT:
        raise PITDataUnavailableError(
            "fina_indicator contains revised values without a provable revision timestamp; "
            "best_effort cannot satisfy exact PIT"
        )
    return selected, mode


class PITDataGateway:
    """Build immutable, replayable PIT bundles through an injected requester."""

    def __init__(
        self,
        requester: Requester,
        *,
        snapshot_store: Optional[ContentAddressedSnapshotStore] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if not callable(requester):
            raise TypeError("requester must be callable")
        self._requester = requester
        self.snapshot_store = snapshot_store or ContentAddressedSnapshotStore()
        self._clock = clock or (lambda: datetime.now(tz=UTC))

    def _fetch(
        self, api_name: str, params: Mapping[str, Any], fields: Sequence[str]
    ) -> list[dict[str, Any]]:
        response = self._requester(api_name, dict(params), ",".join(fields))
        return _response_rows(response)

    def fetch_fina_indicator(
        self,
        ts_code: str,
        context: AsOfContext,
        *,
        lookback_years: int = 5,
        require_exact: bool = True,
    ) -> Tuple[Tuple[dict[str, Any], ...], PITMode]:
        start = date(context.trade_date.year - lookback_years, 1, 1).strftime("%Y%m%d")
        params = {
            "ts_code": ts_code,
            "start_date": start,
            "end_date": context.trade_date.strftime("%Y%m%d"),
        }
        fields = (
            "ts_code", "ann_date", "end_date", "update_flag", "roe", "roa",
            "grossprofit_margin", "netprofit_margin", "or_yoy", "profit_yoy",
            "debt_to_assets", "current_ratio", "ocf_to_or",
        )
        rows = self._fetch("fina_indicator", params, fields)
        return select_fina_indicator_rows(rows, context, require_exact=require_exact)

    @staticmethod
    def _filter_market_rows(
        rows: Iterable[Mapping[str, Any]],
        context: AsOfContext,
        date_field: str,
    ) -> Tuple[dict[str, Any], ...]:
        visible = []
        for raw in rows:
            row = dict(raw)
            raw_date = row.get(date_field)
            if not raw_date:
                continue
            row_date = _parse_date(str(raw_date), date_field)
            available = _parse_datetime(row_date, date_field)
            if row_date <= context.trade_date and available <= context.knowledge_cutoff:
                row[date_field] = row_date.strftime("%Y%m%d")
                visible.append(row)
        return tuple(sorted(visible, key=lambda item: item[date_field], reverse=True))

    def build_stock_bundle(
        self,
        ts_code: str,
        context: AsOfContext,
        *,
        lookback_days: int = 400,
        require_exact: bool = True,
    ) -> PITStockDataBundle:
        if context.data_snapshot_id:
            raise PITBoundaryError("build context must not already reference a snapshot")
        if not ts_code or not isinstance(ts_code, str):
            raise ValueError("ts_code must be a non-empty string")
        end = context.trade_date.strftime("%Y%m%d")
        start = (context.trade_date - timedelta(days=lookback_days)).strftime("%Y%m%d")
        fetched_at = self._clock()
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)

        daily_params = {"ts_code": ts_code, "start_date": start, "end_date": end}
        daily_fields = (
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "vol", "amount",
        )
        daily = self._filter_market_rows(
            self._fetch("daily", daily_params, daily_fields), context, "trade_date"
        )
        if not daily:
            raise PITDataUnavailableError("anchor/history daily prices are unavailable")

        basic_params = {"ts_code": ts_code, "trade_date": end}
        basic_fields = (
            "ts_code", "trade_date", "pe", "pb", "turnover_rate", "total_mv",
        )
        daily_basic = self._filter_market_rows(
            self._fetch("daily_basic", basic_params, basic_fields), context, "trade_date"
        )

        fina_indicator, financial_mode = self.fetch_fina_indicator(
            ts_code, context, require_exact=require_exact
        )

        moneyflow_params = {"ts_code": ts_code, "start_date": start, "end_date": end}
        moneyflow_fields = (
            "ts_code", "trade_date", "buy_sm_amount", "sell_sm_amount",
            "buy_md_amount", "sell_md_amount", "buy_lg_amount", "sell_lg_amount",
            "buy_elg_amount", "sell_elg_amount", "net_mf_amount",
        )
        moneyflow = self._filter_market_rows(
            self._fetch("moneyflow", moneyflow_params, moneyflow_fields),
            context,
            "trade_date",
        )

        rows_by_domain = {
            "daily": (daily, daily_params, PITMode.EXACT, "trade_date"),
            "daily_basic": (daily_basic, basic_params, PITMode.EXACT, "trade_date"),
            "fina_indicator": (
                fina_indicator,
                {
                    "ts_code": ts_code,
                    "start_date": date(context.trade_date.year - 5, 1, 1).strftime("%Y%m%d"),
                    "end_date": end,
                },
                financial_mode,
                "ann_date",
            ),
            "moneyflow": (moneyflow, moneyflow_params, PITMode.EXACT, "trade_date"),
        }
        provenance = []
        for domain, (rows, params, mode, date_field) in rows_by_domain.items():
            for row in rows:
                evidence = TimedEvidence(
                    domain=domain,
                    source="tushare",
                    available_at=_parse_date(str(row[date_field]), date_field),
                    fetched_at=fetched_at,
                    payload=row,
                    query_params=params,
                    channel=EvidenceChannel.FEATURE,
                    pit_mode=mode,
                )
                evidence.assert_allowed(context)
                provenance.append(evidence)

        missing_optional = tuple(
            name for name, rows in (("daily_basic", daily_basic), ("moneyflow", moneyflow)) if not rows
        )
        overall_mode = financial_mode
        snapshot_payload = {
            "schema_version": PIT_SCHEMA_VERSION,
            "stock_code": ts_code,
            # Excluding data_snapshot_id avoids a circular hash.  Replay binds
            # the verified content digest back into the context.
            "context": context.to_dict(include_snapshot=False),
            "pit_mode": overall_mode.value,
            "normalized_inputs": {
                "daily": list(daily),
                "daily_basic": list(daily_basic),
                "fina_indicator": list(fina_indicator),
                "moneyflow": list(moneyflow),
            },
            "provenance": [item.to_dict() for item in provenance],
            "unsupported_domains": ["news", "event"],
            "missing_optional_fields": list(missing_optional),
        }
        snapshot_id = self.snapshot_store.put(snapshot_payload)
        bound_context = context.with_snapshot(snapshot_id)
        return PITStockDataBundle(
            stock_code=ts_code,
            context=bound_context,
            daily=daily,
            daily_basic=daily_basic,
            fina_indicator=fina_indicator,
            moneyflow=moneyflow,
            provenance=tuple(provenance),
            snapshot_id=snapshot_id,
            pit_mode=overall_mode,
            unsupported_domains=("news", "event"),
            missing_optional_fields=missing_optional,
        )

    def replay_stock_bundle(self, snapshot_id: str) -> PITStockDataBundle:
        payload = self.snapshot_store.get(snapshot_id)
        if payload.get("schema_version") != PIT_SCHEMA_VERSION:
            raise PITSnapshotCorruptError("unsupported PIT snapshot schema")
        context_data = payload.get("context")
        normalized = payload.get("normalized_inputs")
        if not isinstance(context_data, Mapping) or not isinstance(normalized, Mapping):
            raise PITSnapshotCorruptError("snapshot lacks context/normalized_inputs")
        context = AsOfContext(
            trade_date=context_data["trade_date"],
            knowledge_cutoff=context_data["knowledge_cutoff"],
            model_profile=context_data["model_profile"],
            prompt_version=context_data["prompt_version"],
            data_snapshot_id=snapshot_id,
            cache_namespace=context_data["cache_namespace"],
            schema_version=context_data["schema_version"],
        )
        provenance_raw = payload.get("provenance", [])
        provenance = tuple(
            TimedEvidence(
                domain=item["domain"],
                source=item["source"],
                available_at=item["available_at"],
                fetched_at=item["fetched_at"],
                payload=item["payload"],
                query_params=item.get("query_params", {}),
                channel=EvidenceChannel(item["channel"]),
                pit_mode=PITMode(item["pit_mode"]),
                row_hash=item["row_hash"],
            )
            for item in provenance_raw
        )
        return PITStockDataBundle(
            stock_code=str(payload["stock_code"]),
            context=context,
            daily=tuple(normalized.get("daily", [])),
            daily_basic=tuple(normalized.get("daily_basic", [])),
            fina_indicator=tuple(normalized.get("fina_indicator", [])),
            moneyflow=tuple(normalized.get("moneyflow", [])),
            provenance=provenance,
            snapshot_id=snapshot_id,
            pit_mode=PITMode(payload["pit_mode"]),
            unsupported_domains=tuple(payload.get("unsupported_domains", ())),
            missing_optional_fields=tuple(payload.get("missing_optional_fields", ())),
        )


__all__ = [
    "AsOfContext",
    "ContentAddressedSnapshotStore",
    "EvidenceChannel",
    "FeatureLabelBoundary",
    "PITBoundaryError",
    "PITDataGateway",
    "PITDataUnavailableError",
    "PITMode",
    "PIT_SCHEMA_VERSION",
    "PITSnapshotCorruptError",
    "PITStockDataBundle",
    "PITTemporalViolation",
    "TimedEvidence",
    "select_fina_indicator_rows",
]
