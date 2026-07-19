"""Durable repository for the published refined-stock pools.

The legacy JSON shape (top-level ``short``/``medium``/``long`` keys) is kept on
purpose.  Repository metadata is stored in a reserved ``_repository`` key, so
older readers continue to work while new readers can distinguish a published
generation from an in-progress staging file.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional


SCHEMA_VERSION = 2
TERMS = ("short", "medium", "long")
STAGING_ACTIVE_SECONDS = 24 * 60 * 60


class RefinedPoolError(RuntimeError):
    """Base error raised by the refined-pool repository."""


class RefinedPoolConflictError(RefinedPoolError):
    """The caller tried to publish on top of a different generation."""


class RefinedPoolValidationError(RefinedPoolError):
    """A staging generation does not satisfy the refined-pool schema."""


class RefinedPoolLockTimeout(RefinedPoolError):
    """The cross-process repository lock could not be acquired in time."""


_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: str) -> threading.RLock:
    absolute = os.path.abspath(path)
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(absolute, threading.RLock())


class _FileMutex:
    """Small stdlib-only advisory file lock for Windows and POSIX."""

    def __init__(self, path: str, timeout: float) -> None:
        self.path = path
        self.timeout = timeout
        self._file = None
        self._thread_lock = _thread_lock(path)

    def __enter__(self) -> "_FileMutex":
        if not self._thread_lock.acquire(timeout=self.timeout):
            raise RefinedPoolLockTimeout(f"timed out waiting for {self.path}")
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._file = open(self.path, "a+b")
            if self._file.seek(0, os.SEEK_END) == 0:
                self._file.write(b"\0")
                self._file.flush()
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    self._lock_os_file()
                    return self
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        raise RefinedPoolLockTimeout(
                            f"timed out waiting for {self.path}"
                        )
                    time.sleep(0.05)
        except Exception:
            if self._file is not None:
                self._file.close()
                self._file = None
            self._thread_lock.release()
            raise

    def _lock_os_file(self) -> None:
        assert self._file is not None
        self._file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._file is not None:
                self._file.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None
            self._thread_lock.release()


def _empty_document() -> Dict[str, Any]:
    return {
        "short": {"stocks": [], "updated_at": "", "version": 0},
        "medium": {"stocks": [], "updated_at": "", "version": 0},
        "long": {"stocks": [], "updated_at": "", "version": 0},
        "blacklist": {"short": [], "medium": [], "long": []},
        "pool_health_history": {"short": [], "medium": [], "long": []},
    }


class RefinedPoolRepository:
    """Versioned, CAS-protected repository with invisible staging generations."""

    def __init__(self, path: str, lock_timeout: float = 10.0) -> None:
        self.path = os.path.abspath(path)
        self.lock_path = self.path + ".lock"
        self.staging_dir = self.path + ".staging"
        self.lock_timeout = lock_timeout

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with _FileMutex(self.lock_path, self.lock_timeout):
            yield

    @staticmethod
    def generation(document: Dict[str, Any]) -> int:
        metadata = document.get("_repository", {})
        try:
            return int(metadata.get("generation", 0))
        except (TypeError, ValueError):
            return 0

    def read(self) -> Dict[str, Any]:
        """Read only the last atomically published generation.

        Legacy documents remain readable during migration.  This is
        intentionally weaker than the validation used for any new publish.
        """
        if not os.path.isfile(self.path):
            return _empty_document()
        try:
            with open(self.path, "r", encoding="utf-8-sig") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RefinedPoolError(f"cannot read published refined pool: {exc}") from exc
        self.validate_read(value)
        return value

    def status(self) -> Dict[str, Any]:
        """Return publication state; staging files never become current here."""
        exists = os.path.isfile(self.path)
        document = self.read()
        stock_count = sum(
            len(document.get(term, {}).get("stocks", [])) for term in TERMS
        )
        active_staging, stale_staging = self._staging_counts(
            self.generation(document)
        )
        metadata = document.get("_repository", {})
        current_status = "current" if exists and stock_count else "empty"
        if active_staging:
            current_status = "updating" if exists and stock_count else "staging"
        return {
            "status": current_status,
            "published_status": "current" if exists and stock_count else "empty",
            "generation": self.generation(document),
            "schema_version": metadata.get("schema_version", SCHEMA_VERSION),
            "published_at": metadata.get("published_at", ""),
            "contract_status": metadata.get("contract_status", "legacy" if exists else "empty"),
            "legacy_terms": metadata.get("legacy_terms", []),
            "stock_count": stock_count,
            "staging": active_staging > 0,
            "staging_count": active_staging,
            "stale_staging": stale_staging > 0,
            "stale_staging_count": stale_staging,
        }

    def _staging_counts(self, current_generation: int) -> tuple[int, int]:
        """Classify active checkpoints separately from abandoned generations."""
        if not os.path.isdir(self.staging_dir):
            return (0, 0)
        active = 0
        stale = 0
        now = datetime.now(timezone.utc)
        try:
            names = os.listdir(self.staging_dir)
        except OSError:
            return (0, 0)
        for name in names:
            if not name.endswith(".json"):
                continue
            try:
                with open(
                    os.path.join(self.staging_dir, name), "r", encoding="utf-8"
                ) as handle:
                    staged = json.load(handle)
                created = datetime.fromisoformat(
                    str(staged.get("created_at", "")).replace("Z", "+00:00")
                )
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                is_current_base = int(staged.get("base_generation", -1)) == current_generation
                is_recent = 0 <= (now - created).total_seconds() <= STAGING_ACTIVE_SECONDS
                is_staged = staged.get("status") == "staged"
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                is_current_base = is_recent = is_staged = False
            if is_current_base and is_recent and is_staged:
                active += 1
            else:
                stale += 1
        return active, stale

    @staticmethod
    def _require_iso_timestamp(value: Any, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise RefinedPoolValidationError(f"{field_name} must be an ISO timestamp")
        try:
            datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise RefinedPoolValidationError(
                f"{field_name} must be an ISO timestamp"
            ) from exc

    def validate_read(self, document: Dict[str, Any]) -> None:
        """Compatibility validation for an already-published legacy file."""
        if not isinstance(document, dict):
            raise RefinedPoolValidationError("refined pool must be a JSON object")
        for term in TERMS:
            pool = document.get(term)
            if pool is None:
                continue  # legacy partial files are readable and normalized by clients
            if not isinstance(pool, dict):
                raise RefinedPoolValidationError(f"{term} pool must be an object")
            stocks = pool.get("stocks", [])
            if not isinstance(stocks, (list, dict)):
                raise RefinedPoolValidationError(
                    f"{term}.stocks must be a list (or legacy mapping)"
                )

        metadata = document.get("_repository")
        if metadata is not None and not isinstance(metadata, dict):
            raise RefinedPoolValidationError("_repository must be an object")

    def validate_publish(self, document: Dict[str, Any]) -> None:
        """Strict schema gate for a newly published generation."""
        if not isinstance(document, dict):
            raise RefinedPoolValidationError("refined pool must be a JSON object")
        for term in TERMS:
            if term not in document:
                raise RefinedPoolValidationError(
                    "new publication must contain short, medium and long pools"
                )
            pool = document[term]
            if not isinstance(pool, dict):
                raise RefinedPoolValidationError(f"{term} pool must be an object")
            stocks = pool.get("stocks")
            if not isinstance(stocks, list):
                raise RefinedPoolValidationError(
                    f"{term}.stocks must be a list in a new publication"
                )
            version = pool.get("version")
            if isinstance(version, bool) or not isinstance(version, int) or version < 0:
                raise RefinedPoolValidationError(
                    f"{term}.version must be a non-negative integer"
                )
            updated_at = pool.get("updated_at")
            if not isinstance(updated_at, str):
                raise RefinedPoolValidationError(f"{term}.updated_at must be a string")
            if stocks:
                if version < 1:
                    raise RefinedPoolValidationError(
                        f"non-empty {term} pool requires version >= 1"
                    )
                self._require_iso_timestamp(updated_at, f"{term}.updated_at")

            seen_codes = set()
            for index, item in enumerate(stocks):
                prefix = f"{term}.stocks[{index}]"
                if not isinstance(item, dict):
                    raise RefinedPoolValidationError(f"{prefix} must be an object")
                code = item.get("code")
                if not isinstance(code, str) or not code.strip():
                    raise RefinedPoolValidationError(f"{prefix}.code is required")
                if code in seen_codes:
                    raise RefinedPoolValidationError(
                        f"{term}.stocks contains duplicate code {code}"
                    )
                seen_codes.add(code)
                if not isinstance(item.get("name"), str):
                    raise RefinedPoolValidationError(f"{prefix}.name must be a string")
                validity = item.get("validity")
                if validity not in {"valid", "legacy_non_actionable"}:
                    raise RefinedPoolValidationError(
                        f"{prefix}.validity must be valid or legacy_non_actionable"
                    )
                score = item.get("final_score")
                score_is_numeric = (
                    not isinstance(score, bool)
                    and isinstance(score, (int, float))
                    and math.isfinite(float(score))
                    and 0.0 <= float(score) <= 100.0
                )
                if validity == "valid" and not score_is_numeric:
                    raise RefinedPoolValidationError(
                        f"{prefix}.final_score must be between 0 and 100"
                    )
                if validity == "legacy_non_actionable" and not (
                    score is None or score_is_numeric
                ):
                    raise RefinedPoolValidationError(
                        f"{prefix}.final_score must be null or between 0 and 100"
                    )
                coverage = item.get("coverage")
                if (
                    isinstance(coverage, bool)
                    or not isinstance(coverage, (int, float))
                    or not math.isfinite(float(coverage))
                    or not 0.0 <= float(coverage) <= 1.0
                ):
                    raise RefinedPoolValidationError(
                        f"{prefix}.coverage must be between 0 and 1"
                    )
                if validity == "valid":
                    if float(coverage) <= 0.0:
                        raise RefinedPoolValidationError(
                            f"{prefix}.coverage must be in (0, 1] for valid scores"
                        )
                    if item.get("missing_core_fields") not in (None, []):
                        raise RefinedPoolValidationError(
                            f"{prefix} cannot have missing_core_fields"
                        )
                    self._require_iso_timestamp(
                        item.get("scored_at"), f"{prefix}.scored_at"
                    )
                else:
                    if float(coverage) != 0.0:
                        raise RefinedPoolValidationError(
                            f"{prefix}.coverage must be 0 for legacy_non_actionable"
                        )
                    missing = item.get("missing_core_fields")
                    if not isinstance(missing, list) or "score_contract" not in missing:
                        raise RefinedPoolValidationError(
                            f"{prefix} legacy item must declare missing score_contract"
                        )
                    if "scored_at" not in item:
                        raise RefinedPoolValidationError(
                            f"{prefix}.scored_at must be explicit"
                        )
                    if item["scored_at"] is not None:
                        self._require_iso_timestamp(
                            item["scored_at"], f"{prefix}.scored_at"
                        )

    @staticmethod
    def _normalize_unchanged_legacy_terms(
        candidate: Dict[str, Any], current: Dict[str, Any]
    ) -> list[str]:
        """Make unchanged legacy memberships explicit but never actionable.

        A single-term refresh must not be blocked by old memberships in the
        other two terms.  Only byte-equivalent/unchanged term payloads may be
        migrated this way; newly changed terms must already satisfy the strict
        valid score contract.
        """
        legacy_terms: list[str] = []
        for term in TERMS:
            pool = candidate.get(term)
            if not isinstance(pool, dict):
                continue
            current_pool = current.get(term)
            changed = pool != current_pool
            stocks = pool.get("stocks")
            if isinstance(stocks, dict) and not changed:
                normalized_stocks = []
                for mapped_code, raw in stocks.items():
                    info = deepcopy(raw) if isinstance(raw, dict) else {}
                    info.setdefault("code", mapped_code)
                    normalized_stocks.append(info)
                pool["stocks"] = stocks = normalized_stocks
            if not isinstance(stocks, list):
                continue
            for index, raw in enumerate(stocks):
                if not isinstance(raw, dict):
                    continue
                validity = raw.get("validity")
                if validity == "legacy_non_actionable":
                    if changed:
                        raise RefinedPoolValidationError(
                            f"changed {term} pool cannot publish legacy_non_actionable items"
                        )
                    legacy_terms.append(term)
                    continue
                if validity is not None:
                    continue
                if changed:
                    # Leave it untouched so strict validation reports the exact
                    # missing field instead of silently blessing new legacy data.
                    continue
                migrated = deepcopy(raw)
                migrated["code"] = str(
                    migrated.get("code") or migrated.get("stock_code") or ""
                )
                migrated["name"] = str(
                    migrated.get("name") or migrated.get("company_name") or ""
                )
                migrated["final_score"] = migrated.get(
                    "final_score", migrated.get("score")
                )
                migrated["validity"] = "legacy_non_actionable"
                migrated["coverage"] = 0.0
                migrated["missing_core_fields"] = ["score_contract"]
                migrated["scored_at"] = None
                stocks[index] = migrated
                legacy_terms.append(term)
        return sorted(set(legacy_terms))

    # Kept as the strict public validation API for callers that validate before
    # publishing.  Compatibility reads use validate_read explicitly.
    validate = validate_publish

    def stage(
        self,
        document: Dict[str, Any],
        expected_generation: Optional[int] = None,
    ) -> str:
        """Write and validate an unpublished generation, returning its token."""
        candidate = deepcopy(document)
        candidate.pop("_repository", None)
        current = self.read()
        legacy_terms = self._normalize_unchanged_legacy_terms(candidate, current)
        self.validate_publish(candidate)
        current_generation = self.generation(current)
        if expected_generation is None:
            expected_generation = current_generation
        token = uuid.uuid4().hex
        staged = {
            "schema_version": SCHEMA_VERSION,
            "status": "staged",
            "token": token,
            "base_generation": int(expected_generation),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "legacy_terms": legacy_terms,
            "document": candidate,
        }
        os.makedirs(self.staging_dir, exist_ok=True)
        self._atomic_write(os.path.join(self.staging_dir, token + ".json"), staged)
        return token

    def publish_staged(
        self, token: str, expected_generation: Optional[int] = None
    ) -> Dict[str, Any]:
        """CAS-publish a staging generation through one atomic pointer switch."""
        stage_path = os.path.join(self.staging_dir, token + ".json")
        with self._locked():
            try:
                with open(stage_path, "r", encoding="utf-8") as handle:
                    staged = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise RefinedPoolError(f"staging generation {token!r} unavailable") from exc
            if staged.get("token") != token or staged.get("status") != "staged":
                raise RefinedPoolValidationError("invalid staging generation metadata")
            current = self.read()
            actual = self.generation(current)
            base = int(staged.get("base_generation", -1))
            wanted = base if expected_generation is None else int(expected_generation)
            if base != wanted or actual != wanted:
                raise RefinedPoolConflictError(
                    f"generation conflict: expected {wanted}, current {actual}"
                )
            document = staged.get("document")
            self.validate_publish(document)
            published = deepcopy(document)
            published["_repository"] = {
                "schema_version": SCHEMA_VERSION,
                "generation": actual + 1,
                "published_at": datetime.now(timezone.utc).isoformat(),
                "publication_id": token,
                "status": "current",
                "contract_status": (
                    "mixed" if staged.get("legacy_terms") else "current"
                ),
                "legacy_terms": list(staged.get("legacy_terms") or []),
            }
            self._atomic_write(self.path, published)
            try:
                os.remove(stage_path)
            except OSError:
                pass
            return published

    def publish(
        self,
        document: Dict[str, Any],
        expected_generation: Optional[int] = None,
    ) -> Dict[str, Any]:
        token = self.stage(document, expected_generation=expected_generation)
        return self.publish_staged(token, expected_generation=expected_generation)

    @staticmethod
    def _atomic_write(path: str, value: Dict[str, Any]) -> None:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".refined_pool_", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            # The file has been fsynced before replace.  Persisting the
            # directory entry is best-effort because Windows does not support
            # opening directories with os.open; a failure here must not roll
            # back a switch that already completed successfully.
            try:
                RefinedPoolRepository._fsync_directory(directory)
            except OSError:
                pass
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    @staticmethod
    def _fsync_directory(directory: str) -> None:
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(directory, os.O_RDONLY)
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
