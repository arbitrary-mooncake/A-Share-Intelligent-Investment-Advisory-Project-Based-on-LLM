"""category 枚举校验 + signal_pack 缓存 schema 版本测试（4.3a / 4.9-1 兜底路径）。"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import cache_utils
from src.utils.analysis_package_builder import normalize_signal_pack
from src.utils.analysis_schema import SIGNAL_PACK_SCHEMA_VERSION, SIGNAL_CATEGORIES


def _pack_with_category(cat):
    return {
        "bias": "bullish",
        "confidence": 0.8,
        "signals": [
            {"factor": "业绩预增", "direction": 1, "strength": 80,
             "confidence": 0.9, "source_level": "official_like", "category": cat},
        ],
        "risk_flags": [],
        "missing_data": [],
    }


def test_valid_category_preserved():
    sp = normalize_signal_pack(_pack_with_category("catalyst_event"))
    assert sp["signals"][0]["category"] == "catalyst_event"


def test_invalid_category_falls_back_to_other():
    sp = normalize_signal_pack(_pack_with_category("不存在的类目"))
    assert sp["signals"][0]["category"] == "other"


def test_missing_category_stays_missing():
    pack = _pack_with_category("catalyst_event")
    del pack["signals"][0]["category"]
    sp = normalize_signal_pack(pack)
    assert "category" not in sp["signals"][0]


def test_all_enum_members_accepted():
    for cat in SIGNAL_CATEGORIES:
        sp = normalize_signal_pack(_pack_with_category(cat))
        assert sp["signals"][0]["category"] == cat


# ── 缓存版本（4.9-1：版本不符按 miss） ──────────────────

def test_write_stamps_schema_version(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_utils, "_get_active_cache_dir", lambda: str(tmp_path))
    cache_utils.write_signal_pack_cache(
        "fundamental_analysis", "sh.600000", "2026-07-20", {"signals": []}
    )
    path = tmp_path / "fundamental_analysis_signal_pack_sh_600000_2026-07-20.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["_schema_version"] == SIGNAL_PACK_SCHEMA_VERSION


def test_read_current_version_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_utils, "_get_active_cache_dir", lambda: str(tmp_path))
    cache_utils.write_signal_pack_cache(
        "fundamental_analysis", "sh.600000", "2026-07-20", {"signals": [], "marker": 1}
    )
    data = cache_utils.read_signal_pack_cache(
        "fundamental_analysis", "sh.600000", "2026-07-20"
    )
    assert data is not None and data.get("marker") == 1


def test_read_old_version_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_utils, "_get_active_cache_dir", lambda: str(tmp_path))
    path = tmp_path / "fundamental_analysis_signal_pack_sh_600000_2026-07-20.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"signals": [], "bias": "neutral"}, f)  # 旧格式：无 _schema_version
    data = cache_utils.read_signal_pack_cache(
        "fundamental_analysis", "sh.600000", "2026-07-20"
    )
    assert data is None
