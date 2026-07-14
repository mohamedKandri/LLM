"""Budget/rate-limit/config behavior — no network calls."""

import pytest

from app.config import load_config
from app.teacher_client import (
    BudgetExceededError,
    BudgetLedger,
    LocalOnlyModeError,
    MissingAPIKeyError,
    TeacherClient,
)


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.raw["storage"]["db_path"] = str(tmp_path / "test.db")
    # db_path property joins with ROOT; absolute paths pass through on join
    return c


def test_model_comes_from_config_tier(cfg):
    assert cfg.teacher.model == cfg.raw["teachers"][cfg.teacher_tier]["model"]
    cfg.raw["teacher_tier"] = "paid"
    assert cfg.teacher.model == cfg.raw["teachers"]["paid"]["model"]


def test_ledger_call_cap_enforced(tmp_path):
    ledger = BudgetLedger(tmp_path / "l.db")
    for _ in range(3):
        ledger.record("free", "teacher", "m", "generate", 10, 10, 0.0)

    class B:
        max_usd = 1.0
        max_calls = 3

    with pytest.raises(BudgetExceededError):
        ledger.check("free", "teacher", B)


def test_ledger_usd_cap_enforced(tmp_path):
    ledger = BudgetLedger(tmp_path / "l.db")
    ledger.record("paid", "teacher", "m", "generate", 10, 10, 5.01)

    class B:
        max_usd = 5.0
        max_calls = 100

    with pytest.raises(BudgetExceededError):
        ledger.check("paid", "teacher", B)


def test_ledger_totals_namespaced_by_tier_and_role(tmp_path):
    ledger = BudgetLedger(tmp_path / "l.db")
    ledger.record("free", "teacher", "m", "generate", 1, 1, 0.0)
    ledger.record("paid", "teacher", "m", "generate", 1, 1, 2.5)
    ledger.record("paid", "judge", "m", "judge", 1, 1, 0.5)
    assert ledger.totals("free", "teacher") == (0.0, 1)
    assert ledger.totals("paid", "teacher") == (2.5, 1)
    assert ledger.totals("paid", "judge") == (0.5, 1)


def test_local_only_blocks_api(cfg, monkeypatch):
    monkeypatch.setenv(cfg.openrouter["api_key_env"], "sk-test")
    cfg.raw["inference"]["local_only"] = True
    client = TeacherClient(cfg, cfg.teacher)
    with pytest.raises(LocalOnlyModeError):
        client.complete([{"role": "user", "content": "hi"}])


def test_missing_api_key_raises(cfg, monkeypatch):
    monkeypatch.delenv(cfg.openrouter["api_key_env"], raising=False)
    client = TeacherClient(cfg, cfg.teacher)
    with pytest.raises(MissingAPIKeyError):
        client.complete([{"role": "user", "content": "hi"}])
