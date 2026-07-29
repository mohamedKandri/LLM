"""settings_store: config read/write, API key .env write/remove, seed
CRUD. All writes redirected to tmp_path — never the real config.yaml,
.env, or data/seeds.jsonl."""

import json

import pytest

from app.config import load_config
from app import settings_store as ss


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.raw["storage"]["db_path"] = str(tmp_path / "test.db")
    return c


@pytest.fixture
def config_path(tmp_path):
    return tmp_path / "config.yaml"


@pytest.fixture
def env_path(tmp_path):
    return tmp_path / ".env"


@pytest.fixture
def seeds_path(tmp_path):
    p = tmp_path / "seeds.jsonl"
    p.write_text(
        json.dumps({"id": "seed-01", "type": "code", "prompt": "Write add(a, b).", "tests": "assert add(1,1)==2"})
        + "\n",
        encoding="utf-8",
    )
    return p


def test_get_settings_shape(cfg, monkeypatch):
    # Isolate from the developer's real .env, which has a real key set.
    monkeypatch.delenv(cfg.openrouter["api_key_env"], raising=False)
    monkeypatch.setattr("app.config._read_dotenv", lambda: {})
    s = ss.get_settings(cfg)
    assert s["teacher_tier"] == "free"
    assert s["teacher_model"] == cfg.teacher.model
    assert s["api_key_set"] is False


def test_update_settings_changes_active_tier_only(cfg, config_path):
    updated = ss.update_settings(cfg, teacher_model="some/new-model", config_path=config_path)
    assert updated["teacher_model"] == "some/new-model"
    assert cfg.raw["teachers"]["free"]["model"] == "some/new-model"
    assert cfg.raw["teachers"]["paid"]["model"] != "some/new-model"
    assert config_path.exists()


def test_update_settings_budget(cfg, config_path):
    updated = ss.update_settings(cfg, budget_max_usd=5.0, budget_max_calls=100, config_path=config_path)
    assert updated["budget_max_usd"] == 5.0
    assert updated["budget_max_calls"] == 100


def test_update_settings_rejects_empty_model(cfg, config_path):
    with pytest.raises(ValueError, match="empty"):
        ss.update_settings(cfg, teacher_model="   ", config_path=config_path)


def test_update_settings_rejects_negative_budget(cfg, config_path):
    with pytest.raises(ValueError, match="negative"):
        ss.update_settings(cfg, budget_max_usd=-1, config_path=config_path)


def test_set_and_remove_api_key(env_path):
    ss.set_api_key("sk-or-test-123", env_path=env_path)
    content = env_path.read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY=sk-or-test-123" in content

    ss.remove_api_key(env_path=env_path)
    content = env_path.read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY" not in content


def test_set_api_key_preserves_other_lines(env_path):
    env_path.write_text("OTHER_VAR=keep-me\n", encoding="utf-8")
    ss.set_api_key("sk-or-test-123", env_path=env_path)
    content = env_path.read_text(encoding="utf-8")
    assert "OTHER_VAR=keep-me" in content
    assert "OPENROUTER_API_KEY=sk-or-test-123" in content


def test_set_api_key_replaces_existing(env_path):
    ss.set_api_key("sk-old", env_path=env_path)
    ss.set_api_key("sk-new", env_path=env_path)
    content = env_path.read_text(encoding="utf-8")
    assert "sk-old" not in content
    assert "OPENROUTER_API_KEY=sk-new" in content


def test_set_api_key_rejects_empty(env_path):
    with pytest.raises(ValueError, match="empty"):
        ss.set_api_key("", env_path=env_path)


def test_remove_api_key_on_missing_file_is_noop(tmp_path):
    ss.remove_api_key(env_path=tmp_path / "does_not_exist.env")  # must not raise


def test_list_seeds(cfg, seeds_path):
    seeds = ss.list_seeds(cfg, seeds_path=seeds_path)
    assert len(seeds) == 1
    assert seeds[0]["id"] == "seed-01"


def test_add_seed_code(cfg, seeds_path):
    entry = ss.add_seed(cfg, "code", "Write is_even(n).", "assert is_even(2) == True", seeds_path=seeds_path)
    assert entry["id"] == "seed-02"
    seeds = ss.list_seeds(cfg, seeds_path=seeds_path)
    assert len(seeds) == 2


def test_add_seed_general(cfg, seeds_path):
    entry = ss.add_seed(cfg, "general", "What is a mutex?", seeds_path=seeds_path)
    assert entry["type"] == "general"
    assert "tests" not in entry


def test_add_seed_code_requires_valid_tests(cfg, seeds_path):
    with pytest.raises(ValueError, match="assert"):
        ss.add_seed(cfg, "code", "Write foo().", "not a real test", seeds_path=seeds_path)
    with pytest.raises(SyntaxError):
        ss.add_seed(cfg, "code", "Write foo().", "assert foo(", seeds_path=seeds_path)


def test_add_seed_rejects_bad_type(cfg, seeds_path):
    with pytest.raises(ValueError, match="type"):
        ss.add_seed(cfg, "weird", "Some prompt here.", seeds_path=seeds_path)


def test_add_seed_rejects_short_prompt(cfg, seeds_path):
    with pytest.raises(ValueError, match="10 characters"):
        ss.add_seed(cfg, "general", "short", seeds_path=seeds_path)


def test_delete_seed(cfg, seeds_path):
    assert ss.delete_seed(cfg, "seed-01", seeds_path=seeds_path) is True
    assert ss.list_seeds(cfg, seeds_path=seeds_path) == []


def test_delete_seed_missing_id_returns_false(cfg, seeds_path):
    assert ss.delete_seed(cfg, "seed-99", seeds_path=seeds_path) is False
    assert len(ss.list_seeds(cfg, seeds_path=seeds_path)) == 1
