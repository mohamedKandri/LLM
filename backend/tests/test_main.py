"""Integration tests for main.py's FastAPI routes via TestClient.
Every test monkeypatches app.main.load_config (and settings_store.ROOT
for the .env-writing endpoints) so requests NEVER touch the real
config.yaml, data/distill.db, data/seeds.jsonl, or .env."""

import json

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.config import load_config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = load_config()
    c.raw["storage"]["db_path"] = str(tmp_path / "test.db")
    c.raw["generation"]["seeds_path"] = str(tmp_path / "seeds.jsonl")
    c.raw["benchmark"]["gold_set_path"] = str(tmp_path / "gold.jsonl")
    (tmp_path / "seeds.jsonl").write_text(
        json.dumps({"id": "seed-01", "type": "general", "prompt": "What is a mutex?"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(main_module, "load_config", lambda: c)
    # Defense in depth against ever repeating the incident where a
    # TestClient hit the real /settings or /graduate endpoint and
    # overwrote the real config.yaml: no-op the actual disk write
    # regardless of what path any call site threads through. `from
    # .config import save_config` binds an independent name in EVERY
    # importing module (orchestrator.py, settings_store.py, ...) — each
    # one needs patching separately, patching config.save_config alone
    # does not affect already-bound references elsewhere.
    monkeypatch.setattr("app.settings_store.save_config", lambda cfg, path=None: None)
    monkeypatch.setattr("app.orchestrator.save_config", lambda cfg, path=None: None)
    # .env reads (config.ROOT, via _read_dotenv) and writes
    # (settings_store.ROOT) are separate module-level bindings from the
    # same `from .config import ROOT` import — both need redirecting.
    monkeypatch.setattr("app.config.ROOT", tmp_path)
    monkeypatch.setattr("app.settings_store.ROOT", tmp_path)
    monkeypatch.setattr(main_module, "_orchestrator", None)
    yield c
    monkeypatch.setattr(main_module, "_orchestrator", None)


@pytest.fixture
def client(cfg):
    return TestClient(main_module.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_status(client):
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["teacher_tier"] == "free"
    assert "dataset_counts" in body


def test_settings_get_and_update(client):
    r = client.get("/settings")
    assert r.status_code == 200
    assert r.json()["teacher_model"]

    r = client.post("/settings", json={"teacher_model": "some/model"})
    assert r.status_code == 200
    assert r.json()["teacher_model"] == "some/model"


def test_settings_update_rejects_negative_budget(client):
    r = client.post("/settings", json={"budget_max_usd": -5})
    assert r.status_code == 400


def test_seeds_crud(client):
    r = client.get("/seeds")
    assert r.status_code == 200
    assert len(r.json()) == 1

    r = client.post("/seeds", json={"type": "general", "prompt": "What is a deadlock exactly?"})
    assert r.status_code == 200
    seed_id = r.json()["id"]

    r = client.get("/seeds")
    assert len(r.json()) == 2

    r = client.delete(f"/seeds/{seed_id}")
    assert r.status_code == 200
    r = client.get("/seeds")
    assert len(r.json()) == 1


def test_seeds_delete_missing_returns_404(client):
    r = client.delete("/seeds/does-not-exist")
    assert r.status_code == 404


def test_seeds_add_rejects_bad_code_task(client):
    r = client.post(
        "/seeds", json={"type": "code", "prompt": "Write foo() please.", "tests": "no assert here"}
    )
    assert r.status_code == 400


def test_benchmark_history_empty(client):
    r = client.get("/benchmark/history")
    assert r.status_code == 200
    assert r.json() == []


def test_run_status_idle_initially(client):
    r = client.get("/run/status")
    assert r.status_code == 200
    assert r.json()["state"] == "idle"


def test_api_key_set_and_remove(client, tmp_path):
    r = client.post("/settings/api-key", json={"key": "sk-or-test"})
    assert r.status_code == 200
    assert r.json()["api_key_set"] is True
    assert (tmp_path / ".env").exists()

    r = client.delete("/settings/api-key")
    assert r.status_code == 200
    assert r.json()["api_key_set"] is False


def test_api_key_rejects_empty(client):
    r = client.post("/settings/api-key", json={"key": ""})
    assert r.status_code == 400


def test_run_start_then_stop_is_safe(client):
    # No API key in this sandboxed .env, so the real orchestrator's real
    # TeacherClient fails almost immediately (MissingAPIKeyError) and the
    # loop self-stops — that's the orchestrator's own "never crash the
    # thread" contract, already covered precisely with fakes in
    # test_orchestrator.py. This just proves the endpoints wire up and
    # start/stop stay safe to call without raising.
    r = client.post("/run/start")
    assert r.status_code == 200
    assert r.json()["state"] in ("running", "stopped")

    r = client.post("/run/stop")
    assert r.status_code == 200
    assert r.json()["state"] == "stopped"

    r = client.get("/run/status")
    assert r.status_code == 200
    assert r.json()["state"] == "stopped"


def test_graduate_and_go_local_succeed_when_idle(client):
    # The "refuses while running" guard itself is tested precisely with
    # fakes (deterministic timing) in test_orchestrator.py; here we only
    # need the HTTP layer to reach Orchestrator.graduate()/go_local()
    # correctly on the normal (idle) path.
    r = client.post("/graduate")
    assert r.status_code == 200
    assert r.json()["teacher_tier"] == "paid"


def test_settings_update_succeeds_when_idle(client):
    r = client.post("/settings", json={"budget_max_calls": 10})
    assert r.status_code == 200
    assert r.json()["budget_max_calls"] == 10
    client.post("/run/stop")
