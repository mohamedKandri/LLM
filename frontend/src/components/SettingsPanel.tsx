import { useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { Settings } from "../types";

export function SettingsPanel() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [modelInput, setModelInput] = useState("");
  const [maxUsd, setMaxUsd] = useState("");
  const [maxCalls, setMaxCalls] = useState("");
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function load() {
    try {
      const s = await api.getSettings();
      setSettings(s);
      setModelInput(s.teacher_model);
      setMaxUsd(String(s.budget_max_usd));
      setMaxCalls(String(s.budget_max_calls));
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load settings.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function saveModelAndBudget(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setNotice(null);
    setError(null);
    try {
      await api.updateSettings({
        teacher_model: modelInput,
        budget_max_usd: Number(maxUsd),
        budget_max_calls: Number(maxCalls),
      });
      setNotice("Saved.");
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to save settings.");
    } finally {
      setBusy(false);
    }
  }

  async function saveApiKey(e: React.FormEvent) {
    e.preventDefault();
    if (!apiKeyInput.trim()) return;
    setBusy(true);
    setNotice(null);
    setError(null);
    try {
      await api.setApiKey(apiKeyInput.trim());
      setApiKeyInput("");
      setNotice("API key saved.");
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to save the API key.");
    } finally {
      setBusy(false);
    }
  }

  async function removeApiKey() {
    if (!confirm("Remove the OpenRouter API key? The app will stop being able to call the teacher.")) return;
    setBusy(true);
    setNotice(null);
    setError(null);
    try {
      await api.removeApiKey();
      setNotice("API key removed.");
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to remove the API key.");
    } finally {
      setBusy(false);
    }
  }

  if (!settings) {
    return error ? <div className="message error">{error}</div> : <div className="message info">Loading…</div>;
  }

  return (
    <div>
      {error && <div className="message error">{error}</div>}
      {notice && <div className="message success">{notice}</div>}

      <div className="card">
        <h2>API key</h2>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 0 }}>
          Status:{" "}
          {settings.api_key_set ? (
            <span style={{ color: "var(--good)", fontWeight: 600 }}>set</span>
          ) : (
            <span style={{ color: "var(--critical)", fontWeight: 600 }}>not set</span>
          )}
          . Stored in a local <code>.env</code> file, never committed to git.
        </p>
        <form onSubmit={saveApiKey}>
          <div className="field">
            <label htmlFor="api-key">OpenRouter API key</label>
            <input
              id="api-key"
              type="password"
              placeholder="sk-or-v1-..."
              value={apiKeyInput}
              onChange={(e) => setApiKeyInput(e.target.value)}
              autoComplete="off"
            />
          </div>
          <div className="btn-row">
            <button className="btn primary" type="submit" disabled={busy || !apiKeyInput.trim()}>
              Save key
            </button>
            {settings.api_key_set && (
              <button className="btn danger" type="button" disabled={busy} onClick={removeApiKey}>
                Remove key
              </button>
            )}
          </div>
        </form>
      </div>

      <div className="card">
        <h2>Teacher &amp; budget ({settings.teacher_tier} tier)</h2>
        <form onSubmit={saveModelAndBudget}>
          <div className="field">
            <label htmlFor="model">Teacher model</label>
            <input
              id="model"
              value={modelInput}
              onChange={(e) => setModelInput(e.target.value)}
              placeholder="e.g. nvidia/nemotron-3-super-120b-a12b:free"
            />
            <span className="field-hint">OpenRouter model slug for the currently active ({settings.teacher_tier}) tier.</span>
          </div>
          <div className="card-row">
            <div className="field" style={{ flex: 1, minWidth: 160 }}>
              <label htmlFor="max-usd">Budget cap (USD)</label>
              <input
                id="max-usd"
                type="number"
                step="0.01"
                min="0"
                value={maxUsd}
                onChange={(e) => setMaxUsd(e.target.value)}
              />
            </div>
            <div className="field" style={{ flex: 1, minWidth: 160 }}>
              <label htmlFor="max-calls">Budget cap (calls)</label>
              <input
                id="max-calls"
                type="number"
                min="0"
                step="1"
                value={maxCalls}
                onChange={(e) => setMaxCalls(e.target.value)}
              />
            </div>
          </div>
          <div className="btn-row">
            <button className="btn primary" type="submit" disabled={busy}>
              Save
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
