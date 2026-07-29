import { useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { Seed, SeedType } from "../types";

export function SeedEditor() {
  const [seeds, setSeeds] = useState<Seed[]>([]);
  const [type, setType] = useState<SeedType>("general");
  const [prompt, setPrompt] = useState("");
  const [tests, setTests] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function load() {
    try {
      setSeeds(await api.listSeeds());
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load seeds.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function addSeed(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setNotice(null);
    setError(null);
    try {
      await api.addSeed({ type, prompt, tests: type === "code" ? tests : undefined });
      setPrompt("");
      setTests("");
      setNotice("Seed added.");
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to add seed.");
    } finally {
      setBusy(false);
    }
  }

  async function removeSeed(id: string) {
    if (!confirm(`Delete seed ${id}?`)) return;
    setBusy(true);
    try {
      await api.deleteSeed(id);
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to delete seed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      {error && <div className="message error">{error}</div>}
      {notice && <div className="message success">{notice}</div>}

      <div className="card">
        <h2>Add seed task</h2>
        <form onSubmit={addSeed}>
          <div className="field">
            <label htmlFor="seed-type">Type</label>
            <select id="seed-type" value={type} onChange={(e) => setType(e.target.value as SeedType)}>
              <option value="general">general</option>
              <option value="code">code</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="seed-prompt">Prompt</label>
            <textarea
              id="seed-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Write a Python function that ..."
              required
              style={{ fontFamily: "inherit", minHeight: 50 }}
            />
          </div>
          {type === "code" && (
            <div className="field">
              <label htmlFor="seed-tests">Tests (Python asserts, run after the solution)</label>
              <textarea
                id="seed-tests"
                value={tests}
                onChange={(e) => setTests(e.target.value)}
                placeholder={"assert my_func(1, 2) == 3"}
                required
              />
            </div>
          )}
          <div className="btn-row">
            <button className="btn primary" type="submit" disabled={busy || !prompt.trim()}>
              Add seed
            </button>
          </div>
        </form>
      </div>

      <div className="card">
        <h2>Seeds ({seeds.length})</h2>
        <div className="seed-list">
          {seeds.map((s) => (
            <div className="seed-item" key={s.id}>
              <span className="seed-type">{s.type}</span>
              <div style={{ flex: 1 }}>
                <div className="seed-prompt">{s.prompt}</div>
                {s.tests && <div className="seed-tests">{s.tests}</div>}
              </div>
              <button
                className="icon-btn"
                title={`Delete ${s.id}`}
                disabled={busy}
                onClick={() => removeSeed(s.id)}
              >
                ✕
              </button>
            </div>
          ))}
          {seeds.length === 0 && (
            <div style={{ fontSize: 13, color: "var(--text-muted)" }}>No seeds yet.</div>
          )}
        </div>
      </div>
    </div>
  );
}
