import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { BenchmarkRun, RunStatus, StatusResponse } from "../types";
import { BenchmarkChart } from "./BenchmarkChart";

const POLL_MS = 3000;

function fmtUsd(n: number) {
  return `$${n.toFixed(4)}`;
}

function BudgetBar({ spent, cap }: { spent: number; cap: number }) {
  const pct = cap > 0 ? Math.min((spent / cap) * 100, 100) : 0;
  const cls = pct >= 100 ? "critical" : pct >= 80 ? "warning" : "";
  return (
    <div className="budget-bar-track">
      <div className={`budget-bar-fill ${cls}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Dashboard() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [history, setHistory] = useState<BenchmarkRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [s, r, h] = await Promise.all([api.status(), api.runStatus(), api.benchmarkHistory()]);
      setStatus(s);
      setRunStatus(r);
      setHistory(h);
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to reach the backend.");
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  async function run<T>(action: () => Promise<T>, successMessage?: string) {
    setBusy(true);
    setNotice(null);
    try {
      await action();
      if (successMessage) setNotice(successMessage);
      await refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Action failed.");
    } finally {
      setBusy(false);
    }
  }

  if (error && !status) {
    return <div className="message error">{error}</div>;
  }
  if (!status || !runStatus) {
    return <div className="message info">Loading…</div>;
  }

  const latestForTier = [...history].reverse().find((r) => r.teacher_tier === status.teacher_tier);
  const canGraduate = status.teacher_tier === "free" && latestForTier?.target_met === true;
  const canGoLocal = status.teacher_tier === "paid" && latestForTier?.target_met === true;
  const isRunning = runStatus.state === "running";

  return (
    <div>
      {error && <div className="message error">{error}</div>}
      {notice && <div className="message success">{notice}</div>}
      {runStatus.last_error && (
        <div className="message error">Run stopped on an error: {runStatus.stop_reason}</div>
      )}

      <div className="card">
        <div className="card-row" style={{ justifyContent: "space-between", alignItems: "center" }}>
          <div className="btn-row" style={{ alignItems: "center" }}>
            <span className={`badge tier-${status.teacher_tier}`}>
              Phase {status.phase} · {status.teacher_tier} teacher
            </span>
            <span className={`badge state-${runStatus.state}`}>{runStatus.state}</span>
            {status.local_only && <span className="badge state-idle">local only</span>}
          </div>
          <div className="btn-row">
            {runStatus.state !== "running" ? (
              <button className="btn primary" disabled={busy} onClick={() => run(api.runStart)}>
                Start
              </button>
            ) : (
              <button className="btn" disabled={busy} onClick={() => run(api.runPause)}>
                Pause
              </button>
            )}
            {runStatus.state === "paused" && (
              <button className="btn" disabled={busy} onClick={() => run(api.runResume)}>
                Resume
              </button>
            )}
            <button
              className="btn"
              disabled={busy || runStatus.state === "stopped" || runStatus.state === "idle"}
              onClick={() => run(api.runStop)}
            >
              Stop
            </button>
          </div>
        </div>
        <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 8 }}>
          Model: {status.teacher_model}
        </div>
      </div>

      <div className="card">
        <h2>This run</h2>
        <div className="stat-grid">
          <div className="stat-tile">
            <div className="value">{runStatus.tasks_generated}</div>
            <div className="label">tasks generated</div>
          </div>
          <div className="stat-tile">
            <div className="value">{runStatus.accepted}</div>
            <div className="label">accepted</div>
          </div>
          <div className="stat-tile">
            <div className="value">{runStatus.rejected}</div>
            <div className="label">rejected</div>
          </div>
          <div className="stat-tile">
            <div className="value">{runStatus.iterations}</div>
            <div className="label">loop iterations</div>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Dataset &amp; spend</h2>
        <div className="card-row">
          <div style={{ flex: 1, minWidth: 220 }}>
            <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>
              Teacher spend — {fmtUsd(status.teacher_spend_usd)} / {fmtUsd(status.budget.max_usd)}
            </div>
            <BudgetBar spent={status.teacher_spend_usd} cap={status.budget.max_usd} />
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
              {status.teacher_calls} / {status.budget.max_calls} calls
            </div>
          </div>
          <div style={{ flex: 1, minWidth: 220 }}>
            <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>
              Judge spend — {fmtUsd(status.judge_spend_usd)}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{status.judge_calls} calls</div>
          </div>
        </div>
        <div className="stat-grid" style={{ marginTop: 16 }}>
          {Object.entries(status.dataset_counts).length === 0 && (
            <div className="stat-tile">
              <div className="value">0</div>
              <div className="label">accepted examples</div>
            </div>
          )}
          {Object.entries(status.dataset_counts).map(([tier, n]) => (
            <div className="stat-tile" key={tier}>
              <div className="value">{n}</div>
              <div className="label">accepted ({tier})</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h2>Benchmark vs teacher</h2>
        <BenchmarkChart runs={history} />
      </div>

      <div className="card">
        <h2>Graduation</h2>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 0 }}>
          Target for the {status.teacher_tier} tier: {(status.benchmark_target * 100).toFixed(0)}% of
          teacher score.{" "}
          {latestForTier
            ? `Latest run: ${(latestForTier.ratio * 100).toFixed(0)}%.`
            : "No benchmark run yet for this tier."}
        </p>
        <div className="btn-row">
          <button
            className="btn primary"
            disabled={busy || isRunning || !canGraduate}
            title={!canGraduate ? "Available once the phase-1 benchmark target is met" : undefined}
            onClick={() => run(api.graduate, "Graduated to phase 2 (paid teacher).")}
          >
            Graduate to phase 2
          </button>
          <button
            className="btn danger"
            disabled={busy || isRunning || !canGoLocal}
            title={!canGoLocal ? "Available once the phase-2 benchmark target is met" : undefined}
            onClick={() =>
              run(api.goLocal, "Local-only mode enabled — API calls are now disabled.")
            }
          >
            Go local
          </button>
        </div>
      </div>
    </div>
  );
}
