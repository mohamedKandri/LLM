import type { BenchmarkRun, RunStatus, Seed, Settings, StatusResponse, Tier } from "./types";

const BASE_URL = "http://127.0.0.1:8791";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(0, "Can't reach the Distill backend — is it running on :8791?");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, body?.detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  status: () => request<StatusResponse>("/status"),

  runStatus: () => request<RunStatus>("/run/status"),
  runStart: () => request<RunStatus>("/run/start", { method: "POST" }),
  runPause: () => request<RunStatus>("/run/pause", { method: "POST" }),
  runResume: () => request<RunStatus>("/run/resume", { method: "POST" }),
  runStop: () => request<RunStatus>("/run/stop", { method: "POST" }),

  graduate: () => request<RunStatus>("/graduate", { method: "POST" }),
  goLocal: () => request<RunStatus>("/go-local", { method: "POST" }),

  getSettings: () => request<Settings>("/settings"),
  updateSettings: (body: {
    teacher_model?: string;
    budget_max_usd?: number;
    budget_max_calls?: number;
  }) => request<Settings>("/settings", { method: "POST", body: JSON.stringify(body) }),
  setApiKey: (key: string) =>
    request<Settings>("/settings/api-key", { method: "POST", body: JSON.stringify({ key }) }),
  removeApiKey: () => request<Settings>("/settings/api-key", { method: "DELETE" }),

  listSeeds: () => request<Seed[]>("/seeds"),
  addSeed: (seed: { type: string; prompt: string; tests?: string }) =>
    request<Seed>("/seeds", { method: "POST", body: JSON.stringify(seed) }),
  deleteSeed: (id: string) => request<{ deleted: string }>(`/seeds/${encodeURIComponent(id)}`, { method: "DELETE" }),

  benchmarkHistory: (tier?: Tier) =>
    request<BenchmarkRun[]>(`/benchmark/history${tier ? `?tier=${tier}` : ""}`),
};
