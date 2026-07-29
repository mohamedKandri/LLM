export type Tier = "free" | "paid";

export interface StatusResponse {
  phase: number;
  teacher_tier: Tier;
  teacher_model: string;
  local_only: boolean;
  teacher_spend_usd: number;
  teacher_calls: number;
  judge_spend_usd: number;
  judge_calls: number;
  budget: { max_usd: number; max_calls: number };
  benchmark_target: number;
  dataset_counts: Record<string, number>;
}

export type RunState = "idle" | "running" | "paused" | "stopped";

export interface RunStatus {
  state: RunState;
  teacher_tier: Tier;
  iterations: number;
  tasks_generated: number;
  accepted: number;
  rejected: number;
  accepted_since_retrain: number;
  last_error: string | null;
  stop_reason: string | null;
  dataset_counts: Record<string, number>;
  teacher_spend_usd: number;
  teacher_calls: number;
  judge_spend_usd: number;
  judge_calls: number;
}

export interface Settings {
  teacher_tier: Tier;
  teacher_model: string;
  budget_max_usd: number;
  budget_max_calls: number;
  local_only: boolean;
  api_key_set: boolean;
}

export type SeedType = "code" | "general";

export interface Seed {
  id: string;
  type: SeedType;
  prompt: string;
  tests?: string;
}

export interface BenchmarkRun {
  id: number;
  teacher_tier: Tier;
  teacher_model: string;
  student_score: number;
  teacher_score: number;
  ratio: number;
  target: number;
  target_met: boolean;
  created_at: string;
}
