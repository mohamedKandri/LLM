import { useMemo, useState } from "react";
import type { BenchmarkRun } from "../types";

const WIDTH = 640;
const HEIGHT = 220;
const PAD = { top: 16, right: 16, bottom: 28, left: 34 };

const SERIES_COLOR: Record<string, string> = {
  free: "var(--series-1)",
  paid: "var(--series-2)",
};

interface Point {
  x: number;
  y: number;
  run: BenchmarkRun;
}

export function BenchmarkChart({ runs }: { runs: BenchmarkRun[] }) {
  const [hover, setHover] = useState<Point | null>(null);

  const { points, target, plotH } = useMemo(() => {
    const plotW = WIDTH - PAD.left - PAD.right;
    const plotH = HEIGHT - PAD.top - PAD.bottom;
    if (runs.length === 0) return { points: [] as Point[], target: null as number | null, plotW, plotH };

    const n = runs.length;
    const xFor = (i: number) => PAD.left + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
    const yFor = (ratio: number) => PAD.top + (1 - Math.min(ratio, 1.15)) * plotH;

    const points = runs.map((run, i) => ({ x: xFor(i), y: yFor(run.ratio), run }));
    const target = runs.length > 0 ? yFor(runs[runs.length - 1].target) : null;
    return { points, target, plotW, plotH };
  }, [runs]);

  if (runs.length === 0) {
    return (
      <div
        style={{
          height: HEIGHT,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--text-muted)",
          fontSize: 13,
        }}
      >
        No benchmark runs yet — these appear once a fine-tuned checkpoint has been
        scored against the gold set.
      </div>
    );
  }

  const seriesTiers = Array.from(new Set(runs.map((r) => r.teacher_tier)));
  const bySeries = seriesTiers.map((tier) => ({
    tier,
    pts: points.filter((p) => p.run.teacher_tier === tier),
  }));

  return (
    <div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width="100%"
        role="img"
        aria-label="Student-vs-teacher benchmark ratio over time"
      >
        {/* gridlines */}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const y = PAD.top + (1 - frac) * plotH;
          return (
            <line
              key={frac}
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={y}
              y2={y}
              stroke="var(--gridline)"
              strokeWidth={1}
            />
          );
        })}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => (
          <text
            key={frac}
            x={PAD.left - 8}
            y={PAD.top + (1 - frac) * plotH + 3}
            textAnchor="end"
            fontSize={10}
            fill="var(--text-muted)"
          >
            {Math.round(frac * 100)}%
          </text>
        ))}

        {/* target reference line */}
        {target !== null && (
          <line
            x1={PAD.left}
            x2={WIDTH - PAD.right}
            y1={target}
            y2={target}
            stroke="var(--text-muted)"
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
        )}

        {/* series lines */}
        {bySeries.map(({ tier, pts }) =>
          pts.length > 0 ? (
            <polyline
              key={tier}
              points={pts.map((p) => `${p.x},${p.y}`).join(" ")}
              fill="none"
              stroke={SERIES_COLOR[tier] ?? "var(--series-1)"}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ) : null,
        )}

        {/* markers + hover targets */}
        {points.map((p, i) => (
          <g key={i}>
            <circle
              cx={p.x}
              cy={p.y}
              r={4}
              fill="var(--surface-1)"
              stroke={SERIES_COLOR[p.run.teacher_tier] ?? "var(--series-1)"}
              strokeWidth={2}
            />
            <circle
              cx={p.x}
              cy={p.y}
              r={10}
              fill="transparent"
              onMouseEnter={() => setHover(p)}
              onMouseLeave={() => setHover(null)}
              style={{ cursor: "pointer" }}
            />
          </g>
        ))}

        {hover && (
          <g transform={`translate(${Math.min(hover.x + 8, WIDTH - 150)}, ${Math.max(hover.y - 34, 4)})`}>
            <rect width={140} height={40} rx={6} fill="var(--surface-1)" stroke="var(--border)" />
            <text x={8} y={16} fontSize={11} fill="var(--text-primary)" fontWeight={600}>
              {(hover.run.ratio * 100).toFixed(0)}% of {hover.run.teacher_tier} teacher
            </text>
            <text x={8} y={30} fontSize={10} fill="var(--text-muted)">
              {new Date(hover.run.created_at + "Z").toLocaleString()}
            </text>
          </g>
        )}
      </svg>

      <div className="legend">
        {seriesTiers.map((tier) => (
          <div className="legend-item" key={tier}>
            <span className="legend-swatch" style={{ background: SERIES_COLOR[tier] }} />
            <span>vs {tier} teacher</span>
          </div>
        ))}
        <div className="legend-item">
          <span
            className="legend-swatch"
            style={{ background: "var(--text-muted)", opacity: 0.6 }}
          />
          <span>graduation target</span>
        </div>
      </div>
    </div>
  );
}
