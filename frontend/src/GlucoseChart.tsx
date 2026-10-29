import type { GlucoseResult } from "./types";

const W = 620, H = 220, PAD = { l: 44, r: 16, t: 14, b: 28 };

/** History (solid), 30/60-minute forecast (dashed) and the 70-180 mg/dL target band. */
export default function GlucoseChart({ g }: { g: GlucoseResult }) {
  const n = g.history.length;
  const last = n - 1;
  const fx = [last + 6, last + 12];
  const fy = [g.forecast["30"], g.forecast["60"]];
  const all = [...g.history, ...fy, 70, 180];
  const lo = Math.min(40, ...all) - 5, hi = Math.max(200, ...all) + 5;
  const x = (i: number) => PAD.l + (i / (last + 12)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);
  const hist = g.history.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const fc = [`${x(last)},${y(g.history[last])}`, ...fx.map((f, i) => `${x(f)},${y(fy[i])}`)].join(" ");
  const ticks = [50, 70, 100, 140, 180, 250, 300].filter((t) => t > lo && t < hi);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Glucose history with 30 and 60 minute forecast" className="chart">
      <rect x={PAD.l} y={y(180)} width={W - PAD.l - PAD.r} height={y(70) - y(180)} className="band" />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} className="grid" />
          <text x={PAD.l - 6} y={y(t) + 4} textAnchor="end" className="tick">{t}</text>
        </g>
      ))}
      <line x1={x(last)} x2={x(last)} y1={PAD.t} y2={H - PAD.b} className="now" />
      <polyline points={hist} className="hist" />
      <polyline points={fc} className="fcast" />
      {fx.map((f, i) => (
        <g key={f}>
          <circle cx={x(f)} cy={y(fy[i])} r={4.5} className="dot" />
          <text x={x(f)} y={y(fy[i]) - 9} textAnchor="middle" className="val">{Math.round(fy[i])}</text>
        </g>
      ))}
      <text x={x(0)} y={H - 8} className="tick">-{last * 5} min</text>
      <text x={x(last)} y={H - 8} textAnchor="middle" className="tick">now</text>
      <text x={x(last + 6)} y={H - 8} textAnchor="middle" className="tick">+30</text>
      <text x={x(last + 12)} y={H - 8} textAnchor="end" className="tick">+60 min</text>
    </svg>
  );
}
