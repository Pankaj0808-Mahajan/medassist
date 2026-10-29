import { FormEvent, useMemo, useState } from "react";
import { analyze } from "./api";
import GlucoseChart from "./GlucoseChart";
import type { AnalysisResult, LabFlag, Level } from "./types";

const SAMPLE = Array.from({ length: 30 }, (_, i) => Math.round(165 - i * 2.4 + 4 * Math.sin(i / 2))).join(", ");

const LEVEL_TEXT: Record<Level, string> = {
  routine: "No urgent findings",
  review: "Doctor review recommended",
  emergency: "Possible emergency - get medical help now",
};
const FLAG_TEXT: Record<LabFlag, string> = {
  normal: "Normal", low: "Low", high: "High", critical_low: "Critically low", critical_high: "Critically high",
};

export default function App() {
  const [glucose, setGlucose] = useState("");
  const [age, setAge] = useState("");
  const [question, setQuestion] = useState("");
  const [report, setReport] = useState<File | null>(null);
  const [xray, setXray] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const xrayUrl = useMemo(() => (xray ? URL.createObjectURL(xray) : ""), [xray]);
  const nReadings = (glucose.match(/\d+(\.\d+)?/g) ?? []).length;

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("glucose", glucose);
      fd.append("question", question);
      if (age) fd.append("age", age);
      if (report) fd.append("report", report);
      if (xray) fd.append("xray", xray);
      setResult(await analyze(fd));
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header>
        <h1>MedAssist</h1>
        <p>Add any combination of glucose readings, a blood report and a chest X-ray. The assistant explains them in plain language and flags anything that needs a doctor.</p>
      </header>

      <form onSubmit={onSubmit} className="panel">
        <label>
          Glucose readings (mg/dL, oldest first, one every 5 minutes, at least 24)
          <textarea rows={3} value={glucose} onChange={(e) => setGlucose(e.target.value)} placeholder="152, 150, 149, ..." />
        </label>
        <div className="row hint">
          <span>{nReadings} readings{nReadings > 0 && nReadings < 24 ? " - need 24" : ""}</span>
          <button type="button" className="link" onClick={() => setGlucose(SAMPLE)}>Fill with sample readings</button>
        </div>
        <div className="grid2">
          <label>
            Blood report (PDF, PNG or JPEG)
            <input type="file" accept=".pdf,image/png,image/jpeg" onChange={(e) => setReport(e.target.files?.[0] ?? null)} />
          </label>
          <label>
            Chest X-ray (PNG or JPEG)
            <input type="file" accept="image/png,image/jpeg" onChange={(e) => setXray(e.target.files?.[0] ?? null)} />
          </label>
          <label>
            Age (optional)
            <input type="number" min={1} max={120} value={age} onChange={(e) => setAge(e.target.value)} />
          </label>
          <label>
            Your question (optional)
            <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="What should I watch for tonight?" />
          </label>
        </div>
        <button type="submit" className="primary" disabled={busy || (!glucose.trim() && !report && !xray)}>
          {busy ? "Analysing..." : "Analyse"}
        </button>
        {error && <p role="alert" className="error">{error}</p>}
      </form>

      {result && <Results r={result} xrayUrl={xrayUrl} />}
    </main>
  );
}

function Results({ r, xrayUrl }: { r: AnalysisResult; xrayUrl: string }) {
  const { safety } = r;
  return (
    <section aria-live="polite">
      <div className={`banner ${safety.level}`} role={safety.level === "emergency" ? "alert" : "status"}>
        <strong>{LEVEL_TEXT[safety.level]}</strong>
        {safety.reasons.length > 0 && <ul>{safety.reasons.map((x) => <li key={x}>{x}</li>)}</ul>}
      </div>

      <div className="panel advice">
        <h2>Advice</h2>
        <p>{r.advice}</p>
        <p className="meta">Written by: {r.advice_source}</p>
      </div>

      {r.glucose && (
        <div className="panel">
          <h2>Glucose forecast</h2>
          <div className="stats">
            <div><b>{Math.round(r.glucose.current)}</b><span>now (mg/dL)</span></div>
            <div><b>{Math.round(r.glucose.forecast["30"])}</b><span>in 30 min</span></div>
            <div><b>{Math.round(r.glucose.forecast["60"])}</b><span>in 60 min</span></div>
          </div>
          <GlucoseChart g={r.glucose} />
          <p className="meta">Model: {r.glucose.model}. Shaded band = 70-180 mg/dL.</p>
        </div>
      )}

      {r.labs && (
        <div className="panel">
          <h2>Blood report</h2>
          <div className="scroll">
            <table>
              <thead><tr><th>Test</th><th>Result</th><th>Reference</th><th>Status</th></tr></thead>
              <tbody>
                {r.labs.results.map((x) => (
                  <tr key={x.name}>
                    <td>{x.name}</td>
                    <td>{x.value} {x.unit}</td>
                    <td>{x.reference}</td>
                    <td><span className={`flag ${x.flag}`}>{FLAG_TEXT[x.flag]}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {r.labs.risk && <p className="meta">Diabetes risk score {r.labs.risk.diabetes_risk.toFixed(2)} ({r.labs.risk.model}).</p>}
          <p className="meta">Values are read by OCR. Check them against the original report.</p>
        </div>
      )}

      {r.xray && (
        <div className="panel">
          <h2>Chest X-ray</h2>
          {r.xray.available ? (
            <>
              <p>
                Model finding: <b>{r.xray.finding === "PNEUMONIA" ? "possible pneumonia pattern" : "no pneumonia pattern"}</b> ({Math.round((r.xray.confidence ?? 0) * 100)}% confidence)
              </p>
              <div className="grid2 images">
                {xrayUrl && <figure><img src={xrayUrl} alt="Uploaded chest X-ray" /><figcaption>Your image</figcaption></figure>}
                {r.xray.heatmap && <figure><img src={r.xray.heatmap} alt="Grad-CAM heatmap" /><figcaption>Where the model looked (red = most influence)</figcaption></figure>}
              </div>
            </>
          ) : (
            <p className="meta">{r.xray.note}</p>
          )}
        </div>
      )}

      {r.sources.length > 0 && (
        <details className="panel">
          <summary>Guideline passages used ({r.sources.length})</summary>
          {r.sources.map((s) => (
            <p key={s.id} className="source"><b>[{s.id}] {s.source}</b> {s.text}</p>
          ))}
        </details>
      )}
      <p className="disclaimer">{r.disclaimer}</p>
    </section>
  );
}
