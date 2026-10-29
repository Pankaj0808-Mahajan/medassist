import type { AnalysisResult } from "./types";

const API = import.meta.env.VITE_API_URL ?? ""; // empty = same origin / Vite proxy

export async function analyze(form: FormData): Promise<AnalysisResult> {
  const res = await fetch(`${API}/api/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail));
  }
  return res.json();
}
