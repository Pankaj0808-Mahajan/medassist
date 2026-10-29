export type Level = "routine" | "review" | "emergency";
export type LabFlag = "normal" | "low" | "high" | "critical_low" | "critical_high";

export interface GlucoseResult {
  current: number;
  forecast: { "30": number; "60": number };
  rate_per_5min: number;
  history: number[];
  model: string;
}
export interface LabRow { name: string; value: number; unit: string; reference: string; flag: LabFlag }
export interface LabResult {
  results: LabRow[];
  abnormal: string[];
  risk: { diabetes_risk: number; model: string } | null;
}
export interface XrayResult {
  available: boolean;
  note?: string;
  finding?: "NORMAL" | "PNEUMONIA";
  p_pneumonia?: number;
  confidence?: number;
  heatmap?: string;
  model?: string;
}
export interface Safety { level: Level; reasons: string[]; tips: string[]; message: string | null }
export interface Source { id: string; source: string; text: string; score: number }
export interface AnalysisResult {
  glucose: GlucoseResult | null;
  labs: LabResult | null;
  xray: XrayResult | null;
  safety: Safety;
  sources: Source[];
  advice: string;
  advice_source: string;
  disclaimer: string;
}
