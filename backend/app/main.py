import json
import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import llm, safety
from .config import CORS_ORIGINS, GROQ_API_KEY, WEIGHTS
from .modules.glucose import GlucoseForecaster
from .modules.labs import LabAnalyzer
from .modules.xray import XrayClassifier
from .rag import Retriever

app = FastAPI(title="MedAssist API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

glucose_model, lab_model, xray_model, retriever = GlucoseForecaster(), LabAnalyzer(), XrayClassifier(), Retriever()


def retrieval_query(question: str, glucose, labs, xray) -> str:
    q = [question]
    if glucose:
        low = min(glucose["current"], *glucose["forecast"].values()) < safety.GLUCOSE_LOW
        q.append("low blood glucose hypoglycemia treatment" if low else "glucose targets time in range hyperglycemia")
    if labs:
        q += labs["abnormal"] + ["lab results reference ranges"]
    if xray and xray.get("available"):
        q.append("pneumonia chest x-ray when to see a doctor")
    return " ".join(q)


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_configured": bool(GROQ_API_KEY),
            "weights": {n: (WEIGHTS / n).exists() for n in ("glucose_gru.pt", "xray_densenet121.pt", "lab_risk_xgb.json")},
            "guideline_chunks": len(retriever.chunks)}


@app.post("/api/analyze")
def analyze(
    glucose: str = Form(""),                 # comma/space/newline separated mg/dL values, oldest first, 5-min spacing
    last_reading_time: str = Form(""),       # ISO time of the newest reading (default: now)
    age: Optional[int] = Form(None),
    question: str = Form(""),
    report: Optional[UploadFile] = File(None),
    xray: Optional[UploadFile] = File(None),
):
    values = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", glucose)]
    if not (values or report or xray):
        raise HTTPException(422, "Provide glucose readings, a blood report, a chest X-ray, or a combination.")
    try:
        g = l = x = None
        if values:
            when = datetime.fromisoformat(last_reading_time) if last_reading_time else None
            g = glucose_model.predict(values, when)
        if report is not None:
            l = lab_model.analyze_file(report.file.read(), report.filename or "report.png", age)
        if xray is not None:
            x = xray_model.predict(xray.file.read())
    except ValueError as e:
        raise HTTPException(422, str(e))

    s = safety.evaluate(g, l, x)
    passages = retriever.search(retrieval_query(question, g, l, x), k=3)

    if s["level"] == "emergency":  # the safety layer overrides the LLM entirely
        advice, advice_source = s["message"] + (" " + " ".join(s["tips"]) if s["tips"] else ""), "safety layer (LLM skipped)"
    else:
        advice, advice_source = llm.advise(llm.build_context(age, g, l, x, s, passages, question), g, l, x)
        if s["level"] == "review":
            advice = "A doctor should review these results soon. " + advice
            if s["tips"]:
                advice += " " + " ".join(s["tips"])

    return {"glucose": g, "labs": l, "xray": x, "safety": s, "sources": passages,
            "advice": advice, "advice_source": advice_source, "disclaimer": safety.DISCLAIMER}
