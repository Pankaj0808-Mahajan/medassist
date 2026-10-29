"""Advice agent: Groq-hosted LLM called through the OpenAI SDK, grounded in structured findings + retrieved passages."""
from typing import Optional

from openai import OpenAI

from .config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
from .safety import is_safe_advice

_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL) if GROQ_API_KEY else None

SYSTEM = (
    "You are a cautious health-information assistant that supports, and never replaces, a doctor. "
    "Use ONLY the structured findings and numbered sources provided. Never diagnose. Never name medicines, "
    "doses or insulin adjustments. Write in plain language (about 8th-grade level), 150-250 words, with these "
    "short sections: What the results show / What it may mean / What to do next / When to seek care. "
    "Cite guideline sources like [S1] when you use them. If the safety status is REVIEW, say a doctor should "
    "review the results soon. If information is missing, say so instead of guessing."
)


def build_context(age: Optional[int], glucose, labs, xray, safety: dict, passages: list, question: str) -> str:
    L = [f"PATIENT: age {age}" if age else "PATIENT: age not given"]
    if glucose:
        f = glucose["forecast"]
        L.append(f"GLUCOSE: now {glucose['current']:.0f} mg/dL; forecast {f['30']:.0f} mg/dL in 30 min and "
                 f"{f['60']:.0f} mg/dL in 60 min; last 5-min change {glucose['rate_per_5min']:+.0f} mg/dL.")
    if labs:
        bad = [r for r in labs["results"] if r["flag"] != "normal"]
        L.append("LABS (abnormal): " + ("; ".join(f"{r['name']} {r['value']} {r['unit']} ({r['flag']}, ref {r['reference']})" for r in bad) or "none")
                 + f". {len(labs['results']) - len(bad)} other values normal.")
        if labs.get("risk"):
            L.append(f"LAB RISK MODEL: diabetes risk score {labs['risk']['diabetes_risk']:.2f} (screening aid only).")
    if xray and xray.get("available"):
        L.append(f"CHEST X-RAY: model finding {xray['finding']} (pneumonia probability {xray['p_pneumonia']:.0%}, "
                 f"confidence {xray['confidence']:.0%}); heatmap highlights the regions the model used.")
    L.append(f"SAFETY STATUS: {safety['level'].upper()}" + (" - " + "; ".join(safety["reasons"]) if safety["reasons"] else ""))
    L.append("SOURCES:\n" + ("\n".join(f"[{p['id']}] ({p['source']}) {p['text']}" for p in passages) or "none"))
    L.append(f"USER QUESTION: {question or 'Please explain my results.'}")
    return "\n".join(L)


def template_advice(glucose, labs, xray) -> str:
    """Deterministic fallback used when no Groq key is set or the LLM output fails the safety check."""
    parts = []
    if glucose:
        f = glucose["forecast"]
        parts.append(f"Your glucose is {glucose['current']:.0f} mg/dL and is forecast to be about {f['30']:.0f} in 30 minutes "
                     f"and {f['60']:.0f} in 60 minutes. A common target range for many adults with diabetes is 70-180 mg/dL.")
    if labs:
        bad = [r for r in labs["results"] if r["flag"] != "normal"]
        parts.append("Outside the usual range: " + ", ".join(f"{r['name']} ({r['value']} {r['unit']})" for r in bad) + "."
                     if bad else "All the lab values we could read are within the usual ranges.")
    if xray and xray.get("available"):
        parts.append(f"The X-ray model's finding is {xray['finding'].lower()} ({xray['confidence']:.0%} confidence). "
                     "Only a doctor can interpret the image.")
    parts.append("Please share these results with your doctor and ask what they mean for you.")
    return " ".join(parts)


def advise(context: str, glucose, labs, xray) -> tuple:
    if _client is None:
        return template_advice(glucose, labs, xray), "template (set GROQ_API_KEY to enable the LLM)"
    try:
        r = _client.chat.completions.create(
            model=GROQ_MODEL, temperature=0.2, max_tokens=600,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": context}])
        text = (r.choices[0].message.content or "").strip()
    except Exception as e:  # network / quota / model errors must never break the safety flow
        return template_advice(glucose, labs, xray), f"template (LLM error: {type(e).__name__})"
    if not is_safe_advice(text):
        return template_advice(glucose, labs, xray), "template (LLM output rejected by safety check)"
    return text, f"Groq {GROQ_MODEL}"
