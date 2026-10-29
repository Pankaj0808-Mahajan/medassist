"""Rule-based safety layer. Deterministic: it never calls the LLM and can override it.

Thresholds are example values for a student prototype; a clinician must review them
before any real-world use.
"""
import re
from typing import Optional

GLUCOSE_CRIT_LOW, GLUCOSE_LOW = 54, 70       # mg/dL
GLUCOSE_HIGH, GLUCOSE_CRIT_HIGH = 250, 300   # mg/dL
MIN_CONFIDENCE = 0.70                        # below this, a model output is not trusted

DISCLAIMER = ("This tool supports, but does not replace, a doctor. It is not a diagnostic device. "
              "Share these results with a qualified clinician.")

EMERGENCY_TEXT = ("One or more values are in a range that can be dangerous. Please do not wait for "
                  "automated advice: contact your doctor or local emergency services now. If you feel "
                  "confused, faint, very weak or short of breath, seek emergency care immediately.")
LOW_GLUCOSE_TIP = ("If your glucose is low and you are awake and able to swallow, the usual first step is "
                   "about 15 g of fast-acting sugar, then recheck in 15 minutes - and call your doctor.")

_RANK = {"routine": 0, "review": 1, "emergency": 2}


def evaluate(glucose: Optional[dict], labs: Optional[dict], xray: Optional[dict]) -> dict:
    level, reasons, tips = "routine", [], []

    def raise_to(new: str, why: str) -> None:
        nonlocal level
        if _RANK[new] > _RANK[level]:
            level = new
        reasons.append(why)

    if glucose:
        cur = glucose["current"]
        f30, f60 = glucose["forecast"]["30"], glucose["forecast"]["60"]
        if cur < GLUCOSE_CRIT_LOW:
            raise_to("emergency", f"Current glucose {cur:.0f} mg/dL is severely low")
            tips.append(LOW_GLUCOSE_TIP)
        elif cur > GLUCOSE_CRIT_HIGH:
            raise_to("emergency", f"Current glucose {cur:.0f} mg/dL is very high")
        elif min(f30, f60) < GLUCOSE_CRIT_LOW:
            raise_to("emergency", f"Glucose is forecast to fall below {GLUCOSE_CRIT_LOW} mg/dL within 60 minutes")
            tips.append(LOW_GLUCOSE_TIP)
        elif cur < GLUCOSE_LOW or min(f30, f60) < GLUCOSE_LOW:
            raise_to("review", "Glucose is low or forecast to go low")
            tips.append(LOW_GLUCOSE_TIP)
        elif cur > GLUCOSE_HIGH or max(f30, f60) > GLUCOSE_CRIT_HIGH:
            raise_to("review", "Glucose is high or forecast to go very high")

    if labs:
        for r in labs.get("results", []):
            if r["flag"] in ("critical_low", "critical_high"):
                raise_to("emergency", f"{r['name']} {r['value']} {r['unit']} is in a critical range")

    if xray and xray.get("available"):
        if xray["confidence"] < MIN_CONFIDENCE:
            raise_to("review", f"X-ray model confidence is low ({xray['confidence']:.0%}); a doctor must read the image")
        elif xray["finding"] == "PNEUMONIA":
            raise_to("review", f"X-ray pattern suggests possible pneumonia ({xray['p_pneumonia']:.0%}); needs doctor review")

    return {"level": level, "reasons": reasons, "tips": list(dict.fromkeys(tips)),
            "message": EMERGENCY_TEXT if level == "emergency" else None}


_DOSING = re.compile(r"\b\d+(?:\.\d+)?\s*(?:units?|iu|mcg|ml|mg(?!\s*/))\b", re.I)


def is_safe_advice(text: str) -> bool:
    """Reject LLM text that contains medication or insulin dosing."""
    return bool(text.strip()) and not _DOSING.search(text)
