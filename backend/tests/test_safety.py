"""Safety-trigger tests: every critical case must escalate (objective: 100% trigger rate)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app import safety
from app.modules.labs import LabAnalyzer


def g(cur, f30, f60):
    return {"current": cur, "forecast": {"30": f30, "60": f60}}


@pytest.mark.parametrize("glucose", [g(45, 45, 45), g(320, 330, 340), g(90, 60, 50), g(110, 80, 52)])
def test_critical_glucose_triggers_emergency(glucose):
    assert safety.evaluate(glucose, None, None)["level"] == "emergency"


@pytest.mark.parametrize("line", ["Potassium 7.1 mmol/L", "Sodium 115 mmol/L", "Hemoglobin 5.9 g/dL",
                                  "Platelet Count 12 10^3/uL", "Glucose (Fasting) 2.5 mmol/L", "Creatinine 400 umol/L"])
def test_critical_labs_trigger_emergency(line):
    labs = LabAnalyzer().analyze_text(line)
    assert labs["results"], line
    assert safety.evaluate(None, labs, None)["level"] == "emergency"


def test_low_confidence_xray_needs_review():
    x = {"available": True, "finding": "NORMAL", "confidence": 0.55, "p_pneumonia": 0.45}
    assert safety.evaluate(None, None, x)["level"] == "review"


def test_normal_case_is_routine():
    labs = LabAnalyzer().analyze_text("Hemoglobin 14.2 g/dL\nPotassium 4.2 mmol/L")
    assert safety.evaluate(g(110, 112, 115), labs, None)["level"] == "routine"


def test_unit_conversion_and_parsing():
    labs = LabAnalyzer().analyze_text("HbA1c (Glycated Hemoglobin) 6.1 %\nGlucose, Fasting 5.5 mmol/L\nWBC 7500 /cumm")
    d = {r["name"]: r for r in labs["results"]}
    assert d["HbA1c"]["value"] == 6.1 and d["HbA1c"]["flag"] == "high"
    assert round(d["Glucose (fasting)"]["value"]) == 99
    assert d["WBC"]["value"] == 7.5 and "Hemoglobin" not in d


def test_llm_dosing_is_rejected():
    assert not safety.is_safe_advice("Take 10 units of insulin now.")
    assert safety.is_safe_advice("Your glucose is 70 mg/dL, which is at the low edge.")
