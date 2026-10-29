"""Lab analyser: report PDF/image -> text (pdfplumber / Tesseract) -> regex parsing -> unit
normalisation -> reference-range flags, plus an optional XGBoost diabetes-risk score (NHANES)."""
import io
import re
from typing import Optional

import numpy as np

from ..config import WEIGHTS

# name: (aliases, canonical unit, low, high, critical_low, critical_high, {reported unit (lowercase, u for micro): factor})
# Reference ranges are typical adult values; critical limits are example thresholds for review by a clinician.
ANALYTES = {
    "Glucose (fasting)": (["fasting glucose", "fasting blood sugar", "fbs", "glucose"], "mg/dL", 70, 99, 54, 400, {"mmol/l": 18.016}),
    "HbA1c": (["hba1c", "glycated hemoglobin", "glycosylated hemoglobin", "a1c"], "%", 4.0, 5.6, None, 14, {}),
    "Hemoglobin": (["hemoglobin", "haemoglobin", "hb"], "g/dL", 12.0, 17.5, 7, 20, {"g/l": 0.1}),
    "WBC": (["wbc", "white blood cell", "total leucocyte count", "tlc"], "10^3/uL", 4.0, 11.0, 2.0, 30, {}),
    "Platelets": (["platelet count", "platelets", "plt"], "10^3/uL", 150, 450, 20, 1000, {}),
    "Creatinine": (["creatinine"], "mg/dL", 0.6, 1.3, None, 4.0, {"umol/l": 1 / 88.4}),
    "Sodium": (["sodium", "na+"], "mmol/L", 135, 145, 120, 160, {"meq/l": 1}),
    "Potassium": (["potassium", "k+"], "mmol/L", 3.5, 5.1, 2.8, 6.5, {"meq/l": 1}),
    "LDL": (["ldl cholesterol", "ldl"], "mg/dL", 0, 99, None, None, {"mmol/l": 38.67}),
    "HDL": (["hdl cholesterol", "hdl"], "mg/dL", 40, 100, None, None, {"mmol/l": 38.67}),
    "Total cholesterol": (["total cholesterol", "cholesterol, total", "cholesterol total"], "mg/dL", 0, 199, None, None, {"mmol/l": 38.67}),
    "Triglycerides": (["triglycerides"], "mg/dL", 0, 149, None, 1000, {"mmol/l": 88.57}),
    "ALT": (["alt", "sgpt", "alanine aminotransferase"], "U/L", 7, 56, None, None, {}),
    "AST": (["ast", "sgot", "aspartate aminotransferase"], "U/L", 10, 40, None, None, {}),
}
AUTO_THOUSANDS = {"WBC": 500, "Platelets": 5000}  # "7500 /cumm" -> 7.5 x10^3/uL
RISK_FEATURES = ["age", "HbA1c", "Total cholesterol", "HDL", "Creatinine", "ALT"]  # order used in colab/03

_NUM = r"(\d+(?:\.\d+)?)\s*(%|[a-z\u00b5\u03bc0-9^*]+/[a-z\u00b5\u03bc0-9^*]+)?"


def parse_text(text: str) -> dict:
    """Return {analyte: (value in canonical unit, canonical unit)}. One analyte per line."""
    found = {}
    for line in text.lower().splitlines():
        if "average" in line or "estimated" in line:
            continue
        for name, (aliases, unit, *_rest) in ANALYTES.items():
            if name in found:
                continue
            conv = ANALYTES[name][6]
            hit = None
            for a in aliases:
                m = re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z])[^\d\n]{{0,40}}?{_NUM}", line)
                if m:
                    hit = m
                    break
            if not hit:
                continue
            val = float(hit.group(1))
            rep = (hit.group(2) or "").replace("\u00b5", "u").replace("\u03bc", "u")
            val *= conv.get(rep, 1.0)
            if name in AUTO_THOUSANDS and val > AUTO_THOUSANDS[name]:
                val /= 1000
            found[name] = (round(val, 2), unit)
            break  # a line normally holds a single analyte
    return found


def _flag(v, low, high, clo, chi) -> str:
    if clo is not None and v < clo:
        return "critical_low"
    if chi is not None and v > chi:
        return "critical_high"
    return "low" if v < low else "high" if v > high else "normal"


def _preprocess(gray: np.ndarray) -> np.ndarray:
    """Upscale, deskew (angle chosen by sharpest text-line profile) and binarise a report photo/scan."""
    import cv2
    if gray.shape[1] < 1400:
        s = 1400 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    pts = np.column_stack(np.where(inv > 0)).astype(np.float32)
    best = gray
    if len(pts) > 200:
        ang = ((cv2.minAreaRect(pts)[-1] + 45) % 90) - 45
        if 0.3 < abs(ang) < 15:
            h, w = gray.shape
            best_score = -1.0
            for a in (0, ang, -ang):
                rot = cv2.warpAffine(gray, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), (w, h),
                                     flags=cv2.INTER_CUBIC, borderValue=255)
                score = float(np.var(cv2.threshold(rot, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1].sum(axis=1)))
                if score > best_score:
                    best, best_score = rot, score
    return cv2.adaptiveThreshold(best, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def read_report(data: bytes, filename: str) -> str:
    import cv2
    import pytesseract
    if filename.lower().endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            if text.strip():
                return text
            # scanned PDF: OCR every page
            return "\n".join(
                pytesseract.image_to_string(_preprocess(np.array(p.to_image(resolution=200).original.convert("L"))), config="--psm 6")
                for p in pdf.pages)
    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not read the report file (use PDF, PNG or JPEG).")
    return pytesseract.image_to_string(_preprocess(gray), config="--psm 6")


class LabAnalyzer:
    def __init__(self):
        self.risk = None
        path = WEIGHTS / "lab_risk_xgb.json"
        if path.exists():
            import xgboost as xgb
            self.risk = xgb.XGBClassifier()
            self.risk.load_model(str(path))

    def analyze_text(self, text: str, age: Optional[int] = None) -> dict:
        found = parse_text(text)
        results = []
        for name, (val, unit) in found.items():
            _a, _u, low, high, clo, chi, _c = ANALYTES[name]
            results.append({"name": name, "value": val, "unit": unit, "reference": f"{low:g}-{high:g}",
                            "flag": _flag(val, low, high, clo, chi)})
        out = {"results": results, "abnormal": [r["name"] for r in results if r["flag"] != "normal"], "risk": None}
        if self.risk is not None:
            row = [float(age) if age else np.nan] + [found.get(n, (np.nan,))[0] for n in RISK_FEATURES[1:]]
            if np.count_nonzero(~np.isnan(row)) >= 2:
                out["risk"] = {"diabetes_risk": round(float(self.risk.predict_proba(np.array([row], dtype=np.float32))[0, 1]), 3),
                               "model": "XGBoost on NHANES (screening aid only)"}
        return out

    def analyze_file(self, data: bytes, filename: str, age: Optional[int] = None) -> dict:
        res = self.analyze_text(read_report(data, filename), age)
        if not res["results"]:
            raise ValueError("No known lab values were found in the report. Try a clearer scan or a text-based PDF.")
        return res
