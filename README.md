# MedAssist - local multimodal medical assistant (Milestone 2 implementation)

Glucose forecasting, blood-report reading and chest X-ray analysis feed a retrieval-augmented advice agent behind a rule-based safety layer.
Decision support only - not a diagnostic device.

```
colab/       training notebooks (.ipynb, generated from the .py files) -> produce the weights
backend/     FastAPI + uvicorn API, PyTorch inference, FAISS retriever, Groq LLM via the OpenAI SDK
frontend/    React + TypeScript (Vite) UI
```

## 1. Train the models on Google Colab (GPU runtime)
| Notebook | Trains | Download | Put in `backend/weights/` |
|---|---|---|---|
| `01_glucose_gru` | GRU + XGBoost baseline, Clarke Error Grid report | `glucose_gru.pt` | yes |
| `02_xray_densenet` | DenseNet121 (Kaggle pneumonia set, patient-grouped 80/10/10 split) | `xray_densenet121.pt`, `xray_meta.json` | yes |
| `03_lab_risk_xgboost` (optional) | XGBoost on NHANES 2017-18 | `lab_risk_xgb.json` | yes |

Notebook 1 needs per-patient CGM files in `/content/cgm` (ShanghaiT1DM / T2DM). Notebook 2 asks for your `kaggle.json`.
Without a weights file the app still runs: glucose falls back to a persistence baseline, and X-ray analysis is reported as unavailable.

## 2. Backend
```powershell
winget install UB-Mannheim.TesseractOCR # needed for scanned reports / images run it in any environment, set its path in environmnet variables
cd backend
python -m venv .venv 
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env             # add GROQ_API_KEY from https://console.groq.com/keys
pytest                                  # safety-trigger + lab-parsing tests
uvicorn app.main:app --reload --port 8000
```
Without `GROQ_API_KEY` the advice falls back to a deterministic template. `GET /api/health` shows what is loaded.
Add the ADA / WHO guideline PDFs (or .md/.txt) to `backend/data/guidelines/`; they are chunked and indexed at startup.

## 3. Frontend
```powershell
cd frontend
npm install
npm run dev                             # http://localhost:5173 (proxies /api to :8000)
```
For a separately hosted API, set `VITE_API_URL` and add the UI origin to `CORS_ORIGINS`.

## How a request flows
`POST /api/analyze` (multipart: `glucose`, `report`, `xray`, `age`, `question`) -> modules -> `safety.evaluate`
-> FAISS retrieval -> Groq LLM (skipped entirely on an emergency) -> `is_safe_advice` check (no doses) -> response.

## Deviations from the design document (deliberate)
- The LLM is a hosted Groq model through the OpenAI SDK, so QLoRA fine-tuning was dropped; grounding comes from RAG and the system prompt.
- Glucose inputs use fixed scaling (value/400, offset from latest reading) instead of per-patient z-scores, so a new user with only 2 h of data can be scored.
- The late-fusion "context aggregator" is `llm.build_context`; the safety thresholds in `safety.py` are example values that a clinician must review.
- Not included: Docker packaging, and the hit-rate@5 / ROUGE / BERTScore evaluation harness for the retriever and advice agent.
