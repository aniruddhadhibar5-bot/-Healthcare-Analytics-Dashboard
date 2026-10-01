# Healthcare Analytics Dashboard

An interactive Dash demo for exploring **synthetic** patient data. It includes cohort filters, interactive population charts, a patient explorer, holdout model evaluation, CSV preview and demo scoring, clinical-note examples, and collaboration/audit-trail demonstrations.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local URL printed by Dash (usually `http://127.0.0.1:8050`). The WSGI entry point for deployment is `app:server`.

## Important limitations

All included patient records and model labels are randomly generated for demonstration. The model learns a synthetic target, not a clinically diagnosed outcome; its scores, feature importances, and evaluation metrics are not validated clinical evidence. The vital-sign chart is illustrative, not live monitoring. The note interpretation uses simple keyword rules and is not medical advice. Do not use this demo to make patient-care decisions or upload identifiable health information.
