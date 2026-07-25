import base64
import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, callback, no_update
from dash import dash_table
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# App metadata
# -----------------------------
APP_TITLE = "Healthcare Analytics Dashboard"
CREATOR = "Aniruddha Dhibar"

# -----------------------------
# Theme colors (fully black background)
# -----------------------------
BLACK = "#000000"
PANEL = "#050505"
BORDER = "#1f1f1f"
TEXT = "#F5F5F5"
MUTED = "#BDBDBD"
ACCENT = "#00E5FF"
ACCENT2 = "#7CFF6B"
WARN = "#FF4D6D"

np.random.seed(7)

# -----------------------------
# Synthetic patient data
# -----------------------------
N = 250
patients = pd.DataFrame({
    "Patient ID": [f"P{i:04d}" for i in range(1, N + 1)],
    "Age": np.random.randint(18, 90, N),
    "BMI": np.round(np.random.normal(27, 5, N).clip(15, 50), 1),
    "Blood Pressure": np.random.randint(90, 180, N),
    "Glucose": np.random.randint(70, 220, N),
    "Smoker": np.random.choice([0, 1], N, p=[0.75, 0.25]),
    "Family History": np.random.choice([0, 1], N, p=[0.7, 0.3]),
    "Biomarker": np.round(np.random.normal(2.4, 0.8, N).clip(0.2, 8), 2),
})

logit = (
    0.03 * (patients["Age"] - 50)
    + 0.06 * (patients["BMI"] - 25)
    + 0.012 * (patients["Blood Pressure"] - 120)
    + 0.015 * (patients["Glucose"] - 100)
    + 0.9 * patients["Smoker"]
    + 0.8 * patients["Family History"]
    + 0.45 * patients["Biomarker"]
)
prob = 1 / (1 + np.exp(-logit / 4))
patients["Cancer Risk"] = (prob > np.quantile(prob, 0.62)).astype(int)
patients["Risk Score"] = np.round(prob, 3)

feature_cols = ["Age", "BMI", "Blood Pressure", "Glucose", "Smoker", "Family History", "Biomarker"]
X = patients[feature_cols]
y = patients["Cancer Risk"]

# -----------------------------
# ML model for cancer risk
# -----------------------------
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=120, random_state=7)),
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
model.fit(Xtr, ytr)
auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])

# Precompute feature importances for explanations
feature_importance = pd.Series(
    model.named_steps["clf"].feature_importances_,
    index=feature_cols,
).sort_values(ascending=False)

# -----------------------------
# Minimal NLP for clinical notes
# -----------------------------
notes_df = pd.DataFrame({
    "note": [
        "Patient reports cough and fever. Chest xray pending. High glucose noted.",
        "No acute distress. Blood pressure elevated. Family history of cancer.",
        "Fatigue and weight loss with smoking history and abnormal biomarker.",
        "Routine follow up. Normal vitals and stable labs.",
    ],
    "summary": [
        "Possible respiratory issue with elevated glucose.",
        "Hypertension risk and family cancer history.",
        "Higher concern due to smoking, weight loss, biomarker.",
        "No major abnormalities identified.",
    ],
})

text_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB()),
])
text_model.fit(notes_df["note"], [1, 1, 1, 0])

# -----------------------------
# Collaboration + audit trail store
# -----------------------------
shared_store = {
    "messages": [],
    "ledger": [
        {"time": datetime.utcnow().strftime("%H:%M:%S"), "event": "Consent enabled"},
        {"time": datetime.utcnow().strftime("%H:%M:%S"), "event": "Audit trail initialized"},
    ],
}

# -----------------------------
# Helpers
# -----------------------------
def black_fig(title: str):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        font=dict(color=TEXT),
        title=dict(text=title, x=0.02),
        margin=dict(l=40, r=20, t=55, b=40),
        legend=dict(bgcolor=BLACK),
    )
    fig.update_xaxes(gridcolor="#222", zerolinecolor="#222", color=TEXT)
    fig.update_yaxes(gridcolor="#222", zerolinecolor="#222", color=TEXT)
    return fig

def kpi_card(title, value, sub, color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, style={"color": MUTED, "fontSize": "0.9rem"}),
            html.Div(value, style={"color": color, "fontSize": "2rem", "fontWeight": "700"}),
            html.Div(sub, style={"color": "#DDD", "fontSize": "0.85rem"}),
        ]),
        style={
            "backgroundColor": PANEL,
            "border": f"1px solid {BORDER}",
            "borderRadius": "16px",
            "boxShadow": "0 0 12px rgba(0,229,255,0.08)",
        },
    )

def simple_clinical_interpretation(note: str):
    note_lower = note.lower()
    findings = []
    if "fever" in note_lower or "cough" in note_lower:
        findings.append("Possible infection or respiratory issue.")
    if "blood pressure" in note_lower or "hypertension" in note_lower:
        findings.append("Blood pressure needs attention.")
    if "smoking" in note_lower:
        findings.append("Smoking increases long-term disease risk.")
    if "weight loss" in note_lower:
        findings.append("Unexplained weight loss may need review.")
    if "biomarker" in note_lower:
        findings.append("Abnormal biomarker should be clinically reviewed.")
    if not findings:
        findings.append("No major red flags detected in the note.")
    return " ".join(findings)

def patient_risk_row(pid):
    row = patients[patients["Patient ID"] == pid].iloc[0]
    row_df = pd.DataFrame([row[feature_cols].to_dict()])
    risk = model.predict_proba(row_df)[:, 1][0]
    return row, risk

def age_group(age):
    if age < 35:
        return "<35"
    elif age < 50:
        return "35-49"
    elif age < 65:
        return "50-64"
    else:
        return "65+"

patients["Age Group"] = patients["Age"].apply(age_group)

# -----------------------------
# Dash app initialization
# -----------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title=APP_TITLE,
    suppress_callback_exceptions=True,
)
server = app.server

# -----------------------------
# Layout (with dedicated upload result area)
# -----------------------------
app.layout = html.Div(
    style={
        "backgroundColor": BLACK,
        "minHeight": "100vh",
        "color": TEXT,
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        dcc.Store(id="chat-store", data=shared_store),
        dcc.Store(id="upload-store", data=None),  # persistent upload result
        dcc.Interval(id="live-interval", interval=2500, n_intervals=0),
        dbc.Container(
            fluid=True,
            children=[
                html.Div(
                    style={"padding": "16px 8px"},
                    children=[
                        html.H2(APP_TITLE, style={"margin": "0", "color": TEXT}),
                        html.Div(f"Created by {CREATOR}", style={"color": ACCENT, "marginTop": "4px"}),
                        html.Div(
                            "Accessible AI-assisted clinical analytics, prevention support, collaboration, NLP reporting, and secure sharing.",
                            style={"color": MUTED},
                        ),
                    ],
                ),
                dbc.Row(
                    [
                        dbc.Col(kpi_card("Patients", str(len(patients)), "Synthetic live demo data", ACCENT), md=3),
                        dbc.Col(kpi_card("Cancer positives", str(int(patients["Cancer Risk"].sum())), f"Model AUC {auc:.2f}", ACCENT2), md=3),
                        dbc.Col(kpi_card("High biomarker", str(int((patients["Biomarker"] > 3.0).sum())), "Flagged for review", WARN), md=3),
                        dbc.Col(kpi_card("Status", "ONLINE", "Realtime updates enabled", ACCENT), md=3),
                    ],
                    className="g-3",
                    style={"marginBottom": "12px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="patient-dd",
                                options=[{"label": p, "value": p} for p in patients["Patient ID"][:20]],
                                value="P0001",
                                clearable=False,
                            ),
                            md=3,
                        ),
                        dbc.Col(
                            dbc.Input(id="chat-input", placeholder="Type a team message...", type="text"),
                            md=6,
                        ),
                        dbc.Col(
                            dbc.Button("Send", id="send-btn", color="info", className="w-100"),
                            md=3,
                        ),
                    ],
                    className="g-2",
                    style={"marginBottom": "12px"},
                ),
                dbc.Tabs(
                    id="tabs",
                    active_tab="tab-live",
                    children=[
                        dbc.Tab(label="Live Data", tab_id="tab-live"),
                        dbc.Tab(label="Cancer Risk", tab_id="tab-cancer"),
                        dbc.Tab(label="Population Insights", tab_id="tab-pop"),
                        dbc.Tab(label="Patient Explorer", tab_id="tab-explorer"),
                        dbc.Tab(label="Clinical Report", tab_id="tab-nlp"),
                        dbc.Tab(label="Data Upload", tab_id="tab-upload"),
                        dbc.Tab(label="Collaboration", tab_id="tab-collab"),
                        dbc.Tab(label="Secure Sharing", tab_id="tab-chain"),
                    ],
                ),
                html.Div(id="tab-content", style={"padding": "14px 4px"}),
                # This div will hold upload result independently of tabs
                html.Div(id="upload-result-container", style={"padding": "0 12px 20px 12px"}),
            ],
        ),
    ],
)

# -----------------------------
# Callbacks (all tabs except upload result)
# -----------------------------
@callback(
    Output("chat-store", "data", allow_duplicate=True),
    Input("send-btn", "n_clicks"),
    State("chat-input", "value"),
    State("chat-store", "data"),
    prevent_initial_call=True,
)
def add_message(n, msg, data):
    if not msg:
        return no_update
    data = data or {"messages": [], "ledger": []}
    data["messages"] = (data.get("messages", []) + [{
        "t": datetime.utcnow().strftime("%H:%M:%S"),
        "m": msg,
    }])[-10:]
    data["ledger"] = (data.get("ledger", []) + [{
        "time": datetime.utcnow().strftime("%H:%M:%S"),
        "event": f"Shared note: {msg[:30]}",
    }])[-15:]
    return data

@callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("live-interval", "n_intervals"),
    Input("patient-dd", "value"),
    Input("chat-store", "data"),
)
def render_tab(tab, n, patient_id, chat_data):
    row, risk = patient_risk_row(patient_id)

    if tab == "tab-live":
        fig = black_fig("Live Vital Trend")
        xs = list(range(20))
        ys = 72 + np.sin(np.linspace(0, 4 * np.pi, 20)) * 6 + np.random.normal(0, 1.2, 20)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line=dict(color=ACCENT, width=3),
                marker=dict(size=7),
                name="Heart rate",
            )
        )

        table = dash_table.DataTable(
            data=patients.head(8).to_dict("records"),
            columns=[{"name": c, "id": c} for c in patients.columns],
            style_table={"overflowX": "auto"},
            style_cell={
                "backgroundColor": PANEL,
                "color": TEXT,
                "border": f"1px solid {BORDER}",
                "fontFamily": "Arial",
                "textAlign": "left",
                "padding": "8px",
            },
            style_header={
                "backgroundColor": "#111",
                "fontWeight": "bold",
                "border": f"1px solid {BORDER}",
            },
        )
        return html.Div([dcc.Graph(figure=fig), table])

    if tab == "tab-cancer":
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=patients["Age"],
                    y=patients["Biomarker"],
                    z=patients["BMI"],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=patients["Risk Score"],
                        colorscale="Viridis",
                        opacity=0.8,
                    ),
                    text=patients["Patient ID"],
                    name="Patients",
                ),
                go.Scatter3d(
                    x=[row["Age"]],
                    y=[row["Biomarker"]],
                    z=[row["BMI"]],
                    mode="markers+text",
                    marker=dict(size=10, color=WARN),
                    text=[f"{patient_id}: {risk:.2f}"],
                    textposition="top center",
                    name="Selected patient",
                ),
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BLACK,
            plot_bgcolor=BLACK,
            font=dict(color=TEXT),
            scene=dict(
                xaxis=dict(backgroundcolor=BLACK, gridcolor="#222", color=TEXT, title="Age"),
                yaxis=dict(backgroundcolor=BLACK, gridcolor="#222", color=TEXT, title="Biomarker"),
                zaxis=dict(backgroundcolor=BLACK, gridcolor="#222", color=TEXT, title="BMI"),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return html.Div([
            dcc.Graph(figure=fig),
            html.Div(f"Predicted cancer risk for {patient_id}: {risk:.2%}", style={"color": ACCENT, "fontSize": "1.2rem"}),
            html.Div("This result supports prevention workflows and should be interpreted by a clinician.", style={"color": MUTED}),
        ])

    if tab == "tab-pop":
        agg = patients.groupby("Age Group", as_index=False).agg(
            {"Risk Score": "mean", "Cancer Risk": "sum", "Patient ID": "count"}
        ).rename(columns={"Patient ID": "Count"})
        fig_age = black_fig("Average Risk Score by Age Group")
        fig_age.add_trace(
            go.Bar(
                x=agg["Age Group"],
                y=agg["Risk Score"],
                marker_color=ACCENT,
                name="Avg Risk Score",
            )
        )
        smoke_agg = patients.groupby("Smoker", as_index=False)["Risk Score"].mean()
        smoke_agg["Label"] = smoke_agg["Smoker"].map({0: "Non-smoker", 1: "Smoker"})
        fig_smoke = black_fig("Risk by Smoking Status")
        fig_smoke.add_trace(
            go.Bar(
                x=smoke_agg["Label"],
                y=smoke_agg["Risk Score"],
                marker_color=[ACCENT2, WARN],
                name="Avg Risk",
            )
        )
        heat = patients.groupby(["Age Group", "Smoker"])["Risk Score"].mean().unstack(fill_value=0)
        heat.index = heat.index.astype(str)
        heat.columns = heat.columns.map({0: "Non-smoker", 1: "Smoker"})
        fig_heat = black_fig("Risk Heatmap: Age Group × Smoking")
        fig_heat.add_trace(
            go.Heatmap(
                z=heat.values,
                x=heat.columns.tolist(),
                y=heat.index.tolist(),
                colorscale="Viridis",
                showscale=True,
            )
        )
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_age), md=6),
                dbc.Col(dcc.Graph(figure=fig_smoke), md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_heat), md=12),
            ], className="g-3 mt-2"),
            html.Div("Population-level insights help prioritize prevention programs.", style={"color": MUTED, "marginTop": "8px"}),
        ])

    if tab == "tab-explorer":
        fig_imp = black_fig("Model Feature Importance")
        fig_imp.add_trace(
            go.Bar(
                x=feature_importance.values,
                y=feature_importance.index,
                orientation="h",
                marker_color=ACCENT,
                name="Importance",
            )
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})

        vals = [row[feat] for feat in feature_cols]
        norm_vals = [
            (v - patients[feat].min()) / (patients[feat].max() - patients[feat].min() + 1e-6)
            for feat, v in zip(feature_cols, vals)
        ]
        fig_radar = black_fig(f"Risk Factor Profile: {patient_id}")
        fig_radar.add_trace(
            go.Scatterpolar(
                r=norm_vals + [norm_vals[0]],
                theta=feature_cols + [feature_cols[0]],
                fill="toself",
                line=dict(color=ACCENT2, width=3),
                name="Patient",
            )
        )

        top_factors = feature_importance.head(3).index.tolist()
        explanation = (
            f"For {patient_id}, the model's prediction is driven mainly by: "
            + ", ".join(top_factors)
            + ". Higher values in these factors generally increase risk."
        )
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_imp), md=6),
                dbc.Col(dcc.Graph(figure=fig_radar), md=6),
            ], className="g-3"),
            html.Div(explanation, style={"color": TEXT, "marginTop": "12px", "fontSize": "1.05rem"}),
            html.Div("Use this to discuss personalized prevention strategies with a clinician.", style={"color": MUTED}),
        ])

    if tab == "tab-nlp":
        idx = n % len(notes_df)
        note = notes_df.iloc[idx]["note"]
        summary = notes_df.iloc[idx]["summary"]
        generated = simple_clinical_interpretation(note)
        return html.Div([
            dbc.Textarea(
                value=note,
                style={"backgroundColor": PANEL, "color": TEXT, "height": "140px"},
                readOnly=True,
            ),
            html.Hr(style={"borderColor": BORDER}),
            html.Div("AI-assisted summary:", style={"color": ACCENT2, "fontWeight": "700"}),
            html.Div(summary, style={"color": TEXT, "marginBottom": "8px"}),
            html.Div("Plain-language clinical note interpretation:", style={"color": ACCENT, "fontWeight": "700"}),
            html.Div(generated, style={"color": TEXT}),
        ])

    if tab == "tab-upload":
        # Only show the upload box here; result is shown in upload-result-container
        return html.Div([
            html.Div(
                [
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div([
                            "Drag & Drop or ",
                            html.A("Click to Upload", style={"color": ACCENT}),
                            " any CSV file.",
                        ]),
                        style={
                            "width": "100%",
                            "height": "80px",
                            "lineHeight": "80px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "10px",
                            "textAlign": "center",
                            "margin": "10px 0",
                            "backgroundColor": PANEL,
                            "color": TEXT,
                            "borderColor": BORDER,
                        },
                        multiple=False,
                    ),
                ],
                style={"backgroundColor": PANEL, "padding": "16px", "border": f"1px solid {BORDER}", "borderRadius": "12px"},
            ),
            html.Div(
                "You can upload any CSV file. If it contains healthcare-like columns (Age, BMI, Blood Pressure, Glucose, Smoker, Family History, Biomarker), the dashboard will also show risk predictions.",
                style={"color": MUTED, "marginTop": "8px"},
            ),
        ])

    if tab == "tab-collab":
        msgs = (chat_data or {}).get("messages", [])
        items = [
            html.Div(
                f"{m['t']}  {m['m']}",
                style={"padding": "6px 0", "borderBottom": f"1px solid {BORDER}"},
            )
            for m in msgs
        ]
        if not items:
            items = [html.Div("No shared messages yet.", style={"color": MUTED})]
        return html.Div([
            html.Div(
                items,
                style={
                    "backgroundColor": PANEL,
                    "padding": "12px",
                    "border": f"1px solid {BORDER}",
                    "borderRadius": "12px",
                    "minHeight": "120px",
                },
            ),
            html.Div("Shared room updates appear for all connected users.", style={"marginTop": "8px", "color": MUTED}),
        ])

    fig = black_fig("Secure Sharing Ledger")
    ledger = (chat_data or {}).get("ledger", [])
    x = [x["time"] for x in ledger] if ledger else ["--"]
    y = list(range(1, len(x) + 1)) if ledger else [1]
    fig.add_trace(go.Bar(x=x, y=y, marker_color=ACCENT))
    return html.Div([
        dcc.Graph(figure=fig),
        html.Div("Blockchain-style audit trail demo for secure healthcare sharing.", style={"color": MUTED}),
        html.Div("Only hashes and events should be shared on-chain in a real deployment.", style={"color": ACCENT2}),
    ])

# -----------------------------
# Upload callback (robust, no index errors)
# -----------------------------
def safe_read_csv(contents):
    if contents is None:
        return None
    if isinstance(contents, list):
        if len(contents) == 0:
            return None
        contents = contents[0]
    if not isinstance(contents, str):
        return None

    try:
        content_type, content_string = contents.split(",", 1)
    except Exception:
        return None

    try:
        decoded = base64.b64decode(content_string)
    except Exception:
        return None

    try:
        df = pd.read_csv(io.BytesIO(decoded), encoding_errors="ignore")
        return df
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(decoded), engine="python", encoding_errors="ignore")
            return df
        except Exception:
            return None

def map_to_standard_columns(df):
    col_map = {
        "age": "Age",
        "ages": "Age",
        "patient_age": "Age",
        "bmi": "BMI",
        "body_mass_index": "BMI",
        "bloodpressure": "Blood Pressure",
        "blood pressure": "Blood Pressure",
        "bp": "Blood Pressure",
        "systolic": "Blood Pressure",
        "glucose": "Glucose",
        "blood_glucose": "Glucose",
        "sugar": "Glucose",
        "smoker": "Smoker",
        "smoking": "Smoker",
        "smoking_status": "Smoker",
        "familyhistory": "Family History",
        "family history": "Family History",
        "family_hx": "Family History",
        "biomarker": "Biomarker",
        "marker": "Biomarker",
        "risk_marker": "Biomarker",
    }
    rename_dict = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in col_map:
            rename_dict[c] = col_map[cl]
    return df.rename(columns=rename_dict)

@callback(
    Output("upload-store", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        return no_update

    df = safe_read_csv(contents)
    if df is None or df.empty:
        return {
            "error": "Could not parse this CSV file. Please check the format.",
            "type": "error",
        }

    rows, cols = df.shape
    summary_text = f"Uploaded CSV: {rows} rows × {cols} columns."

    df_mapped = map_to_standard_columns(df.copy())
    required = ["Age", "BMI", "Blood Pressure", "Glucose", "Smoker", "Family History", "Biomarker"]
    present = [c for c in required if c in df_mapped.columns]

    result = {
        "type": "ok",
        "summary": summary_text,
        "is_healthcare": len(present) == len(required),
        "missing_cols": None if len(present) == len(required) else [c for c in required if c not in df_mapped.columns],
        "raw_cols": [str(c) for c in df.columns],
        "raw_head": df.head(10).to_dict("records"),
        "risk_head": None,
        "risk_cols": None,
        "high_risk_count": None,
        "total_risk_rows": None,
    }

    if result["is_healthcare"]:
        try:
            for c in ["Age", "BMI", "Blood Pressure", "Glucose", "Biomarker"]:
                df_mapped[c] = pd.to_numeric(df_mapped[c], errors="coerce")
            for c in ["Smoker", "Family History"]:
                df_mapped[c] = pd.to_numeric(df_mapped[c], errors="coerce").fillna(0).astype(int)

            df_clean = df_mapped.dropna(subset=required)

            if len(df_clean) > 0:
                Xnew = df_clean[required]
                risk_new = model.predict_proba(Xnew)[:, 1]
                df_clean["Predicted Risk"] = np.round(risk_new, 3)
                high = int((df_clean["Predicted Risk"] > 0.6).sum())
                total = len(df_clean)

                result["risk_head"] = df_clean.head(10).to_dict("records")
                result["risk_cols"] = [str(c) for c in df_clean.columns]
                result["high_risk_count"] = high
                result["total_risk_rows"] = total
        except Exception:
            # If prediction fails, just show raw data
            result["is_healthcare"] = False
            result["missing_cols"] = ["(prediction failed)"]

    return result

# -----------------------------
# Render upload result (separate callback, stays visible)
# -----------------------------
@callback(
    Output("upload-result-container", "children"),
    Input("upload-store", "data"),
    prevent_initial_call=True,
)
def render_upload_result(stored):
    if stored is None:
        return html.Div()

    if stored.get("type") == "error":
        return html.Div(
            stored["error"],
            style={"color": WARN, "padding": "12px"},
        )

    summary_text = stored["summary"]
    is_hc = stored["is_healthcare"]
    raw_head = stored["raw_head"]
    raw_cols = stored["raw_cols"]

    parts = [
        html.Div(summary_text, style={"color": ACCENT, "fontSize": "1.05rem", "marginBottom": "8px"}),
    ]

    if is_hc:
        risk_head = stored["risk_head"]
        risk_cols = stored["risk_cols"]
        high = stored["high_risk_count"]
        total = stored["total_risk_rows"]

        parts.append(
            html.Div(
                "This file looks like healthcare data. Showing risk predictions for matching rows:",
                style={"color": ACCENT2, "fontWeight": "700", "marginTop": "12px"},
            )
        )
        parts.append(
            html.Div(
                f"Valid rows used for prediction: {total}. High-risk (>60%): {high}",
                style={"color": ACCENT, "fontSize": "1.05rem"},
            )
        )
        parts.append(
            html.Div("Preview of predictions:", style={"color": MUTED, "marginTop": "8px"})
        )
        parts.append(
            dash_table.DataTable(
                data=risk_head,
                columns=[{"name": c, "id": c} for c in risk_cols],
                style_table={"overflowX": "auto", "marginTop": "8px"},
                style_cell={
                    "backgroundColor": PANEL,
                    "color": TEXT,
                    "border": f"1px solid {BORDER}",
                    "fontFamily": "Arial",
                    "textAlign": "left",
                    "padding": "6px",
                },
                style_header={
                    "backgroundColor": "#111",
                    "fontWeight": "bold",
                    "border": f"1px solid {BORDER}",
                },
            )
        )
    else:
        missing = stored["missing_cols"]
        if missing:
            parts.append(
                html.Div(
                    f"This CSV does not look like healthcare data (missing: {', '.join(missing)}). Showing raw preview only.",
                    style={"color": MUTED, "marginTop": "12px"},
                )
            )

    parts.append(
        html.Div("Raw CSV preview (first 10 rows):", style={"color": TEXT, "marginTop": "12px"})
    )
    parts.append(
        dash_table.DataTable(
            data=raw_head,
            columns=[{"name": c, "id": c} for c in raw_cols],
            style_table={"overflowX": "auto", "marginTop": "8px"},
            style_cell={
                "backgroundColor": PANEL,
                "color": TEXT,
                "border": f"1px solid {BORDER}",
                "fontFamily": "Arial",
                "textAlign": "left",
                "padding": "6px",
            },
            style_header={
                "backgroundColor": "#111",
                "fontWeight": "bold",
                "border": f"1px solid {BORDER}",
            },
        )
    )

    return html.Div(parts, style={"padding": "0 4px 20px 4px"})

if __name__ == "__main__":
    app.run(debug=False)