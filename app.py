import base64
import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, callback, ctx, no_update
from dash import dash_table
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
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
patients["Demo Positive Label"] = (prob > np.quantile(prob, 0.62)).astype(int)
patients["Risk Score"] = np.round(prob, 3)

feature_cols = ["Age", "BMI", "Blood Pressure", "Glucose", "Smoker", "Family History", "Biomarker"]
X = patients[feature_cols]
y = patients["Demo Positive Label"]

# -----------------------------
# Model benchmark for a synthetic classification target
# -----------------------------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
model_definitions = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=7),
    "Random Forest": RandomForestClassifier(
        n_estimators=240, min_samples_leaf=3, class_weight="balanced", random_state=7, n_jobs=1
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.08, l2_regularization=1.0, random_state=7
    ),
}

model_bank = {}
model_probabilities = {}
model_importance_bank = {}
metric_rows = []
for model_name, estimator in model_definitions.items():
    fitted_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])
    fitted_model.fit(Xtr, ytr)
    probabilities = fitted_model.predict_proba(Xte)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    model_bank[model_name] = fitted_model
    model_probabilities[model_name] = probabilities
    model_importance_bank[model_name] = permutation_importance(
        fitted_model, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=7
    ).importances_mean
    metric_rows.append({
        "Model": model_name,
        "ROC AUC": roc_auc_score(yte, probabilities),
        "PR AUC": average_precision_score(yte, probabilities),
        "Accuracy": accuracy_score(yte, predictions),
        "Precision": precision_score(yte, predictions, zero_division=0),
        "Recall": recall_score(yte, predictions, zero_division=0),
        "F1": f1_score(yte, predictions, zero_division=0),
        "Brier score": brier_score_loss(yte, probabilities),
    })

model_metrics = pd.DataFrame(metric_rows).sort_values("ROC AUC", ascending=False)
model = model_bank["Random Forest"]
auc = roc_auc_score(yte, model_probabilities["Random Forest"])
test_probabilities = model_probabilities["Random Forest"]
test_predictions = (test_probabilities >= 0.5).astype(int)
roc_fpr, roc_tpr, _ = roc_curve(yte, test_probabilities)
test_confusion = confusion_matrix(yte, test_predictions, labels=[0, 1])
true_negative, false_positive, false_negative, true_positive = test_confusion.ravel()
sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)
patients["Risk Score"] = np.round(model.predict_proba(X)[:, 1], 3)

# Unsupervised profiles are descriptive clusters, not diagnostic groups.
cluster_scaler = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cluster_features = cluster_scaler.fit_transform(X)
cluster_model = KMeans(n_clusters=4, n_init=20, random_state=7)
patients["Demo Cluster"] = cluster_model.fit_predict(cluster_features)
cluster_coordinates = PCA(n_components=2, random_state=7).fit_transform(cluster_features)
patients["PC1"] = cluster_coordinates[:, 0]
patients["PC2"] = cluster_coordinates[:, 1]
cluster_silhouette = silhouette_score(cluster_features, patients["Demo Cluster"])

# Global importance is descriptive; coefficient magnitude and tree importance
# are shown as model-level summaries, not patient-specific causal explanations.
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

def generate_live_reading(patient_id, tick):
    patient = patients.loc[patients["Patient ID"] == patient_id].iloc[0]
    patient_number = int(patient_id[1:])
    rng = np.random.default_rng(7 + patient_number * 1009 + tick)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Patient ID": patient_id,
        "Heart rate (bpm)": int(np.clip(rng.normal(70 + (patient["Age"] >= 65) * 5, 5), 48, 132)),
        "Systolic BP (mmHg)": int(np.clip(rng.normal(patient["Blood Pressure"], 7), 82, 205)),
        "Glucose (mg/dL)": int(np.clip(rng.normal(patient["Glucose"], 10), 55, 280)),
        "SpO2 (%)": round(float(np.clip(rng.normal(97.5, 0.7), 88, 100)), 1),
    }

def live_demo_flags(reading):
    checks = [
        ("Heart rate", reading["Heart rate (bpm)"] < 50 or reading["Heart rate (bpm)"] > 120),
        ("Systolic BP", reading["Systolic BP (mmHg)"] < 90 or reading["Systolic BP (mmHg)"] > 180),
        ("Glucose", reading["Glucose (mg/dL)"] < 70 or reading["Glucose (mg/dL)"] > 200),
        ("SpO2", reading["SpO2 (%)"] < 92),
    ]
    return [label for label, flagged in checks if flagged]

def append_live_reading(tick, patient_id, readings, patient_changed=False):
    readings = readings or []
    if patient_changed or (
        readings and readings[-1].get("Patient ID") != patient_id
    ):
        readings = []
    readings.append(generate_live_reading(patient_id, tick or 0))
    return readings[-60:]

def live_monitoring_layout(readings):
    current = readings[-1] if readings else None
    flags = live_demo_flags(current) if current else []
    flag_children = (
        [html.Span(f"{flag} demo threshold", style={"display": "inline-block", "padding": "6px 10px", "margin": "3px", "borderRadius": "16px", "backgroundColor": "#4a1520", "color": WARN}) for flag in flags]
        if flags else [html.Span("No demo thresholds crossed in latest reading.", style={"color": ACCENT2})]
    )
    metrics = [
        ("Heart rate", "live-heart-value", "bpm", ACCENT),
        ("Systolic BP", "live-bp-value", "mmHg", ACCENT2),
        ("Glucose", "live-glucose-value", "mg/dL", WARN),
        ("Oxygen saturation", "live-spo2-value", "%", ACCENT),
    ]
    cards = []
    for title, component_id, unit, color in metrics:
        value = current.get({
            "live-heart-value": "Heart rate (bpm)",
            "live-bp-value": "Systolic BP (mmHg)",
            "live-glucose-value": "Glucose (mg/dL)",
            "live-spo2-value": "SpO2 (%)",
        }[component_id], "--") if current else "--"
        cards.append(dbc.Col(
            dbc.Card(dbc.CardBody([
                html.Div(title, style={"color": MUTED, "fontSize": "0.85rem"}),
                html.Div([
                    html.Span(str(value), id=component_id, style={"color": color, "fontSize": "1.9rem", "fontWeight": "700"}),
                    html.Span(f" {unit}", style={"color": MUTED}),
                ]),
            ]), style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "14px"}),
            md=3,
        ))
    chart_cards = [
        ("Heart rate", "live-heart-chart"),
        ("Systolic blood pressure", "live-bp-chart"),
        ("Glucose", "live-glucose-chart"),
        ("Oxygen saturation", "live-spo2-chart"),
    ]
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.Span("● ", style={"color": ACCENT2}),
                html.Strong("SIMULATOR RUNNING", style={"color": ACCENT2}),
                html.Span("  •  Selected demo profile: P0001", id="live-selected-patient", style={"color": MUTED}),
            ]), md=8),
            dbc.Col(dbc.Button("Export readings CSV", id="live-export-button", color="info", outline=True, className="w-100"), md=4),
        ], className="align-items-center g-2 mb-3"),
        dbc.Row(cards, className="g-3 mb-3"),
        html.Div(id="live-alerts", children=flag_children, style={"padding": "8px 12px", "backgroundColor": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "12px", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id=chart_id,
                figure=black_fig(title),
                animate=True,
                config={"displaylogo": False, "scrollZoom": True},
            ), md=6)
            for title, chart_id in chart_cards
        ], className="g-3"),
        html.Div([
            html.Span("Last simulated sample: ", style={"color": MUTED}),
            html.Span(current["timestamp"] if current else "Waiting for first sample...", id="live-updated-at", style={"color": TEXT}),
        ], style={"padding": "4px 0 10px"}),
        dash_table.DataTable(
            id="live-readings-table",
            data=list(reversed(readings[-15:])),
            columns=[{"name": col, "id": col} for col in (readings[-1].keys() if readings else ["timestamp", "Patient ID", "Heart rate (bpm)", "Systolic BP (mmHg)", "Glucose (mg/dL)", "SpO2 (%)"])],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": PANEL, "color": TEXT, "border": f"1px solid {BORDER}", "padding": "7px", "textAlign": "left"},
            style_header={"backgroundColor": "#111", "fontWeight": "bold"},
        ),
    ])

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
        dcc.Store(id="live-feed-store", data=[], storage_type="session"),
        dcc.Interval(id="live-interval", interval=3000, n_intervals=0),
        dcc.Download(id="live-export"),
        dbc.Container(
            fluid=True,
            children=[
                html.Div(
                    style={"padding": "16px 8px"},
                    children=[
                        html.Div("HEALTH ANALYTICS / SYNTHETIC SANDBOX", style={"color": ACCENT, "fontSize": "0.72rem", "letterSpacing": "0.16em", "fontWeight": "700"}),
                        html.H2(APP_TITLE, style={"margin": "6px 0 2px", "color": TEXT, "fontWeight": "800"}),
                        html.Div(
                            f"Interactive analytics • Model benchmarking • Cohort discovery  |  Built by {CREATOR}",
                            style={"color": MUTED, "fontSize": "0.94rem"},
                        ),
                        dbc.Alert(
                            [
                                html.Strong("SIMULATED DATA - NOT FOR CLINICAL USE. "),
                                "No medical devices, EHR, or real patient data are connected. Demo thresholds are not clinical alarms.",
                            ],
                            color="warning",
                            className="mt-3 mb-0",
                        ),
                    ],
                ),
                dbc.Row(
                    [
                        dbc.Col(kpi_card("Patients", str(len(patients)), "Synthetic demo records", ACCENT), md=3),
                        dbc.Col(kpi_card("Models benchmarked", str(len(model_bank)), f"Best demo AUC {model_metrics.iloc[0]['ROC AUC']:.2f}", ACCENT2), md=3),
                        dbc.Col(kpi_card("Unsupervised profiles", "4", f"Silhouette {cluster_silhouette:.2f}", WARN), md=3),
                        dbc.Col(kpi_card("Data mode", "SYNTHETIC", "Not for clinical use", ACCENT), md=3),
                    ],
                    className="g-3",
                    style={"marginBottom": "12px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="patient-dd",
                                options=[{"label": p, "value": p} for p in patients["Patient ID"]],
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
                        dbc.Tab(label="Model Studio", tab_id="tab-model"),
                        dbc.Tab(label="Patient Phenotypes", tab_id="tab-phenotypes"),
                        dbc.Tab(label="Risk Explorer", tab_id="tab-cancer"),
                        dbc.Tab(label="Population Insights", tab_id="tab-pop"),
                        dbc.Tab(label="Patient Explorer", tab_id="tab-explorer"),
                        dbc.Tab(label="Clinical Report", tab_id="tab-nlp"),
                        dbc.Tab(label="Data Upload", tab_id="tab-upload"),
                        dbc.Tab(label="Collaboration", tab_id="tab-collab"),
                        dbc.Tab(label="Secure Sharing", tab_id="tab-chain"),
                    ],
                ),
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Candidate model", html_for="model-select", style={"color": MUTED}),
                                        dcc.Dropdown(
                                            id="model-select",
                                            options=[{"label": name, "value": name} for name in model_bank],
                                            value="Random Forest",
                                            clearable=False,
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Decision threshold", html_for="threshold-select", style={"color": MUTED}),
                                        dcc.Slider(
                                            id="threshold-select",
                                            min=0.1,
                                            max=0.9,
                                            step=0.05,
                                            value=0.5,
                                            marks={0.1: "0.10", 0.5: "0.50", 0.9: "0.90"},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        )
                    ],
                    id="model-controls",
                    style={"display": "none", "padding": "12px 4px 0"},
                ),
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Age cohort", html_for="age-filter", style={"color": MUTED}),
                                        dcc.Dropdown(
                                            id="age-filter",
                                            options=[
                                                {"label": "All age groups", "value": "All"},
                                                *[
                                                    {"label": group, "value": group}
                                                    for group in ["<35", "35-49", "50-64", "65+"]
                                                ],
                                            ],
                                            value="All",
                                            clearable=False,
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Smoking status", html_for="smoker-filter", style={"color": MUTED}),
                                        dcc.Dropdown(
                                            id="smoker-filter",
                                            options=[
                                                {"label": "All", "value": "all"},
                                                {"label": "Smoker", "value": "1"},
                                                {"label": "Non-smoker", "value": "0"},
                                            ],
                                            value="all",
                                            clearable=False,
                                        ),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        )
                    ],
                    id="population-filters",
                    style={"display": "none", "padding": "12px 4px 0"},
                ),
                html.Div(
                    live_monitoring_layout([]),
                    id="live-monitor-container",
                    style={"display": "none", "padding": "14px 4px"},
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
    Output("population-filters", "style"),
    Input("tabs", "active_tab"),
)
def show_population_filters(tab):
    return {"display": "block", "padding": "12px 4px 0"} if tab == "tab-pop" else {"display": "none"}

@callback(
    Output("model-controls", "style"),
    Input("tabs", "active_tab"),
)
def show_model_controls(tab):
    return {"display": "block", "padding": "12px 4px 0"} if tab == "tab-model" else {"display": "none"}

@callback(
    Output("live-monitor-container", "style"),
    Input("tabs", "active_tab"),
)
def show_live_monitor(tab):
    return {"display": "block", "padding": "14px 4px"} if tab == "tab-live" else {"display": "none"}

@callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("patient-dd", "value"),
    Input("chat-store", "data"),
    Input("age-filter", "value"),
    Input("smoker-filter", "value"),
    Input("model-select", "value"),
    Input("threshold-select", "value"),
)
def render_tab(tab, patient_id, chat_data, age_filter, smoker_filter, selected_model, threshold):
    row, risk = patient_risk_row(patient_id)

    if tab == "tab-model":
        selected_probabilities = model_probabilities[selected_model]
        selected_predictions = (selected_probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, selected_predictions, labels=[0, 1]).ravel()
        sensitivity_at_threshold = tp / (tp + fn) if tp + fn else 0
        specificity_at_threshold = tn / (tn + fp) if tn + fp else 0
        selected_metrics = model_metrics.set_index("Model").loc[selected_model]

        fig_roc = black_fig("Out-of-sample ROC comparison")
        fig_pr = black_fig("Precision-recall comparison")
        for name, probabilities in model_probabilities.items():
            fpr, tpr, _ = roc_curve(yte, probabilities)
            precision, recall, _ = precision_recall_curve(yte, probabilities)
            auc_score = model_metrics.set_index("Model").loc[name, "ROC AUC"]
            line_color = ACCENT if name == selected_model else "#667085"
            width = 3 if name == selected_model else 1.5
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line={"color": line_color, "width": width},
                name=f"{name} (AUC {auc_score:.2f})",
            ))
            fig_pr.add_trace(go.Scatter(
                x=recall, y=precision, mode="lines",
                line={"color": line_color, "width": width},
                name=name,
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line={"color": MUTED, "dash": "dash"}, name="Random baseline",
        ))
        fig_roc.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision")

        fig_matrix = black_fig(f"{selected_model} confusion matrix at {threshold:.0%} threshold")
        fig_matrix.add_trace(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted negative", "Predicted positive"],
            y=["Actual negative", "Actual positive"],
            colorscale="Viridis",
            text=[[tn, fp], [fn, tp]],
            texttemplate="%{text}",
            showscale=False,
        ))

        calibrated_true, calibrated_predicted = calibration_curve(
            yte, selected_probabilities, n_bins=5, strategy="quantile"
        )
        fig_calibration = black_fig("Probability calibration (holdout)")
        fig_calibration.add_trace(go.Scatter(
            x=calibrated_predicted,
            y=calibrated_true,
            mode="lines+markers",
            line={"color": ACCENT2, "width": 3},
            name=selected_model,
        ))
        fig_calibration.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line={"color": MUTED, "dash": "dash"}, name="Perfect calibration",
        ))
        fig_calibration.update_layout(xaxis_title="Mean predicted score", yaxis_title="Observed demo positive rate")

        model_importance = pd.Series(
            model_importance_bank[selected_model], index=feature_cols
        ).sort_values()
        fig_importance = black_fig(f"{selected_model} global feature signal")
        fig_importance.add_trace(go.Bar(
            x=model_importance.values,
            y=model_importance.index,
            orientation="h",
            marker_color=ACCENT,
            name="Relative model signal",
        ))
        fig_importance.update_layout(xaxis_title="Relative signal (not causal impact)")

        leaderboard = dash_table.DataTable(
            data=model_metrics.round(3).to_dict("records"),
            columns=[{"name": column, "id": column} for column in model_metrics.columns],
            sort_action="native",
            style_table={"overflowX": "auto", "marginTop": "8px"},
            style_cell={
                "backgroundColor": PANEL, "color": TEXT, "border": f"1px solid {BORDER}",
                "fontFamily": "Arial", "textAlign": "left", "padding": "9px",
            },
            style_header={
                "backgroundColor": "#111", "fontWeight": "bold", "border": f"1px solid {BORDER}",
            },
            style_data_conditional=[{
                "if": {"filter_query": f'{{Model}} = "{selected_model}"'},
                "backgroundColor": "#10272b",
                "color": ACCENT,
            }],
        )
        return html.Div([
            dbc.Row([
                dbc.Col(kpi_card("Selected model", selected_model, "Trained on synthetic labels", ACCENT), md=3),
                dbc.Col(kpi_card("Holdout ROC AUC", f"{selected_metrics['ROC AUC']:.3f}", "Same stratified holdout", ACCENT2), md=3),
                dbc.Col(kpi_card("Sensitivity", f"{sensitivity_at_threshold:.1%}", f"At {threshold:.0%} threshold", WARN), md=3),
                dbc.Col(kpi_card("Specificity", f"{specificity_at_threshold:.1%}", f"At {threshold:.0%} threshold", ACCENT), md=3),
            ], className="g-3 mb-2"),
            html.H5("Model leaderboard", style={"color": TEXT, "marginTop": "20px"}),
            html.Div("All candidates share the same stratified holdout. Metrics describe this simulated task only.", style={"color": MUTED}),
            leaderboard,
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_roc), md=6),
                dbc.Col(dcc.Graph(figure=fig_pr), md=6),
            ], className="g-3 mt-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_matrix), md=6),
                dbc.Col(dcc.Graph(figure=fig_calibration), md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_importance), md=12),
            ], className="g-3"),
            html.Div(
                "Threshold changes are an interactive operating-point demonstration, not clinical guidance. Calibration and feature signals are not validated on real-world data.",
                style={"color": MUTED, "padding": "8px 0 20px"},
            ),
        ])

    if tab == "tab-phenotypes":
        cluster_summary = patients.groupby("Demo Cluster", as_index=False).agg(
            patients=("Patient ID", "count"),
            mean_age=("Age", "mean"),
            mean_bmi=("BMI", "mean"),
            mean_glucose=("Glucose", "mean"),
            mean_biomarker=("Biomarker", "mean"),
            mean_demo_score=("Risk Score", "mean"),
            demo_positive_rate=("Demo Positive Label", "mean"),
        )
        fig_pca = black_fig("Unsupervised patient profiles (PCA projection)")
        colors = [ACCENT, ACCENT2, WARN, "#A78BFA"]
        for cluster_id, color in zip(sorted(patients["Demo Cluster"].unique()), colors):
            group = patients[patients["Demo Cluster"] == cluster_id]
            fig_pca.add_trace(go.Scatter(
                x=group["PC1"],
                y=group["PC2"],
                mode="markers",
                marker={"size": 10, "color": color, "opacity": 0.85, "line": {"color": TEXT, "width": 0.5}},
                text=group["Patient ID"],
                customdata=np.column_stack([group["Age"], group["BMI"], group["Glucose"], group["Biomarker"]]),
                hovertemplate="%{text}<br>Age %{customdata[0]}<br>BMI %{customdata[1]}<br>Glucose %{customdata[2]}<br>Biomarker %{customdata[3]}<extra></extra>",
                name=f"Profile {cluster_id + 1} (n={len(group)})",
            ))
        fig_pca.update_layout(xaxis_title="Principal component 1", yaxis_title="Principal component 2")

        fig_3d = go.Figure()
        for cluster_id, color in zip(sorted(patients["Demo Cluster"].unique()), colors):
            group = patients[patients["Demo Cluster"] == cluster_id]
            fig_3d.add_trace(go.Scatter3d(
                x=group["Age"],
                y=group["BMI"],
                z=group["Glucose"],
                mode="markers",
                marker={
                    "size": 5 + group["Risk Score"] * 5,
                    "color": color,
                    "opacity": 0.82,
                    "line": {"color": "#101820", "width": 0.5},
                },
                text=group["Patient ID"],
                customdata=np.column_stack([
                    group["Biomarker"], group["Risk Score"], group["Demo Cluster"] + 1,
                ]),
                hovertemplate=(
                    "%{text}<br>Age %{x}<br>BMI %{y}<br>Glucose %{z}"
                    "<br>Biomarker %{customdata[0]:.2f}"
                    "<br>Demo score %{customdata[1]:.1%}"
                    "<br>Unsupervised profile %{customdata[2]}<extra></extra>"
                ),
                name=f"Profile {cluster_id + 1}",
            ))
        fig_3d.update_layout(
            template="plotly_dark",
            paper_bgcolor=BLACK,
            plot_bgcolor=BLACK,
            font={"color": TEXT},
            scene={
                "bgcolor": BLACK,
                "xaxis": {"title": "Age (years)", "backgroundcolor": BLACK, "gridcolor": BORDER, "color": MUTED},
                "yaxis": {"title": "BMI", "backgroundcolor": BLACK, "gridcolor": BORDER, "color": MUTED},
                "zaxis": {"title": "Glucose (mg/dL)", "backgroundcolor": BLACK, "gridcolor": BORDER, "color": MUTED},
                "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
            },
            title={"text": "Interactive 3D feature space", "x": 0.02},
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
            legend={"bgcolor": BLACK},
        )

        profile_values = cluster_summary.set_index("Demo Cluster")[
            ["mean_age", "mean_bmi", "mean_glucose", "mean_biomarker"]
        ]
        profile_values.columns = ["Age", "BMI", "Glucose", "Biomarker"]
        cluster_features_for_profile = patients[["Age", "BMI", "Glucose", "Biomarker"]]
        profile_zscores = (
            profile_values - cluster_features_for_profile.mean()
        ) / cluster_features_for_profile.std()
        fig_profile = black_fig("Cluster feature profiles (population z-score)")
        fig_profile.add_trace(go.Heatmap(
            z=profile_zscores.values,
            x=["Age", "BMI", "Glucose", "Biomarker"],
            y=[f"Profile {cluster_id + 1}" for cluster_id in profile_zscores.index],
            colorscale="RdBu",
            zmid=0,
            text=np.round(profile_zscores.values, 1),
            texttemplate="%{text}",
            colorbar={"title": "Standard deviations"},
        ))
        fig_profile.update_layout(margin={"l": 60, "r": 20, "t": 55, "b": 40})

        return html.Div([
            dbc.Row([
                dbc.Col(kpi_card("Profiles found", "4", "K-means on standardized features", ACCENT), md=4),
                dbc.Col(kpi_card("Silhouette score", f"{cluster_silhouette:.3f}", "Internal separation metric", ACCENT2), md=4),
                dbc.Col(kpi_card("Patients grouped", f"{len(patients):,}", "Synthetic dataset", WARN), md=4),
            ], className="g-3 mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_3d, config={"displaylogo": False}), md=7),
                dbc.Col(dcc.Graph(figure=fig_pca, config={"displaylogo": False}), md=5),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_profile), md=12),
            ], className="g-3"),
            html.H5("Profile summary", style={"color": TEXT, "marginTop": "16px"}),
            dash_table.DataTable(
                data=cluster_summary.round(2).to_dict("records"),
                columns=[{"name": column.replace("_", " ").title(), "id": column} for column in cluster_summary.columns],
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": PANEL, "color": TEXT, "border": f"1px solid {BORDER}", "padding": "8px", "textAlign": "left"},
                style_header={"backgroundColor": "#111", "fontWeight": "bold"},
            ),
            html.Div(
                "Clusters are mathematical groupings of generated feature values, not patient diagnoses or meaningful clinical phenotypes.",
                style={"color": MUTED, "padding": "12px 0"},
            ),
        ])

    if tab == "tab-live":
        return html.Div()

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
        fig_roc = black_fig("Holdout ROC curve")
        fig_roc.add_trace(
            go.Scatter(
                x=roc_fpr,
                y=roc_tpr,
                mode="lines",
                line=dict(color=ACCENT, width=3),
                name=f"AUC {auc:.2f}",
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color=MUTED, dash="dash"),
                name="Random baseline",
            )
        )
        fig_roc.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")

        fig_matrix = black_fig("Holdout confusion matrix (threshold 0.50)")
        fig_matrix.add_trace(
            go.Heatmap(
                z=test_confusion,
                x=["Predicted low", "Predicted high"],
                y=["Actual low", "Actual high"],
                colorscale="Viridis",
                text=test_confusion,
                texttemplate="%{text}",
                showscale=False,
            )
        )
        return html.Div([
            dcc.Graph(figure=fig),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_roc), md=6),
                dbc.Col(dcc.Graph(figure=fig_matrix), md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(kpi_card("Holdout AUC", f"{auc:.2f}", "Threshold-independent ranking", ACCENT), md=4),
                dbc.Col(kpi_card("Sensitivity", f"{sensitivity:.1%}", "At a 0.50 demo threshold", ACCENT2), md=4),
                dbc.Col(kpi_card("Specificity", f"{specificity:.1%}", "At a 0.50 demo threshold", WARN), md=4),
            ], className="g-3 mb-3"),
            html.Div(f"Model score for {patient_id}: {risk:.1%}", style={"color": ACCENT, "fontSize": "1.2rem"}),
            html.Div(
                "Educational synthetic demo: the target is generated from simulated features, so these scores and holdout metrics are not clinical evidence or validated cancer predictions.",
                style={"color": MUTED},
            ),
        ])

    if tab == "tab-pop":
        cohort = patients.copy()
        if age_filter and age_filter != "All":
            cohort = cohort[cohort["Age Group"] == age_filter]
        if smoker_filter in {"0", "1"}:
            cohort = cohort[cohort["Smoker"] == int(smoker_filter)]
        if cohort.empty:
            return html.Div("No patients match the selected cohort.", style={"color": MUTED})

        age_order = ["<35", "35-49", "50-64", "65+"]
        agg = cohort.groupby("Age Group", as_index=False, observed=False).agg(
            mean_score=("Risk Score", "mean"),
            positive_rate=("Demo Positive Label", "mean"),
            patients=("Patient ID", "count"),
        )
        agg["Age Group"] = pd.Categorical(agg["Age Group"], categories=age_order, ordered=True)
        agg = agg.sort_values("Age Group")
        agg["positive_rate"] *= 100

        fig_age = black_fig("Mean model score by age group")
        fig_age.add_trace(
            go.Bar(
                x=agg["Age Group"],
                y=agg["mean_score"],
                marker_color=ACCENT,
                name="Mean model score",
                customdata=agg[["patients", "positive_rate"]],
                hovertemplate="Age %{x}<br>Mean score %{y:.1%}<br>Patients %{customdata[0]}<br>Demo positive rate %{customdata[1]:.1f}%<extra></extra>",
            )
        )
        fig_age.update_layout(yaxis_title="Mean model score")

        smoke_agg = cohort.groupby("Smoker", as_index=False).agg(
            mean_score=("Risk Score", "mean"),
            positive_rate=("Demo Positive Label", "mean"),
            patients=("Patient ID", "count"),
        )
        smoke_agg["Label"] = smoke_agg["Smoker"].map({0: "Non-smoker", 1: "Smoker"})
        smoke_agg["positive_rate"] *= 100
        fig_smoke = black_fig("Model score by smoking status")
        fig_smoke.add_trace(
            go.Bar(
                x=smoke_agg["Label"],
                y=smoke_agg["mean_score"],
                marker_color=smoke_agg["Smoker"].map({0: ACCENT2, 1: WARN}),
                name="Mean model score",
                customdata=smoke_agg[["patients", "positive_rate"]],
                hovertemplate="%{x}<br>Mean score %{y:.1%}<br>Patients %{customdata[0]}<br>Demo positive rate %{customdata[1]:.1f}%<extra></extra>",
            )
        )
        fig_smoke.update_layout(yaxis_title="Mean model score")

        heat = cohort.pivot_table(
            index="Age Group",
            columns="Smoker",
            values="Risk Score",
            aggfunc="mean",
            observed=False,
        ).reindex(index=age_order, columns=[0, 1])
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
                colorbar={"title": "Mean score"},
            )
        )
        fig_scatter = black_fig("BMI and glucose by patient")
        fig_scatter.add_trace(
            go.Scatter(
                x=cohort["BMI"],
                y=cohort["Glucose"],
                mode="markers",
                marker={
                    "size": 8,
                    "color": cohort["Risk Score"],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Model score"},
                    "line": {"width": 0.5, "color": TEXT},
                },
                text=cohort["Patient ID"],
                customdata=np.column_stack([cohort["Age"], cohort["Risk Score"]]),
                hovertemplate="%{text}<br>BMI %{x}<br>Glucose %{y}<br>Age %{customdata[0]}<br>Model score %{customdata[1]:.1%}<extra></extra>",
                name="Patients",
            )
        )
        fig_scatter.update_layout(xaxis_title="BMI", yaxis_title="Glucose")

        top_group = agg.loc[agg["mean_score"].idxmax()]
        insight = (
            f"{len(cohort)} synthetic patients in this cohort. "
            f"The highest mean model score is in age group {top_group['Age Group']} "
            f"({top_group['mean_score']:.1%}; n={int(top_group['patients'])}). "
            "These are descriptive patterns in generated demo data, not population-health estimates."
        )
        return html.Div([
            dbc.Row([
                dbc.Col(kpi_card("Patients in cohort", f"{len(cohort):,}", "After selected filters", ACCENT), md=4),
                dbc.Col(kpi_card("Mean model score", f"{cohort['Risk Score'].mean():.1%}", "Not a clinical probability", ACCENT2), md=4),
                dbc.Col(kpi_card("Demo positive labels", f"{cohort['Demo Positive Label'].mean():.1%}", "Simulated target labels", WARN), md=4),
            ], className="g-3 mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_age), md=6),
                dbc.Col(dcc.Graph(figure=fig_smoke), md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_heat), md=12),
            ], className="g-3 mt-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_scatter), md=12),
            ], className="g-3 mt-2"),
            html.Div(insight, style={"color": MUTED, "marginTop": "8px"}),
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
            f"The demo model's highest global feature importances are: {', '.join(top_factors)}. "
            f"{patient_id}'s model score is {risk:.1%}. Global feature importance does not explain "
            "the cause of an individual patient's score."
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
        return html.Div([
            html.Div(
                [
                    html.H5(f"Example note {index + 1}", style={"color": ACCENT}),
                    dbc.Textarea(
                        value=note,
                        style={"backgroundColor": PANEL, "color": TEXT, "height": "90px"},
                        readOnly=True,
                    ),
                    html.Div("Demo summary:", style={"color": ACCENT2, "fontWeight": "700", "marginTop": "8px"}),
                    html.Div(summary, style={"color": TEXT, "marginBottom": "8px"}),
                    html.Div("Rule-based interpretation:", style={"color": ACCENT, "fontWeight": "700"}),
                    html.Div(simple_clinical_interpretation(note), style={"color": TEXT}),
                ],
                style={"backgroundColor": PANEL, "padding": "14px", "border": f"1px solid {BORDER}", "borderRadius": "12px", "marginBottom": "12px"},
            )
            for index, (note, summary) in enumerate(zip(notes_df["note"], notes_df["summary"]))
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
                "You can upload a CSV for preview. Matching synthetic-demo feature columns enable illustrative model scores, not clinical predictions.",
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
            html.Div(
                "Demo messages stay in this browser session; users are not connected and messages are not shared across browsers.",
                style={"marginTop": "8px", "color": MUTED},
            ),
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

@callback(
    Output("live-feed-store", "data"),
    Input("live-interval", "n_intervals"),
    Input("patient-dd", "value"),
    State("live-feed-store", "data"),
)
def update_live_feed(tick, patient_id, readings):
    return append_live_reading(
        tick,
        patient_id,
        readings,
        patient_changed=ctx.triggered_id == "patient-dd",
    )

def make_live_chart(readings, title, value_key, color, unit):
    fig = black_fig(title)
    fig.add_trace(go.Scatter(
        x=[reading["timestamp"] for reading in readings],
        y=[reading[value_key] for reading in readings],
        mode="lines+markers",
        line={"color": color, "width": 2.5, "shape": "spline"},
        marker={"size": 6},
        name=title,
        hovertemplate="%{x}<br>%{y} " + unit + "<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="UTC timestamp",
        yaxis_title=unit,
        showlegend=False,
        margin={"l": 50, "r": 20, "t": 45, "b": 50},
        transition={"duration": 450, "easing": "cubic-in-out"},
    )
    return fig

@callback(
    Output("live-heart-chart", "figure"),
    Output("live-bp-chart", "figure"),
    Output("live-glucose-chart", "figure"),
    Output("live-spo2-chart", "figure"),
    Output("live-heart-value", "children"),
    Output("live-bp-value", "children"),
    Output("live-glucose-value", "children"),
    Output("live-spo2-value", "children"),
    Output("live-alerts", "children"),
    Output("live-readings-table", "data"),
    Output("live-updated-at", "children"),
    Output("live-selected-patient", "children"),
    Input("live-feed-store", "data"),
)
def render_live_feed(readings):
    readings = readings or []
    if not readings:
        return (no_update,) * 12
    current = readings[-1]
    flags = live_demo_flags(current)
    flag_children = (
        [html.Span(
            f"{flag} demo threshold",
            style={"display": "inline-block", "padding": "6px 10px", "margin": "3px", "borderRadius": "16px", "backgroundColor": "#4a1520", "color": WARN},
        ) for flag in flags]
        if flags else [html.Span("No demo thresholds crossed in latest reading.", style={"color": ACCENT2})]
    )
    return (
        make_live_chart(readings, "Heart rate", "Heart rate (bpm)", ACCENT, "bpm"),
        make_live_chart(readings, "Systolic blood pressure", "Systolic BP (mmHg)", ACCENT2, "mmHg"),
        make_live_chart(readings, "Glucose", "Glucose (mg/dL)", WARN, "mg/dL"),
        make_live_chart(readings, "Oxygen saturation", "SpO2 (%)", "#A78BFA", "%"),
        current["Heart rate (bpm)"],
        current["Systolic BP (mmHg)"],
        current["Glucose (mg/dL)"],
        current["SpO2 (%)"],
        flag_children,
        list(reversed(readings[-15:])),
        current["timestamp"],
        f"  •  Selected demo profile: {current['Patient ID']}",
    )

@callback(
    Output("live-export", "data"),
    Input("live-export-button", "n_clicks"),
    State("live-feed-store", "data"),
    prevent_initial_call=True,
)
def export_live_readings(n_clicks, readings):
    if not readings:
        return no_update
    return dcc.send_data_frame(
        pd.DataFrame(readings).to_csv,
        "simulated-vitals.csv",
        index=False,
    )

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
                df_clean["Demo Model Score"] = np.round(risk_new, 3)
                high = int((df_clean["Demo Model Score"] > 0.6).sum())
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
                "Feature columns match the synthetic demo model. Showing illustrative scores for matching rows:",
                style={"color": ACCENT2, "fontWeight": "700", "marginTop": "12px"},
            )
        )
        parts.append(
            html.Div(
                f"Valid rows scored: {total}. Demo scores above 60%: {high}",
                style={"color": ACCENT, "fontSize": "1.05rem"},
            )
        )
        parts.append(
            html.Div("Preview of demo model scores:", style={"color": MUTED, "marginTop": "8px"})
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