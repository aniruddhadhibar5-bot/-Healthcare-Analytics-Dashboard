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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
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
patients["Demo Positive Label"] = (prob > np.quantile(prob, 0.62)).astype(int)
patients["Risk Score"] = np.round(prob, 3)

feature_cols = ["Age", "BMI", "Blood Pressure", "Glucose", "Smoker", "Family History", "Biomarker"]
X = patients[feature_cols]
y = patients["Demo Positive Label"]

# -----------------------------
# ML model for a synthetic classification target
# -----------------------------
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=120, random_state=7)),
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
model.fit(Xtr, ytr)
auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
test_probabilities = model.predict_proba(Xte)[:, 1]
test_predictions = (test_probabilities >= 0.5).astype(int)
roc_fpr, roc_tpr, _ = roc_curve(yte, test_probabilities)
test_confusion = confusion_matrix(yte, test_predictions, labels=[0, 1])
true_negative, false_positive, false_negative, true_positive = test_confusion.ravel()
sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)
patients["Risk Score"] = np.round(model.predict_proba(X)[:, 1], 3)

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
        dbc.Container(
            fluid=True,
            children=[
                html.Div(
                    style={"padding": "16px 8px"},
                    children=[
                        html.H2(APP_TITLE, style={"margin": "0", "color": TEXT}),
                        html.Div(f"Created by {CREATOR}", style={"color": ACCENT, "marginTop": "4px"}),
                        html.Div(
                            "Interactive synthetic-data analytics demo with cohort exploration, model evaluation, and explainability.",
                            style={"color": MUTED},
                        ),
                    ],
                ),
                dbc.Row(
                    [
                        dbc.Col(kpi_card("Patients", str(len(patients)), "Synthetic demo records", ACCENT), md=3),
                        dbc.Col(kpi_card("Demo positive labels", str(int(patients["Demo Positive Label"].sum())), f"Holdout AUC {auc:.2f}", ACCENT2), md=3),
                        dbc.Col(kpi_card("Biomarker > 3.0", str(int((patients["Biomarker"] > 3.0).sum())), "Synthetic threshold count", WARN), md=3),
                        dbc.Col(kpi_card("Data mode", "DEMO", "Synthetic records only", ACCENT), md=3),
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
                        dbc.Tab(label="Risk Model Demo", tab_id="tab-cancer"),
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
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("patient-dd", "value"),
    Input("chat-store", "data"),
    Input("age-filter", "value"),
    Input("smoker-filter", "value"),
)
def render_tab(tab, patient_id, chat_data, age_filter, smoker_filter):
    row, risk = patient_risk_row(patient_id)

    if tab == "tab-live":
        fig = black_fig("Illustrative 24-hour heart-rate pattern")
        xs = np.arange(24)
        baseline = 68 + (row["Age"] >= 65) * 4 + row["Risk Score"] * 3
        ys = baseline + np.sin(np.linspace(0, 4 * np.pi, 24)) * 5
        fig.add_trace(
            go.Scatter(
                x=[f"{hour:02d}:00" for hour in xs],
                y=np.round(ys, 1),
                mode="lines+markers",
                line=dict(color=ACCENT, width=3),
                marker=dict(size=7),
                name="Illustrative heart rate",
            )
        )
        fig.update_layout(xaxis_title="Hour", yaxis_title="Heart rate (bpm)")

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
        return html.Div([
            dcc.Graph(figure=fig),
            html.Div(
                "Synthetic illustration only - this project contains no timestamped vital-sign records and is not a live patient-monitoring system.",
                style={"color": MUTED, "marginBottom": "12px"},
            ),
            table,
        ])

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