🏥 Healthcare Analytics Dashboard

Explore synthetic health data with interactive charts and machine-learning demos

<img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img alt="Dash" src="https://img.shields.io/badge/Dash-Interactive%20App-008DE4?style=for-the-badge&logo=plotly&logoColor=white">
<img alt="Plotly" src="https://img.shields.io/badge/Plotly-Visualizations-3F4F75?style=for-the-badge&logo=plotly&logoColor=white">
<img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">

<br>

Model Studio · Live-feed simulator · 3D patient explorer · Cohort analytics

</div>



[!WARNING]
Synthetic-data project - not for clinical use. Patient records, labels, and streaming readings are generated. This app is not connected to a medical device, EHR, or real patient, and its model outputs and threshold flags are not validated for care decisions.

✨ Explore the dashboard

<table>
  <tr>
    <td width="50%" bgcolor="#102a43">
      <h3>🧠 Model Studio</h3>
      Compare logistic regression, random forest, and gradient boosting on a shared synthetic holdout. Explore ROC and precision-recall curves, calibration, model metrics, permutation importance, and an adjustable threshold.
    </td>
    <td width="50%" bgcolor="#172554">
      <h3>🫀 Live-feed simulator</h3>
      Watch generated heart-rate, blood-pressure, glucose, and SpO₂ readings update every three seconds. Inspect rolling charts, demo-only threshold flags, and export readings to CSV.
    </td>
  </tr>
  <tr>
    <td width="50%" bgcolor="#12332b">
      <h3>🧬 Patient profiles</h3>
      Explore unsupervised K-means groupings with PCA, silhouette score, profile summaries, and an interactive 3D feature-space view.
    </td>
    <td width="50%" bgcolor="#3b1d3b">
      <h3>📊 Population insights</h3>
      Filter synthetic cohorts by age and smoking status. Compare model scores, review heatmaps, and explore patient-level feature distributions.
    </td>
  </tr>
  <tr>
    <td width="50%" bgcolor="#3b2f1a">
      <h3>🔎 Patient explorer</h3>
      Review an individual synthetic profile alongside model-level feature summaries and interactive visualizations.
    </td>
    <td width="50%" bgcolor="#26213a">
      <h3>🗂️ More to explore</h3>
      Preview CSV files, browse example clinical notes with simple keyword summaries, and explore collaboration and audit-trail interface demonstrations.
    </td>
  </tr>
</table>

🧰 Built with

<table>
  <tr>
    <td align="center" width="140">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="42" alt="Python logo"><br>
      <b>Python</b><br><sub>Application runtime</sub>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.simpleicons.org/plotly/3F4F75" width="42" alt="Plotly logo"><br>
      <b>Dash + Plotly</b><br><sub>Interactive dashboard</sub>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.simpleicons.org/pandas/150458" width="42" alt="pandas logo"><br>
      <b>pandas</b><br><sub>Data analysis</sub>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.simpleicons.org/numpy/013243" width="42" alt="NumPy logo"><br>
      <b>NumPy</b><br><sub>Numerical computing</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="140">
      <img src="https://cdn.simpleicons.org/scikitlearn/F7931E" width="42" alt="scikit-learn logo"><br>
      <b>scikit-learn</b><br><sub>ML + clustering</sub>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.simpleicons.org/bootstrap/7952B3" width="42" alt="Bootstrap logo"><br>
      <b>Bootstrap</b><br><sub>Responsive components</sub>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.simpleicons.org/flask/000000" width="42" alt="Flask logo"><br>
      <b>Flask + Gunicorn</b><br><sub>WSGI serving</sub>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.simpleicons.org/github/181717" width="42" alt="GitHub logo"><br>
      <b>Open source</b><br><sub>GitHub project</sub>
    </td>
  </tr>
</table>

Machine-learning methods

Logistic Regression · Random Forest · Histogram Gradient Boosting · K-means · PCA · TF-IDF + Naive Bayes

🚀 Run locally

Requirements: Python 3.10 or newer.

git clone https://github.com/aniruddhadhibar5-bot/-Healthcare-Analytics-Dashboard.git healthcare-analytics-dashboard
cd healthcare-analytics-dashboard
python -m venv .venv

Activate the environment and launch:

# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py

# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
python app.py

Open the local URL printed by Dash (typically http://127.0.0.1:8050). The WSGI application entry point is app:server.

🧭 Dashboard guide

Tab
What you can explore
Live Data
Timestamped simulated vital readings, rolling charts, demo threshold flags, and CSV export
Model Studio
Candidate model benchmark, threshold exploration, ROC/PR, calibration, and metrics
Patient Phenotypes
Interactive 3D features, PCA projection, and unsupervised profile summaries
Population Insights
Age/smoking filters, cohort comparisons, and population charts
Patient Explorer
Selected synthetic profile and model feature summaries
Data Upload
CSV preview and illustrative scoring for matching demo columns

⚠️ Data, safety, and limitations

Patient records, model labels, and streaming vitals are generated; they do not represent real people or live monitoring.
The classifiers learn a synthetic target, not a clinically diagnosed outcome. Metrics, calibration, clusters, feature importance, and threshold flags are educational examples, not clinical evidence or medical advice.
The note interpretation uses simple keyword rules. It is not a medical language model or a clinical decision-support system.
The app is not suitable for clinical use or patient-care decisions. Do not upload identifiable health information.
Connecting a real EHR or medical device requires an approved data source, documented integration, access controls, privacy/security review, and clinical and regulatory validation.
