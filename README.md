# Healthcare Analytics Dashboard

Accessible AI-assisted clinical analytics dashboard with:

- Live patient data visualization  
- Cancer risk prediction (synthetic demo)  
- Population insights  
- Patient explorer with model explanations  
- Clinical note NLP summary  
- Universal CSV upload (any healthcare CSV)  
- Collaboration chat + audit trail demo  

## Tech Stack

- Python 3.10+  
- Dash + Plotly  
- scikit-learn (RandomForest for risk model)  
- pandas, numpy  

## Local Setup

```bash
# 1. Clone repo
git clone https://github.com/your-username/healthcare-dashboard.git
cd healthcare-dashboard

# 2. Create virtual environment
python -m venv venv
# On Windows:
# venvScriptsactivate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python healthcare_dashboard.py