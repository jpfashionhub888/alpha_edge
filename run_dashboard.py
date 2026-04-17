# run_dashboard.py

"""
Clean launcher for AlphaEdge Dashboard.
"""

import os
import warnings

os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

warnings.filterwarnings('ignore')

import logging
logging.getLogger('joblib').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)

from monitoring.dashboard import create_app

print("\n" + "🌐" * 25)
print("ALPHAEDGE DASHBOARD")
print("🌐" * 25)
print("\nOpen browser: http://localhost:8050")
print("Press Ctrl+C to stop\n")

app = create_app()
app.run(debug=False, host='0.0.0.0', port=8050)