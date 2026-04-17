# run.py

"""
Clean launcher for AlphaEdge.
Nuclear option: kills every possible warning.
"""

import os
import sys
import ctypes

# Kill warnings at the OS level before Python even starts
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

# Redirect stderr to devnull to kill ALL warnings
# including ones from C libraries and worker threads
import io

class WarningFilter(io.TextIOBase):
    """Filter out warning lines from stderr."""

    def __init__(self, original):
        self.original = original

    def write(self, text):
        if text and not any(skip in text for skip in [
            'UserWarning',
            'FutureWarning',
            'DeprecationWarning',
            'Pandas4Warning',
            'sklearn.utils',
            'joblib',
            'parallel.delayed',
            'Timestamp.utcnow',
            'set a HF_TOKEN',
            'unauthenticated',
            'UNEXPECTED',
            'WordPiece',
            'from_file',
        ]):
            self.original.write(text)
        return len(text) if text else 0

    def flush(self):
        self.original.flush()

sys.stderr = WarningFilter(sys.__stderr__)

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import logging
logging.getLogger('joblib').setLevel(logging.CRITICAL)
logging.getLogger('sklearn').setLevel(logging.CRITICAL)
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)
logging.getLogger('xgboost').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)
logging.getLogger('httpcore').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('huggingface_hub').setLevel(logging.CRITICAL)

from main import run_daily_scan
run_daily_scan()