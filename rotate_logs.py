# rotate_logs.py
# Keeps only last 30 days of logs
# Add to Task Scheduler to run weekly

import os
from pathlib import Path
from datetime import datetime, timedelta

LOG_DIR = Path("logs")
MAX_AGE_DAYS = 30
MAX_LOG_SIZE_MB = 50

def rotate():
    cutoff = datetime.now() - timedelta(days=MAX_AGE_DAYS)
    
    for log_file in LOG_DIR.glob("*.log"):
        # Check size
        size_mb = log_file.stat().st_size / (1024 * 1024)
        if size_mb > MAX_LOG_SIZE_MB:
            # Keep last 1000 lines
            lines = log_file.read_text(encoding='utf-8', 
                                        errors='ignore').splitlines()
            log_file.write_text(
                '\n'.join(lines[-1000:]),
                encoding='utf-8'
            )
            print(f"Trimmed {log_file.name}: {size_mb:.1f}MB → kept last 1000 lines")

    for log_file in LOG_DIR.glob("alphaedge_daily_*.log"):
        try:
            date_str = log_file.stem.split('_')[-1]
            file_date = datetime.strptime(date_str, '%Y%m%d')
            if file_date < cutoff:
                log_file.unlink()
                print(f"Deleted old log: {log_file.name}")
        except Exception as e:
            print(f"WARNING: could not process {log_file.name}: {e}")
            continue

    print(f"Log rotation complete: {datetime.now()}")

if __name__ == "__main__":
    rotate()