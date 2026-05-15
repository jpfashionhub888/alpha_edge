@echo off
:: AlphaEdge Daily Scanner
:: Runs main.py and logs output

cd /d C:\Users\giris\alpha_edge

:: Activate virtual environment if you have one
:: If not using venv, comment out the next line
:: call venv\Scripts\activate.bat

:: Set encoding for emoji support
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

:: Run the scan and log output
echo ========================================
echo AlphaEdge Scan Started: %date% %time%
echo ========================================

python main.py >> logs\alphaedge_daily.log 2>&1

echo ========================================
echo AlphaEdge Scan Finished: %date% %time%
echo ========================================