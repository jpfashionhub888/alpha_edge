@echo off
:: AlphaEdge Weekly Report — runs every Sunday
cd /d C:\Users\giris\alpha_edge
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo Weekly Report Started: %date% %time%
python critic_agent.py >> logs\weekly_report.log 2>&1
echo Weekly Report Finished: %date% %time%