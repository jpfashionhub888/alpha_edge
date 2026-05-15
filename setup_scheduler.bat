@echo off
:: AlphaEdge Task Scheduler Setup
:: Run this file ONCE as Administrator

echo Setting up AlphaEdge scheduled tasks...

:: ── Delete existing tasks if any ──────────────────────────────
schtasks /delete /tn "AlphaEdge_945AM" /f 2>nul
schtasks /delete /tn "AlphaEdge_1230PM" /f 2>nul
schtasks /delete /tn "AlphaEdge_415PM" /f 2>nul

:: ── 9:45 AM ET = 7:15 PM IST ──────────────────────────────────
:: Convert: ET is IST - 9:30
:: 9:45 AM ET = 19:15 IST
schtasks /create ^
  /tn "AlphaEdge_945AM" ^
  /tr "C:\Users\giris\alpha_edge\run_alphaedge.bat" ^
  /sc daily ^
  /st 19:15 ^
  /mo 1 ^
  /ru "%USERNAME%" ^
  /f

:: ── 12:30 PM ET = 10:00 PM IST ────────────────────────────────
schtasks /create ^
  /tn "AlphaEdge_1230PM" ^
  /tr "C:\Users\giris\alpha_edge\run_alphaedge.bat" ^
  /sc daily ^
  /st 22:00 ^
  /mo 1 ^
  /ru "%USERNAME%" ^
  /f

:: ── 4:15 PM ET = 1:45 AM IST (next day) ──────────────────────
schtasks /create ^
  /tn "AlphaEdge_415PM" ^
  /tr "C:\Users\giris\alpha_edge\run_alphaedge.bat" ^
  /sc daily ^
  /st 01:45 ^
  /mo 1 ^
  /ru "%USERNAME%" ^
  /f

echo.
echo ✅ AlphaEdge tasks created:
echo    19:15 IST  → 9:45 AM ET scan
echo    22:00 IST  → 12:30 PM ET scan
echo    01:45 IST  → 4:15 PM ET scan
echo.
echo Verify with: schtasks /query /tn "AlphaEdge_945AM"
:: ── Sunday Weekly Report — 9:00 AM IST ───────────────────────
schtasks /delete /tn "AlphaEdge_WeeklyReport" /f 2>nul

schtasks /create ^
  /tn "AlphaEdge_WeeklyReport" ^
  /tr "C:\Users\giris\alpha_edge\run_weekly_report.bat" ^
  /sc weekly ^
  /d SUN ^
  /st 09:00 ^
  /ru "%USERNAME%" ^
  /f

echo    09:00 IST Sunday → Weekly critic report
pause