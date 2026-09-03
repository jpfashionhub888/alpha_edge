#!/usr/bin/env bash
# AlphaEdge VPS audit — read-only. Makes no changes.
# Usage:  bash vps_audit.sh 2>&1 | tee /tmp/vps_audit.txt
cd ~/alpha_edge || exit 1
export GIT_PAGER=cat
p(){ printf '\n===== %s =====\n' "$1"; }

p "IDENTITY"
hostname; pwd; date -u; python3 -V
git --no-pager log -1 --format='HEAD %h %ad %s' --date=iso

p "GIT DIVERGENCE"
git --no-pager log --oneline 84bda84..HEAD --no-merges 2>/dev/null | head -20
echo "-- uncommitted --"; git status --porcelain | head -20
echo "-- remotes (token check) --"; git remote -v | sed -E 's/(ghp_|github_pat_)[A-Za-z0-9_]+/[REDACTED]/g'

p "STATE FILES TRACKED DESPITE .gitignore"
git ls-files | grep -E 'logs/.*\.json' | head -20
echo "-- who commits them --"
git --no-pager log --oneline -3 -- logs/latest_signals.json
crontab -l 2>/dev/null | grep -iE 'git|scan|main.py' | head
grep -rn 'git add\|git commit\|git push' --include=*.py --include=*.sh . 2>/dev/null \
  | grep -v venv | grep -v '/alpha_edge/alpha_edge/' | head

p "COMPILE CHECK (blank = all good)"
python3 - <<'PY'
import os,py_compile
skip={'venv','.git','__pycache__','node_modules','.pytest_cache','catboost_info','saved_models','alpha_edge'}
bad=[];n=0
for r,d,f in os.walk('.'):
    d[:]=[x for x in d if x not in skip]
    for fn in f:
        if fn.endswith('.py'):
            p=os.path.join(r,fn);n+=1
            try: py_compile.compile(p,doraise=True,cfile='/tmp/c.pyc')
            except Exception as e: bad.append((p,str(e).split('\n')[0][:100]))
print(f'files={n} broken={len(bad)}')
for x in bad: print(' ',*x)
PY

p "UNDEFINED NAMES (real crashers)"
python3 -m pyflakes . 2>/dev/null | grep -E 'undefined name|syntax' | grep -v venv | head -20 \
  || echo "(pyflakes not installed: pip3 install pyflakes)"

p "SPECIFIC DEFECTS FROM WINDOWS AUDIT"
echo "-- main.py __main__ guard --";        grep -n "__main__" main.py || echo "  ABSENT (silent no-op!)"
echo "-- main.py last 3 lines --";          tail -3 main.py
echo "-- dashboard logger defined? --";     grep -cn "^logger\s*=\|logger = logging" monitoring/dashboard.py
echo "-- crash recovery class --";          grep -n "^class" tests/test_crash_recovery.py
echo "-- telegram_bot complete? --";        tail -2 monitoring/telegram_bot.py
echo "-- veto_agent bad_value fix? --";     grep -c "bad_value" veto_agent.py
echo "-- feature_engine returns? --";       tail -3 data/feature_engine.py

p "STALE STATE"
echo "-- circuit breaker --"; cat logs/circuit_breaker.json 2>/dev/null
echo "-- kill switch --";     cat logs/kill_switch.json 2>/dev/null
echo "-- newest logs --";     ls -lt --time-style=+%m-%d_%H:%M logs/*.json 2>/dev/null | head -8

p "SERVICE HEALTH"
systemctl is-active alphaedge alphaedge-dashboard gateio_live 2>/dev/null
systemctl list-timers --no-pager 2>/dev/null | grep -i alpha | head
journalctl -u alphaedge --since '48 hours ago' -p err --no-pager 2>/dev/null | tail -15

p "PERF: last scan duration"
grep -hoE 'SCAN COMPLETE|complete .* [0-9]+ signals|Duration.*' logs/alphaedge_daily.log 2>/dev/null | tail -5
echo "-- watchlist size --"
python3 -c "
import re;s=open('main.py',encoding='utf-8',errors='ignore').read()
m=re.search(r'def get_full_watchlist.*?return \[(.*?)\]',s,re.S)
print(len(re.findall(r\"'[A-Z.-]+'\",m.group(1))) if m else 'n/a')"

p "SILENT EXCEPT HANDLERS"
python3 - <<'PY'
import os,ast
skip={'venv','.git','__pycache__','.pytest_cache','catboost_info','saved_models','alpha_edge'}
from collections import Counter
c=Counter()
for r,d,f in os.walk('.'):
    d[:]=[x for x in d if x not in skip]
    for n in f:
        if not n.endswith('.py'): continue
        try: t=ast.parse(open(os.path.join(r,n),encoding='utf-8',errors='ignore').read())
        except Exception: continue
        for nd in ast.walk(t):
            if isinstance(nd,ast.ExceptHandler):
                s=ast.dump(ast.Module(body=nd.body,type_ignores=[])).lower()
                if not any(k in s for k in ('logger','logging','print','raise','alert')):
                    c[os.path.join(r,n)]+=1
print('total:',sum(c.values()))
for k,v in c.most_common(8): print(f'  {v:3d} {k}')
PY

p "TESTS"
python3 -m pytest tests/ --collect-only -q 2>&1 | tail -5

p "DONE"
