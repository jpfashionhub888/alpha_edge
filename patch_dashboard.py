# patch_dashboard2.py
# Run once: python patch_dashboard2.py

import re

with open('generate_dashboard.py', 'r', encoding='utf-8') as f:
    src = f.read()

# ── Fix 1: remove py-3\.5 CSS line entirely ──────────────────────────────
src = re.sub(r'[ \t]*\.py-3[^\n]*5[^\n]*padding-top: 12px[^\n]*\n', '', src)

# ── Fix 2: replace broken inline sector dict comprehension in f-string ────
old_sector = (
    "  const rawData = {_json.dumps([\n"
    "      {{'name': k, 'mom': v.get('momentum_21d', 0), 'flow': v.get('flow','NEUTRAL')}}\n"
    "      for k, v in sectors.items()\n"
    "  ])};"
)
new_sector = "  const rawData = {sector_chart_json};"
src = src.replace(old_sector, new_sector)

# ── Fix 3: inject sector_chart_json variable before import json block ─────
old_import = "    import json as _json\n\n    up_trend"
new_import = (
    "    import json as _json\n\n"
    "    # Pre-built outside f-string — dicts inside {{}} cause TypeError\n"
    "    sector_chart_data = [\n"
    "        {'name': k, 'mom': v.get('momentum_21d', 0), 'flow': v.get('flow', 'NEUTRAL')}\n"
    "        for k, v in sectors.items()\n"
    "    ]\n"
    "    sector_chart_json = _json.dumps(sector_chart_data)\n\n"
    "    up_trend"
)
src = src.replace(old_import, new_import)

with open('generate_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(src)

print("Patches applied")
print("  Fixed: py-3.5 CSS escape warning")
print("  Fixed: sector chart unhashable dict in f-string")
print("  Added: sector_chart_json pre-built variable")
print()
print("Now run: python generate_dashboard.py")