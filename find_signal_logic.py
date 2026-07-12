from pathlib import Path

keys = ["combined", "BUY_THRESHOLD", "compute_signal", "sizing_combined", "0.65"]

for p in Path(".").rglob("*.py"):
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        continue

    matches = []
    for i, line in enumerate(lines, 1):
        if any(k in line for k in keys):
            matches.append(f"{p}:{i}: {line}")

    if matches:
        print("\n" + "=" * 80)
        print(p)
        print("=" * 80)
        for m in matches:
            print(m)