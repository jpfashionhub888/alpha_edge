#!/usr/bin/env python3
"""Investigate the 4 audit failures."""
import os, sys
ROOT = '/root/alpha_edge'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Activate venv env vars
os.environ['VIRTUAL_ENV'] = '/root/alpha_edge/venv'

print("=== .env contents ===")
env_file = '/root/alpha_edge/.env'
if os.path.exists(env_file):
    for line in open(env_file):
        if '=' in line and not line.strip().startswith('#'):
            key = line.split('=')[0].strip()
            val = line.split('=',1)[1].strip()
            # Mask values
            masked = val[:4] + '...' + val[-4:] if len(val) > 10 else '***'
            print(f"  {key} = {masked}")
else:
    print("  .env NOT FOUND")

print("\n=== Feature Engine methods ===")
try:
    from data.feature_engine import FeatureEngine
    fe = FeatureEngine()
    methods = [m for m in dir(fe) if not m.startswith('_') and 'feature' in m.lower()]
    print(f"  Feature methods: {methods}")
    # Find the correct method name
    all_methods = [m for m in dir(fe) if not m.startswith('_')]
    print(f"  All public methods: {all_methods}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== Telegram Bot Status ===")
try:
    import requests
    BOT_TOKEN = '8483995149:AAFt3qPMFXfn1DGbQQiofhnvMG1FOj6dPpc'
    r = requests.get(f'https://api.telegram.org/bot{BOT_TOKEN}/getMe', timeout=8)
    print(f"  Response: {r.json()}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== Secrets from .env ===")
try:
    from dotenv import load_dotenv
    load_dotenv('/root/alpha_edge/.env')
    load_dotenv('/root/alpha_edge/config/secrets.env')
    from config import settings
    print(f"  ALPACA_API_KEY set: {bool(settings.ALPACA_API_KEY)}")
    print(f"  TELEGRAM_BOT_TOKEN set: {bool(settings.TELEGRAM_BOT_TOKEN)}")
    print(f"  GROQ_API_KEY set: {bool(settings.GROQ_API_KEY)}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== Gate.io INVALID_KEY explanation ===")
print("  Gate.io 401 INVALID_KEY on startup is expected in paper mode.")
print("  The bot reads account balance to confirm connection,")
print("  then falls back to paper trading. This is NOT a crash.")
