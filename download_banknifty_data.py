"""
Download multi-year intraday historical data for Bank Nifty from Kite Connect.

Kite's historical API caps how many days of data you can pull per request,
and the cap shrinks as the candle interval gets finer. This script loops
over date chunks sized to the interval's limit and stitches results into
one CSV.

SETUP (one-time, per Kite Connect docs):
    pip install kiteconnect pandas --break-system-packages

    1. Log in to developers.kite.trade, create a Connect app (₹500/mo plan).
    2. You'll have an api_key and api_secret.
    3. Each trading day, generate a request_token by logging in via:
       https://kite.trade/connect/login?api_key=YOUR_API_KEY&v=3
       -> After login, Zerodha redirects to your app's redirect URL with
          ?request_token=... in the query string. Copy that value.
    4. Exchange request_token for an access_token (see get_access_token()
       below) — access_token is valid for that trading day only.

Fill in the CONFIG block below, then run:
    python3 download_banknifty_data.py
"""

import os
import time
import datetime as dt
import pandas as pd
from kiteconnect import KiteConnect

# ============ CONFIG — fill these in ============
# SECURITY FIX: these were hardcoded in source and committed to git —
# deep_audit.py flagged this as a live P0 (secret exposed in repo history).
# Set KITE_API_KEY / KITE_API_SECRET in config/secrets.env (or .env) instead.
# The exposed key/secret above must be rotated in the Kite Connect developer
# console regardless — removing them from this file does not remove them
# from git history.
API_KEY = os.getenv("KITE_API_KEY", "")
API_SECRET = os.getenv("KITE_API_SECRET", "")
REQUEST_TOKEN = "PASTE_TODAYS_REQUEST_TOKEN_HERE"  # regenerate daily

# What to download:
#   "index"    -> NIFTY BANK spot index (no expiry, continuous history)
#   "futures"  -> Bank Nifty continuous futures (stitched monthly contracts)
INSTRUMENT_TYPE = "index"   # change to "futures" if you trade the futures contract

INTERVAL = "5minute"        # one of: minute, 3minute, 5minute, 10minute,
                             # 15minute, 30minute, 60minute, day
YEARS_BACK = 10              # how far back to pull
OUTPUT_CSV = "banknifty_5min_10y.csv"

# Max days per request, by interval (per Kite's current documented limits)
MAX_DAYS_PER_REQUEST = {
    "minute": 60,
    "2minute": 60,
    "3minute": 100,
    "4minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "day": 2000,
}
# ==================================================


def get_access_token(kite: KiteConnect) -> str:
    """Exchange today's request_token for an access_token."""
    data = kite.generate_session(REQUEST_TOKEN, api_secret=API_SECRET)
    return data["access_token"]


def find_instrument_token(kite: KiteConnect) -> int:
    """
    Look up the instrument_token needed for historical_data() calls.
    Index and futures use different lookup paths.
    """
    if INSTRUMENT_TYPE == "index":
        instruments = kite.instruments("NSE")
        for inst in instruments:
            if inst["tradingsymbol"] == "NIFTY BANK" and inst["segment"] == "INDICES":
                return inst["instrument_token"]
        raise ValueError("Could not find NIFTY BANK index instrument token — "
                          "check tradingsymbol against kite.instruments('NSE') output.")
    else:
        # Futures: find the current-month Bank Nifty future, then use
        # continuous=1 in historical_data() to get a stitched continuous series.
        instruments = kite.instruments("NFO")
        bn_futs = [i for i in instruments
                   if i["name"] == "BANKNIFTY" and i["instrument_type"] == "FUT"]
        if not bn_futs:
            raise ValueError("No BANKNIFTY futures instruments found.")
        # nearest expiry
        bn_futs.sort(key=lambda i: i["expiry"])
        return bn_futs[0]["instrument_token"]


def download_chunked(kite: KiteConnect, instrument_token: int) -> pd.DataFrame:
    max_days = MAX_DAYS_PER_REQUEST.get(INTERVAL, 100)
    to_date = dt.datetime.now()
    from_date_overall = to_date - dt.timedelta(days=365 * YEARS_BACK)

    all_candles = []
    chunk_to = to_date
    request_count = 0

    while chunk_to > from_date_overall:
        chunk_from = max(chunk_to - dt.timedelta(days=max_days), from_date_overall)

        print(f"Fetching {chunk_from.date()} to {chunk_to.date()} ...")
        try:
            candles = kite.historical_data(
                instrument_token=instrument_token,
                from_date=chunk_from,
                to_date=chunk_to,
                interval=INTERVAL,
                continuous=(INSTRUMENT_TYPE == "futures"),
            )
            all_candles.extend(candles)
        except Exception as e:
            print(f"  Request failed for this chunk: {e}")

        chunk_to = chunk_from - dt.timedelta(days=1)

        # Kite allows 3 requests/second — stay safely under that.
        request_count += 1
        if request_count % 3 == 0:
            time.sleep(1.2)
        else:
            time.sleep(0.35)

    df = pd.DataFrame(all_candles)
    if df.empty:
        raise RuntimeError("No data returned. Check instrument_token, date range, "
                            "and that your access_token is valid for today.")
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def main():
    kite = KiteConnect(api_key=API_KEY)
    access_token = get_access_token(kite)
    kite.set_access_token(access_token)
    print(f"Authenticated. Access token valid for today only.")

    instrument_token = find_instrument_token(kite)
    print(f"Using instrument_token={instrument_token} ({INSTRUMENT_TYPE})")

    df = download_chunked(kite, instrument_token)

    earliest = df["date"].min()
    latest = df["date"].max()
    print(f"\nGot {len(df)} candles, from {earliest} to {latest}")
    if (latest.year - earliest.year) < YEARS_BACK - 1:
        print(f"NOTE: you asked for {YEARS_BACK} years but only got "
              f"~{latest.year - earliest.year} years of data. This is expected "
              f"if minute-level data isn't available that far back for this "
              f"instrument — verify against Kite's actual retention for it.")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
