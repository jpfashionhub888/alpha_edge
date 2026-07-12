# options_analyzer.py
# AlphaEdge — Options Flow Intelligence (Layer 10)
#
# Uses FREE data only (yfinance) — no paid API required.
#
# What it calculates:
#   Put/Call Ratio (PCR)     — bearish signal if PCR > 1.3
#   IV Rank (IVR)            — high IV = expensive options = caution
#   Unusual Activity         — volume > 3× OI on any strike
#
# Output:
#   options_score in [-0.30, +0.30]
#   Positive = options market is bullish (boost signal)
#   Negative = options market is bearish (penalty)
#
# Integration:
#   Called from main.py as Layer 10 after AI Veto.
#   options_score is added to the raw signal score (capped).

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    logger.warning("yfinance not installed — options_analyzer disabled. "
                   "Run: pip install yfinance")
    _YF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  THRESHOLDS
# ─────────────────────────────────────────────────────────────
PCR_BEARISH_THRESHOLD  = 1.30   # PCR above this → bearish
PCR_BULLISH_THRESHOLD  = 0.70   # PCR below this → bullish
IVR_HIGH_THRESHOLD     = 70.0   # IV rank above this → expensive, caution
IVR_LOW_THRESHOLD      = 30.0   # IV rank below this → cheap, ok
UNUSUAL_VOL_OI_RATIO   = 3.0    # volume > 3× OI → unusual activity
MIN_VOLUME_THRESHOLD   = 100    # ignore strikes with tiny volume
SCORE_CAP              = 0.30   # max absolute score output


class OptionsAnalyzer:
    """
    Fetches and interprets options chain data using yfinance.
    Produces a float options_score per symbol.

    score > 0 → options market bullish  (add to signal)
    score < 0 → options market bearish  (penalise signal)
    score = 0 → neutral or data unavailable
    """

    def __init__(self, cache_minutes: int = 30):
        """
        cache_minutes: how long to cache options data before re-fetching.
        Set to 0 to disable caching (useful for testing).
        """
        self._cache: dict = {}          # symbol → (timestamp, result)
        self._cache_ttl = timedelta(minutes=cache_minutes)

    # ─────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────

    def get_options_score(self, symbol: str) -> float:
        """
        Main entry point.  Returns options_score in [-0.30, +0.30].
        Returns 0.0 on any error so the rest of the pipeline is unaffected.
        _YF_AVAILABLE is NOT checked here so that tests can patch _fetch_chain
        and exercise the scoring logic without yfinance installed.
        """
        # Cache hit
        cached = self._cache.get(symbol)
        if cached:
            ts, result = cached
            if datetime.now() - ts < self._cache_ttl:
                logger.debug(f"{symbol}: options cache hit")
                return result

        try:
            result = self._compute_score(symbol)
        except Exception as e:
            logger.warning(f"{symbol}: options_analyzer error — {e}")
            result = 0.0

        self._cache[symbol] = (datetime.now(), result)
        return result

    def get_full_analysis(self, symbol: str) -> dict:
        """
        Returns the detailed breakdown dict (for logging / dashboard).
        """
        if not _YF_AVAILABLE:
            return self._empty_analysis(symbol, "yfinance not available")

        try:
            return self._full_analysis(symbol)
        except Exception as e:
            logger.warning(f"{symbol}: options full analysis error — {e}")
            return self._empty_analysis(symbol, str(e))

    # ─────────────────────────────────────────────────────────
    #  INTERNALS
    # ─────────────────────────────────────────────────────────

    def _fetch_chain(self, symbol: str):
        """
        Fetch the nearest-expiry options chain via yfinance.
        Returns (calls_df, puts_df, expiry_str) or raises on failure.
        """
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            raise ValueError(f"No options expirations found for {symbol}")

        # Pick the nearest expiry that is at least 7 days out
        # to avoid expiry-week noise
        today = datetime.now().date()
        chosen = None
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            if (exp_date - today).days >= 7:
                chosen = exp
                break

        if chosen is None:
            chosen = expirations[0]   # fallback to nearest

        chain  = ticker.option_chain(chosen)
        calls  = chain.calls
        puts   = chain.puts
        return calls, puts, chosen

    def _put_call_ratio(self, calls, puts) -> float | None:
        """
        PCR = total put volume / total call volume.
        Uses open interest as a fallback when volume is all zeros.
        """
        call_vol = calls['volume'].fillna(0).sum()
        put_vol  = puts['volume'].fillna(0).sum()

        if call_vol == 0 and put_vol == 0:
            # Fall back to open interest
            call_oi = calls['openInterest'].fillna(0).sum()
            put_oi  = puts['openInterest'].fillna(0).sum()
            if call_oi == 0:
                return None
            return put_oi / call_oi

        if call_vol == 0:
            return None

        return put_vol / call_vol

    def _iv_rank(self, calls, puts) -> float | None:
        """
        IV Rank = (current ATM IV - 52-week IV low) /
                  (52-week IV high - 52-week IV low) × 100

        We approximate 52-week range from all strikes in today's chain.
        This is not a perfect IVR (which ideally needs historical IV data),
        but it is free and directionally accurate.
        """
        try:
            all_iv = (
                list(calls['impliedVolatility'].dropna()) +
                list(puts['impliedVolatility'].dropna())
            )
            if len(all_iv) < 4:
                return None

            iv_min = min(all_iv)
            iv_max = max(all_iv)
            iv_mid = sorted(all_iv)[len(all_iv) // 2]   # median ≈ ATM IV

            if iv_max == iv_min:
                return 50.0   # flat IV surface → neutral

            return (iv_mid - iv_min) / (iv_max - iv_min) * 100
        except Exception:
            return None

    def _unusual_activity(self, calls, puts) -> tuple[bool, str]:
        """
        Detect unusual options activity:
        volume > UNUSUAL_VOL_OI_RATIO × open_interest on any strike
        AND volume > MIN_VOLUME_THRESHOLD.

        Returns (is_unusual, direction) where direction is 'call' or 'put'.
        """
        unusual_calls = False
        unusual_puts  = False

        for _, row in calls.iterrows():
            vol = row.get('volume', 0) or 0
            oi  = row.get('openInterest', 0) or 0
            if oi > 0 and vol >= MIN_VOLUME_THRESHOLD and vol >= UNUSUAL_VOL_OI_RATIO * oi:
                unusual_calls = True
                break

        for _, row in puts.iterrows():
            vol = row.get('volume', 0) or 0
            oi  = row.get('openInterest', 0) or 0
            if oi > 0 and vol >= MIN_VOLUME_THRESHOLD and vol >= UNUSUAL_VOL_OI_RATIO * oi:
                unusual_puts = True
                break

        if unusual_calls and not unusual_puts:
            return True, 'call'
        elif unusual_puts and not unusual_calls:
            return True, 'put'
        elif unusual_calls and unusual_puts:
            return True, 'mixed'
        return False, 'none'

    def _compute_score(self, symbol: str) -> float:
        """
        Core scoring logic. Combines PCR, IVR, and unusual activity.

        Scoring table:
            PCR < 0.70            →  +0.10  (bullish, everyone buying calls)
            PCR > 1.30            →  -0.10  (bearish, heavy put buying)
            IVR > 70              →  -0.05  (expensive options = smart money hedging)
            IVR < 30              →  +0.05  (cheap options = low fear)
            Unusual call activity →  +0.15  (large player buying calls)
            Unusual put activity  →  -0.15  (large player buying puts / hedging)
        """
        calls, puts, expiry = self._fetch_chain(symbol)

        score = 0.0
        components = {}

        # ── Put/Call Ratio ──
        pcr = self._put_call_ratio(calls, puts)
        if pcr is not None:
            if pcr < PCR_BULLISH_THRESHOLD:
                score += 0.10
                components['pcr'] = f"{pcr:.2f} (bullish +0.10)"
            elif pcr > PCR_BEARISH_THRESHOLD:
                score -= 0.10
                components['pcr'] = f"{pcr:.2f} (bearish -0.10)"
            else:
                components['pcr'] = f"{pcr:.2f} (neutral)"
        else:
            components['pcr'] = "N/A"

        # ── IV Rank ──
        ivr = self._iv_rank(calls, puts)
        if ivr is not None:
            if ivr > IVR_HIGH_THRESHOLD:
                score -= 0.05
                components['ivr'] = f"{ivr:.1f}% (expensive -0.05)"
            elif ivr < IVR_LOW_THRESHOLD:
                score += 0.05
                components['ivr'] = f"{ivr:.1f}% (cheap +0.05)"
            else:
                components['ivr'] = f"{ivr:.1f}% (neutral)"
        else:
            components['ivr'] = "N/A"

        # ── Unusual Activity ──
        is_unusual, direction = self._unusual_activity(calls, puts)
        if is_unusual:
            if direction == 'call':
                score += 0.15
                components['unusual'] = "CALL sweep +0.15"
            elif direction == 'put':
                score -= 0.15
                components['unusual'] = "PUT sweep -0.15"
            else:
                components['unusual'] = "mixed (neutral)"
        else:
            components['unusual'] = "none"

        # Clamp to [-SCORE_CAP, +SCORE_CAP]
        score = max(-SCORE_CAP, min(SCORE_CAP, score))

        logger.info(
            f"{symbol} options [{expiry}]: "
            f"score={score:+.2f} | {components}"
        )
        return round(score, 3)

    def _full_analysis(self, symbol: str) -> dict:
        """Returns full dict with all computed components."""
        calls, puts, expiry = self._fetch_chain(symbol)

        pcr          = self._put_call_ratio(calls, puts)
        ivr          = self._iv_rank(calls, puts)
        unusual, dir = self._unusual_activity(calls, puts)
        score        = self._compute_score.__wrapped__(self, symbol) \
                       if hasattr(self._compute_score, '__wrapped__') \
                       else self.get_options_score(symbol)

        return {
            'symbol'          : symbol,
            'expiry'          : expiry,
            'put_call_ratio'  : round(pcr, 3) if pcr is not None else None,
            'iv_rank'         : round(ivr, 1) if ivr is not None else None,
            'unusual_activity': unusual,
            'unusual_direction': dir,
            'options_score'   : score,
            'timestamp'       : datetime.now().isoformat(),
        }

    @staticmethod
    def _empty_analysis(symbol: str, reason: str) -> dict:
        return {
            'symbol'           : symbol,
            'expiry'           : None,
            'put_call_ratio'   : None,
            'iv_rank'          : None,
            'unusual_activity' : False,
            'unusual_direction': 'none',
            'options_score'    : 0.0,
            'error'            : reason,
            'timestamp'        : datetime.now().isoformat(),
        }


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import json

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s  %(message)s')

    test_symbols = ['AAPL', 'NVDA', 'SPY', 'QQQ']
    analyzer = OptionsAnalyzer(cache_minutes=0)

    print("\n" + "=" * 60)
    print("  OPTIONS FLOW ANALYSIS")
    print("=" * 60)

    for sym in test_symbols:
        analysis = analyzer.get_full_analysis(sym)
        score    = analysis['options_score']
        pcr      = analysis['put_call_ratio']
        ivr      = analysis['iv_rank']
        unusual  = analysis['unusual_activity']
        direction= analysis['unusual_direction']
        expiry   = analysis['expiry']

        emoji = "🟢" if score > 0 else ("🔴" if score < 0 else "⚪")
        print(f"\n  {emoji} {sym:<6}  score={score:+.3f}  expiry={expiry}")
        print(f"     PCR: {pcr}  |  IVR: {ivr}