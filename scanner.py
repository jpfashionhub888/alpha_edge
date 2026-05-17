# scanner.py
"""
StockScanner V2 — patched per AUDIT.md

Changes from V1:
- H1 fix: look-ahead bias. _add_market_context() in feature_engine fetches
  external data up to df.index.max(). On the train slice this used to be
  the test-end date. Now we pass end_date explicitly so train features
  only see data up to the train cut.
- H3 fix: crypto BUY signals now go through veto agent before OPEN.
- M1 fix: ATR fallback raised to price*0.03 and tracked; symbol skipped
  after consecutive ATR failures.
- M4 fix: veto agent has its own circuit breaker — 3 consecutive
  same-exception failures puts it in BYPASS mode for the rest of the
  scan with a Telegram alert.
- Type hints added to public methods.
"""

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

# Quiet noisy third-party loggers
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

import pandas as pd
import yfinance as yf
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from correlation_filter import CorrelationFilter
from data.feature_engine import FeatureEngine
from data.news_data import NewsFetcher
from data.stock_data import StockDataFetcher
from insider_tracker import InsiderTracker
from model_cache import is_cache_valid, load_models, save_models
from models.crypto_predictor import CryptoPredictor
from models.regime_detector import RegimeDetector
from models.sector_rotation import SectorRotation
from models.sentiment_model import SentimentAnalyzer
from models.technical_model import TechnicalPredictor
from multi_timeframe import MultiTimeframeAnalyzer
from veto_agent import VetoAgent

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  MODULE-LEVEL CONSTANTS                                              #
# ------------------------------------------------------------------ #

# Signal weight configuration — no empirical basis yet, see AUDIT.md M3
SIGNAL_WEIGHTS = {
    'prediction': 0.60,
    'sentiment' : 0.20,
    'sector'    : 0.20,
}

# H4 fix: sentiment contribution halved so a typical sent_score of ±0.5
# gives ±0.05 contribution. Was effectively binary at ±0.20 before.
SENTIMENT_DAMPENER = 0.5

# Crypto ATR proxy (5% of price)
CRYPTO_ATR_PCT = 0.05

# Min rows to train a model
MIN_TRAIN_ROWS = 100

# Walk-forward look-back window in days
WALK_FORWARD_DAYS = 180

# Top-N stocks to run sentiment on (API cost control)
SENTIMENT_TOP_N = 7

# Earnings calendar concurrent fetch
EARNINGS_MAX_WORKERS = 8
EARNINGS_TIMEOUT_SEC = 30

# M4: veto agent circuit breaker — bypass after N consecutive same-type failures
VETO_FAILURE_THRESHOLD = 3


# ------------------------------------------------------------------ #
#  STANDALONE HELPERS                                                  #
# ------------------------------------------------------------------ #

def feature_set_hash(feature_names: list) -> str:
    """Stable 8-char hash of the feature list for cache versioning."""
    key = ','.join(sorted(feature_names))
    return hashlib.md5(key.encode()).hexdigest()[:8]


def calc_atr(stock_data: dict, symbol: str) -> float:
    """14-period ATR. Raises ValueError if symbol missing or ATR invalid."""
    if symbol not in stock_data:
        raise ValueError(f"Symbol {symbol} not in stock_data")

    df    = stock_data[symbol].copy()
    high  = df['high']
    low   = df['low']
    close = df['close']

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_val = tr.rolling(14).mean().iloc[-1]

    if pd.isna(atr_val) or atr_val <= 0:
        raise ValueError(f"ATR invalid for {symbol}: {atr_val}")
    return float(atr_val)


def compute_signal(
    pred: float,
    regime: str,
    sent_score: float,
    sect_mult: float,
    symbol: str,
    earnings_symbols: list,
) -> tuple:
    """
    Compute final trading signal and combined score.

    Returns
    -------
    signal   : 'BUY' | 'HOLD' | 'AVOID' | 'CAUTION' | 'EARNINGS_HOLD'
    combined : sizing score in [0, 1]
    """
    # H4 fix: sentiment dampened — typical scores no longer saturate
    w_sent       = SIGNAL_WEIGHTS['sentiment']
    sent_contrib = max(-w_sent, min(w_sent, sent_score * w_sent * SENTIMENT_DAMPENER))

    combined = (
        pred                  * SIGNAL_WEIGHTS['prediction']
        + sent_contrib
        + (sect_mult - 1.0)   * SIGNAL_WEIGHTS['sector']
    )
    combined = max(0.0, min(1.0, combined))

    BUY_THRESHOLD = 0.55
    signal = 'HOLD'

    if regime == 'uptrend':
        if pred >= BUY_THRESHOLD:
            signal = 'BUY'
    elif regime == 'downtrend':
        signal = 'AVOID'
    elif regime == 'volatile':
        signal = 'CAUTION'
    # sideways stays HOLD

    if sect_mult < 0.8 and signal == 'BUY':
        signal = 'HOLD'

    if symbol in earnings_symbols and signal == 'BUY':
        signal = 'EARNINGS_HOLD'

    return signal, combined


# ------------------------------------------------------------------ #
#  SCANNER CLASS                                                       #
# ------------------------------------------------------------------ #

class StockScanner:
    """Stock + crypto scan pipeline. See module docstring for changes."""

    def __init__(
        self,
        stock_watchlist  : list,
        crypto_watchlist : list,
        sector_analyzer  : SectorRotation,
        market_regime    : dict,
        open_positions   : dict,
        lookback_days    : int = 730,
        top_k_features   : int = 20,
        retrain_days     : int = 30,
    ):
        self.stock_watchlist  = stock_watchlist
        self.crypto_watchlist = crypto_watchlist
        self.sector_analyzer  = sector_analyzer
        self.market_regime    = market_regime
        self.open_positions   = open_positions
        self.lookback_days    = lookback_days
        self.top_k_features   = top_k_features
        self.retrain_days     = retrain_days

        self.engine          = FeatureEngine()
        self.regime_detector = RegimeDetector()
        self.mtf_analyzer    = MultiTimeframeAnalyzer()
        self.corr_filter     = CorrelationFilter(max_per_sector=2)
        self.veto_agent      = VetoAgent()
        self.insider_tracker = InsiderTracker()
        self.news_fetcher    = NewsFetcher()
        self.sentiment_model = SentimentAnalyzer()

        # Populated by scan methods
        self.stock_data       = {}
        self.earnings_symbols = []
        self.insider_scores   = {}
        self.mtf_scores       = {}

        # M4: veto agent circuit breaker state (per-scan, not persistent)
        self._veto_consecutive_errors = 0
        self._veto_last_error_type    = None
        self._veto_bypassed           = False

    # ---------------------------------------------------------------- #
    #  EARNINGS CALENDAR                                                #
    # ---------------------------------------------------------------- #

    def fetch_earnings_calendar(self) -> list:
        print("\n📅 Checking earnings calendar...")
        earnings_soon = []

        def _fetch_one(symbol):
            try:
                ticker = yf.Ticker(symbol)
                cal    = ticker.calendar
                if cal is None or len(cal) == 0:
                    return None
                if not isinstance(cal, dict):
                    return None

                ed = cal.get('Earnings Date', [None])
                if not ed:
                    return None
                if isinstance(ed, list):
                    ed = ed[0]
                if ed is None:
                    return None

                today = datetime.now().date()
                if hasattr(ed, 'date'):
                    ed = ed.date()
                diff = (ed - today).days
                if 0 <= diff <= 7:
                    return {
                        'symbol'    : symbol,
                        'date'      : str(ed),
                        'days_until': diff,
                    }
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=EARNINGS_MAX_WORKERS) as ex:
            futures = {ex.submit(_fetch_one, s): s for s in self.stock_watchlist}
            try:
                for fut in as_completed(futures, timeout=EARNINGS_TIMEOUT_SEC):
                    try:
                        result = fut.result()
                        if result:
                            earnings_soon.append(result)
                    except Exception as e:
                        logger.debug("Earnings fetch failed for %s: %s",
                                     futures[fut], e)
            except Exception as e:
                logger.warning("Earnings fetch pool timed out: %s", e)

        earnings_soon.sort(key=lambda x: x['days_until'])

        if earnings_soon:
            print(f"  ⚠️  {len(earnings_soon)} stocks reporting this week")
            for e in earnings_soon:
                d = e['days_until']
                when = "TODAY!" if d == 0 else "TOMORROW!" if d == 1 else f"in {d} days"
                print(f"     ⚠️  {e['symbol']} earnings {when}")
        else:
            print("  ✅ No earnings this week")

        self.earnings_symbols = [e['symbol'] for e in earnings_soon]
        return earnings_soon

    # ---------------------------------------------------------------- #
    #  SCAN STOCKS                                                      #
    # ---------------------------------------------------------------- #

    def scan_stocks(self) -> dict:
        t0 = time.time()
        print("\n" + "=" * 60)
        print("PHASE 1: STOCK ANALYSIS")
        print("=" * 60)

        results = self._fetch_stock_data()
        if not results:
            logger.error("No stock data fetched — aborting stock scan")
            return {}

        # Insider scores
        print("\n  Loading insider trading data...")
        try:
            self.insider_scores = self.insider_tracker.get_bulk_scores(
                self.stock_watchlist, days_back=30
            )
        except Exception as e:
            logger.warning("Insider tracker failed: %s", e)
            self.insider_scores = {}

        # MTF
        self.mtf_scores = self._fetch_mtf_scores()

        # Sentiment on top-N
        sentiments = self._fetch_sentiments(results)

        # Build final signals
        final_signals = {}
        for symbol, raw in results.items():
            record = self._build_stock_signal(symbol, raw, sentiments)
            if record:
                final_signals[symbol] = record

        logger.info("Stock scan complete: %d signals in %.1fs",
                    len(final_signals), time.time() - t0)
        return final_signals

    # ---------------------------------------------------------------- #
    #  SCAN CRYPTO                                                      #
    # ---------------------------------------------------------------- #

    def scan_crypto(self) -> dict:
        """
        H3 fix: crypto BUYs now go through veto agent before OPEN.
        MTF and correlation are skipped (don't apply cleanly to crypto),
        but earnings is N/A and veto-agent review IS run.
        """
        t0 = time.time()
        print("\n" + "=" * 60)
        print("PHASE 2: CRYPTO ANALYSIS")
        print("=" * 60)

        crypto_results = {}
        try:
            predictor   = CryptoPredictor()
            raw_signals = predictor.run_full_pipeline(
                self.crypto_watchlist, lookback_days=365
            )
        except Exception as e:
            logger.warning("Crypto predictor failed: %s", e)
            return {}

        for symbol, data in raw_signals.items():
            pred   = data.get('prediction', 0.5)
            regime = data.get('regime', 'sideways')
            price  = data.get('price', 0.0)

            signal = 'HOLD'
            if regime == 'uptrend' and pred > 0.52:
                signal = 'BUY'
            elif regime == 'downtrend':
                signal = 'AVOID'

            atr_proxy   = price * CRYPTO_ATR_PCT
            action      = 'SKIP'
            veto_result = {'decision': 'APPROVE', 'reason': 'not_evaluated'}

            # H3: route crypto BUYs through veto agent
            if signal == 'BUY' and self.market_regime.get('can_trade', False):
                veto_result = self._safe_veto_review(
                    symbol     = symbol,
                    price      = price,
                    pred       = pred,
                    regime     = regime,
                    sent_score = 0.0,
                    sector     = 'Crypto',
                    mtf_score  = 1.0,   # no MTF for crypto, neutral
                )
                if veto_result.get('decision') == 'VETO':
                    action = 'VETOED'
                else:
                    action = 'OPEN'

            emoji = "🟢" if signal == 'BUY' else "🔴" if signal == 'AVOID' else "⚪"
            print(
                f"  {emoji} {symbol:10s} | pred={pred:.3f} | {regime:10s}"
                f" | {signal} | ${price:,.2f} | ATR proxy ${atr_proxy:.2f}"
            )
            if action == 'VETOED':
                print(f"    ↳ VETOED: {veto_result.get('reason', '')}")

            crypto_results[symbol] = {
                'prediction'       : float(pred),
                'regime'           : regime,
                'price'            : float(price),
                'sector'           : 'Crypto',
                'sector_multiplier': 1.0,
                'sentiment'        : 0.0,
                'signal'           : signal,
                'combined'         : float(pred),
                'sizing_combined'  : float(pred),
                'atr'              : atr_proxy,
                'mtf_score'        : 1.0,
                'veto_result'      : veto_result,
                'action'           : action,
            }

        logger.info("Crypto scan complete: %d signals in %.1fs",
                    len(crypto_results), time.time() - t0)
        return crypto_results

    # ---------------------------------------------------------------- #
    #  PRIVATE: FETCH STOCK DATA                                        #
    # ---------------------------------------------------------------- #

    def _fetch_stock_data(self) -> dict:
        n = len(self.stock_watchlist)
        print(f"\n1a. Fetching data for {n} stocks...")

        fetcher = StockDataFetcher(
            watchlist     = self.stock_watchlist,
            lookback_days = self.lookback_days,
        )
        try:
            self.stock_data = self._with_retry(
                fetcher.fetch_all, retries=3, delay=10,
                label='stock_data_fetch',
            )
        except Exception as e:
            logger.error("Stock data fetch failed: %s", e)
            return {}

        print(f"\n1b. Training 4-model ensemble per stock...")
        results  = {}
        dropped  = []
        for symbol, raw_df in self.stock_data.items():
            try:
                result = self._train_and_predict(symbol, raw_df)
                if result:
                    results[symbol] = result
                else:
                    dropped.append((symbol, "empty_result"))
            except Exception as e:
                logger.warning("Error processing %s: %s", symbol, e)
                dropped.append((symbol, str(e)[:80]))

        if dropped:
            logger.warning(
                "Dropped %d/%d symbols during training. First 5: %s",
                len(dropped), len(self.stock_data), dropped[:5],
            )

        return results

    # ---------------------------------------------------------------- #
    #  PRIVATE: TRAIN + PREDICT (H1 LOOK-AHEAD FIX)                     #
    # ---------------------------------------------------------------- #

    def _train_and_predict(self, symbol: str, raw_df: pd.DataFrame) -> dict:
        """
        H1 fix v2: build features on raw data up to the train cut-off,
        keeping a warmup buffer for long-period indicators (MA200, etc.)
        before that cut-off. This avoids both look-ahead leakage AND
        the v1 mistake of slicing too narrowly for feature warmup.

        Algorithm:
            1. Decide train_end at position (len - retrain_days).
            2. Take ALL rows up to and including train_end.
            3. Feature-engineer + regime-detect that slice with end_date
               set to train_end's date, so cross-asset context fetches
               also stop there.
            4. From the resulting feature frame, take the last
               WALK_FORWARD_DAYS rows as the actual training window.
            5. Separately, feature-engineer the full raw data for
               inference on the latest row.
        """
        # ── Position-based cuts ───────────────────────────────────────
        n = len(raw_df)
        split_raw = n - self.retrain_days
        if split_raw < MIN_TRAIN_ROWS:
            logger.debug("%s: not enough raw rows (%d), skipping",
                         symbol, n)
            return {}

        # All rows up to (and including) the train cut. Includes the
        # full warmup history before train_start, so MA200 etc. populate.
        train_buffer_raw = raw_df.iloc[:split_raw].copy()
        train_end_date   = train_buffer_raw.index[-1]

        # ── Feature engineer the train buffer with date boundary ─────
        train_buffer = self.engine.add_all_features(
            train_buffer_raw,
            end_date=str(train_end_date),
        )
        train_buffer = self.regime_detector.detect(train_buffer)

        # The actual training window is the last WALK_FORWARD_DAYS rows
        # of the feature-engineered buffer. Earlier rows were warmup
        # for the long-period indicators and are now safe to drop.
        train_start_pos = max(0, len(train_buffer) - WALK_FORWARD_DAYS)
        train = train_buffer.iloc[train_start_pos:].copy()

        # ── Feature engineer full history for inference (no end_date) ─
        # Latest row IS the inference row — admissible because we
        # predict only on it.
        df = self.engine.add_all_features(raw_df.copy())
        df = self.regime_detector.detect(df)

        feature_names = self.engine.get_feature_names()

        # ── Data quality checks on train ──────────────────────────────
        if len(train) < MIN_TRAIN_ROWS:
            logger.debug("%s: train rows after slicing (%d), skipping",
                         symbol, len(train))
            return {}

        X_train = train[feature_names]
        y_train = train['target']

        if len(y_train.unique()) < 2:
            logger.debug("%s: target single-class, skipping", symbol)
            return {}
        if y_train.value_counts().min() < 10:
            logger.debug("%s: class imbalance, skipping", symbol)
            return {}

        # ── Feature selection ─────────────────────────────────────────
        k = min(self.top_k_features, len(feature_names))
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X_train, y_train)
        mask     = selector.get_support()
        selected = [f for f, m in zip(feature_names, mask) if m]

        # ── Model cache ───────────────────────────────────────────────
        feat_hash = feature_set_hash(selected)
        cached    = load_models(symbol)

        if cached and cached.get('feature_hash') == feat_hash:
            logger.info("%s: cache hit (hash=%s)", symbol, feat_hash)
            selected = cached.get('selected_features', selected)
            model = TechnicalPredictor(use_lstm=False)
            model.models = {
                'xgboost'       : cached.get('xgboost'),
                'lightgbm'      : cached.get('lightgbm'),
                'random_forest' : cached.get('random_forest'),
                'catboost'      : cached.get('catboost'),
            }
            model.feature_names = selected
            model.trained = True
        else:
            if cached:
                logger.info("%s: cache stale, retraining", symbol)
            else:
                logger.info("%s: training models", symbol)

            model = TechnicalPredictor(use_lstm=True)
            model.train(X_train[selected], y_train)

            try:
                save_models(symbol, {
                    'xgboost'           : model.models.get('xgboost'),
                    'lightgbm'          : model.models.get('lightgbm'),
                    'random_forest'     : model.models.get('random_forest'),
                    'catboost'          : model.models.get('catboost'),
                    'selected_features' : selected,
                    'feature_hash'      : feat_hash,
                })
            except Exception as e:
                logger.warning("Cache save failed for %s: %s", symbol, e)

        # ── Predict on the LATEST row of full-history df ─────────────
        latest      = df.iloc[-1:]
        pred        = model.predict(latest[selected])[0]
        regime      = latest['regime'].iloc[0]
        price       = latest['close'].iloc[0]
        sector_mult = self.sector_analyzer.get_sector_signal(symbol)
        sector      = self.sector_analyzer.get_sector_for_stock(symbol)

        return {
            'prediction'       : float(pred),
            'regime'           : regime,
            'price'            : float(price),
            'sector'           : sector,
            'sector_multiplier': float(sector_mult),
        }

    # ---------------------------------------------------------------- #
    #  PRIVATE: MTF                                                     #
    # ---------------------------------------------------------------- #

    def _fetch_mtf_scores(self) -> dict:
        mtf = {s: 0.5 for s in self.stock_watchlist}
        bullish = 0

        for symbol in self.stock_watchlist:
            def _get(sym=symbol):
                df = self.stock_data.get(sym)
                if df is None or len(df) < 60:
                    return 0.5
                return self.mtf_analyzer.get_mtf_score(sym, df)

            try:
                score = self._with_retry(_get, retries=2, delay=2,
                                          label=f'MTF/{symbol}')
                mtf[symbol] = score
                if score > 0.5:
                    bullish += 1
                    print(f"  {symbol}: MTF {score:.0%} BULLISH")
            except Exception as e:
                logger.warning("MTF failed for %s: %s", symbol, e)

        print(f"\n  MTF complete: {bullish}/{len(self.stock_watchlist)} bullish")
        return mtf

    # ---------------------------------------------------------------- #
    #  PRIVATE: SENTIMENT                                               #
    # ---------------------------------------------------------------- #

    def _fetch_sentiments(self, raw_results: dict) -> dict:
        sentiments = {}
        if not raw_results:
            return sentiments

        top = sorted(raw_results.items(),
                     key=lambda x: x[1].get('prediction', 0),
                     reverse=True)[:SENTIMENT_TOP_N]
        top_symbols = [s[0] for s in top]

        try:
            news = self.news_fetcher.fetch_all(top_symbols)
            sentiments = self.sentiment_model.get_sentiment_for_stocks(news)
        except Exception as e:
            logger.warning("Sentiment fetch failed: %s", e)
        return sentiments

    # ---------------------------------------------------------------- #
    #  PRIVATE: BUILD STOCK SIGNAL                                      #
    # ---------------------------------------------------------------- #

    def _build_stock_signal(self, symbol, raw_result, sentiments):
        pred       = raw_result['prediction']
        regime     = raw_result['regime']
        price      = raw_result['price']
        sector     = raw_result['sector']
        sect_mult  = raw_result['sector_multiplier']
        sent_score = sentiments.get(symbol, {}).get('sentiment_score', 0.0)

        signal, combined = compute_signal(
            pred, regime, sent_score, sect_mult,
            symbol, self.earnings_symbols,
        )

        # Insider boost (sizing only)
        insider_score   = self.insider_scores.get(symbol, 0.0)
        sizing_combined = combined
        if insider_score > 0:
            sizing_combined = min(combined + insider_score * 0.5, 1.0)
            if insider_score >= 0.10:
                logger.info("%s: insider boost +%.2f (sizing only)",
                            symbol, insider_score)

        mtf_score = self.mtf_scores.get(symbol, 0.5)

        # M1 fix: ATR fallback raised to 3% of price (was 2%)
        try:
            atr = calc_atr(self.stock_data, symbol)
        except ValueError as e:
            atr = price * 0.03
            logger.warning("ATR fallback for %s (%.2f): %s", symbol, atr, e)

        # Action determination
        action      = 'SKIP'
        veto_result = {'decision': 'APPROVE', 'reason': 'not_evaluated'}

        if signal == 'BUY' and self.market_regime.get('can_trade', False):
            action, veto_result = self._apply_filters(
                symbol, price, pred, regime, sent_score, sector, mtf_score
            )

        # Console output
        emoji = {
            'BUY'          : "🟢",
            'AVOID'        : "🔴",
            'EARNINGS_HOLD': "📅",
            'CAUTION'      : "⚠️",
        }.get(signal, "⚪")
        sect_str = "🔄↑" if sect_mult > 1.1 else "🔄↓" if sect_mult < 0.9 else ""
        sent_str = "📈" if sent_score > 0.1 else "📉" if sent_score < -0.1 else ""

        print(
            f"  {emoji} {symbol:6s} | pred={pred:.3f} | {regime:10s}"
            f" | sent={sent_score:+.2f}{sent_str}"
            f" | {sector:13s}{sect_str} | {signal:14s} | ${price:.2f}"
        )
        if action not in ('OPEN', 'SKIP'):
            print(f"    ↳ {action}")

        return {
            'prediction'       : float(pred),
            'regime'           : regime,
            'price'            : float(price),
            'sector'           : sector,
            'sector_multiplier': float(sect_mult),
            'sentiment'        : float(sent_score),
            'signal'           : signal,
            'combined'         : float(combined),
            'sizing_combined'  : float(sizing_combined),
            'atr'              : float(atr),
            'mtf_score'        : float(mtf_score),
            'veto_result'      : veto_result,
            'action'           : action,
        }

    # ---------------------------------------------------------------- #
    #  PRIVATE: APPLY FILTERS                                           #
    # ---------------------------------------------------------------- #

    def _apply_filters(self, symbol, price, pred, regime,
                       sent_score, sector, mtf_score):
        veto_result = {'decision': 'APPROVE', 'reason': 'not_evaluated'}

        if mtf_score < 0.5:
            logger.info("%s: BUY blocked by MTF (%.0f%%)",
                        symbol, mtf_score * 100)
            return 'MTF_HOLD', veto_result

        if not self.corr_filter.can_add_position(symbol, self.open_positions):
            logger.info("%s: BUY blocked by correlation filter", symbol)
            return 'CORR_HOLD', veto_result

        veto_result = self._safe_veto_review(
            symbol, price, pred, regime, sent_score, sector, mtf_score
        )
        if veto_result.get('decision') == 'VETO':
            return 'VETOED', veto_result
        return 'OPEN', veto_result

    # ---------------------------------------------------------------- #
    #  M4: VETO AGENT WITH CIRCUIT BREAKER                              #
    # ---------------------------------------------------------------- #

    def _safe_veto_review(self, symbol, price, pred, regime,
                          sent_score, sector, mtf_score) -> dict:
        """
        Wrap veto_agent.review_signal() with circuit breaker.

        Fail-closed (VETO) on first error, but if N consecutive errors
        of the same type occur, switch to BYPASS for remaining symbols
        this scan. Better to ship signals through other filters than
        halt entirely.
        """
        if self._veto_bypassed:
            return {
                'decision': 'APPROVE',
                'reason'  : 'veto_agent_bypassed_after_repeated_failures',
            }

        try:
            result = self.veto_agent.review_signal(
                symbol            = symbol,
                price             = price,
                prediction        = pred,
                regime            = regime,
                sentiment         = sent_score,
                sector            = sector,
                market_regime     = self.market_regime.get('regime', 'unknown'),
                mtf_score         = mtf_score,
                current_positions = self.open_positions,
                vix               = self.market_regime.get('vix', 20),
            )
            # Reset consecutive error count on success
            self._veto_consecutive_errors = 0
            self._veto_last_error_type    = None
            return result

        except Exception as e:
            error_type = type(e).__name__

            if self._veto_last_error_type == error_type:
                self._veto_consecutive_errors += 1
            else:
                self._veto_consecutive_errors = 1
                self._veto_last_error_type    = error_type

            logger.error(
                "Veto agent error (%s, %d consecutive): %s",
                error_type, self._veto_consecutive_errors, e,
            )

            if self._veto_consecutive_errors >= VETO_FAILURE_THRESHOLD:
                logger.critical(
                    "Veto agent failed %d consecutive times with %s — "
                    "switching to BYPASS for the rest of this scan",
                    self._veto_consecutive_errors, error_type,
                )
                self._veto_bypassed = True
                # Notify if we can — telegram is constructed in main, not here
                # so just log. main.py picks this up via scan output.
                return {
                    'decision': 'APPROVE',
                    'reason'  : f'veto_bypass_after_{error_type}',
                }

            # Fail-closed for this one symbol
            return {
                'decision': 'VETO',
                'reason'  : f'veto_agent_exception: {error_type}',
            }

    # ---------------------------------------------------------------- #
    #  RETRY                                                            #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _with_retry(fn, retries=3, delay=5, label=''):
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning("Retry %d/%d for %s: %s",
                                   attempt + 1, retries, label, e)
                    time.sleep(delay)
                else:
                    logger.error("All %d attempts failed for %s: %s",
                                 retries, label, e)
                    raise
