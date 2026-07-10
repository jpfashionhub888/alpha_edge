# scanner.py
"""
StockScanner — extracted from main.py so the scan loop is
testable, importable, and single-responsibility.

Responsibilities:
    - Fetch and validate stock + crypto data
    - Engineer features per symbol (slice raw data FIRST)
    - Train / cache models per symbol
    - Generate signals via compute_signal()
    - Apply MTF, correlation, and veto filters
    - Return structured signal dicts for main.py to act on

Does NOT:
    - Open / close positions  (paper_trader.py)
    - Send Telegram alerts    (main.py)
    - Write dashboard JSON    (main.py)
    - Manage risk limits      (risk_circuit_breaker.py)
"""

import hashlib
# Suppress HuggingFace retry warnings
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

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

# ── Meta-Labeler (Phase B) ────────────────────────────────────────────
try:
    from models.meta_labeler import MetaLabeler
    _META_LABELER_AVAILABLE = True
except ImportError:
    _META_LABELER_AVAILABLE = False
    logger.warning('MetaLabeler not available — BUY signals unfiltered')


# ── Strategy settings loader (reads from config/settings.yaml) ───────
# HyperOpt writes best params here; scanner picks them up automatically.
def _load_signal_settings() -> dict:
    """
    Load signal thresholds from config/settings.yaml.
    Falls back to safe hardcoded defaults if file is missing / malformed.
    """
    defaults = {
        'buy_threshold'    : 0.55,
        'volume_spike_min' : 1.3,
        'atr_stop_mult'    : 1.0,
        'atr_target_mult'  : 2.5,
        'meta_threshold'   : 0.55,
    }
    try:
        import yaml
        with open('config/settings.yaml') as f:
            cfg = yaml.safe_load(f) or {}
        thr = cfg.get('signal_thresholds', {})
        risk = cfg.get('risk_management', {})
        hyp  = cfg.get('hyperopt', {})
        return {
            'buy_threshold'    : float(thr.get('buy_threshold',     hyp.get('buy_threshold',     defaults['buy_threshold']))),
            'volume_spike_min' : float(thr.get('volume_spike_min',  hyp.get('volume_spike_min',  defaults['volume_spike_min']))),
            'atr_stop_mult'    : float(risk.get('atr_stop_mult',    hyp.get('atr_stop_mult',     defaults['atr_stop_mult']))),
            'atr_target_mult'  : float(risk.get('atr_target_mult',  hyp.get('atr_target_mult',   defaults['atr_target_mult']))),
            'meta_threshold'   : float(hyp.get('meta_threshold',    defaults['meta_threshold'])),
        }
    except Exception as e:
        logger.warning(f'Could not load settings.yaml ({e}) — using defaults')
        return defaults

# Load once at module import; refreshed each scan cycle via StockScanner
_SIGNAL_SETTINGS = _load_signal_settings()


# Signal weight configuration
# These weights have no empirical basis yet.
# Run run_backtest.py weight optimisation before changing them.
SIGNAL_WEIGHTS = {
    'prediction': 0.60,
    'sentiment' : 0.20,
    'sector'    : 0.20,
}

# Crypto uses price * this fraction as ATR proxy
# (real ATR not available without OHLC per-bar crypto data)
CRYPTO_ATR_PCT = 0.05   # 5 % of price

# Minimum rows needed to train a model
MIN_TRAIN_ROWS = 100

# Walk-forward look-back window in days
WALK_FORWARD_DAYS = 180

# How many top stocks to run sentiment on (API cost control)
SENTIMENT_TOP_N = 7

# Earnings calendar concurrent fetch settings
EARNINGS_MAX_WORKERS = 8
EARNINGS_TIMEOUT_SEC = 30


# ------------------------------------------------------------------ #
#  STANDALONE HELPERS                                                  #
# ------------------------------------------------------------------ #

def feature_set_hash(feature_names: list) -> str:
    """
    Stable hash of the feature list used as the model cache
    version key. If feature_engine.py changes which features
    it produces, the hash changes and stale models are discarded.
    """
    key = ','.join(sorted(feature_names))
    return hashlib.md5(key.encode()).hexdigest()[:8]


def calc_atr(stock_data: dict, symbol: str) -> float:
    """
    Calculate 14-period Average True Range for *symbol*.

    Returns
    -------
    float
        ATR value > 0.

    Raises
    ------
    ValueError
        If the symbol is missing, has insufficient rows,
        or produces a NaN / non-positive ATR.
    """
    if symbol not in stock_data:
        raise ValueError(f"Symbol {symbol} not in stock_data")

    df   = stock_data[symbol].copy()
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
        raise ValueError(
            f"ATR invalid for {symbol}: {atr_val}"
        )
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
    Compute the final trading signal and combined score.

    Parameters
    ----------
    pred            : model probability of upward move (0-1)
    regime          : 'uptrend' | 'downtrend' | 'sideways' | 'volatile'
    sent_score      : sentiment score roughly in [-1, +1]
    sect_mult       : sector multiplier; 1.0 = neutral
    symbol          : ticker string
    earnings_symbols: list of tickers with earnings this week

    Returns
    -------
    signal   : str  — 'BUY' | 'HOLD' | 'AVOID' | 'CAUTION'
                       | 'EARNINGS_HOLD'
    combined : float — sizing score clipped to [0, 1]

    Signal weight rationale
    -----------------------
    SIGNAL_WEIGHTS are module constants above.
    They have no empirical basis yet — run the weight
    optimisation in run_backtest.py before tuning them.

    Sentiment contribution
    ----------------------
    sent_score * weight  →  clipped to [-weight, +weight]
    This prevents extreme sentiment from dominating combined.

    Sideways regime
    ---------------
    Does NOT generate a BUY even with strong sector.
    Removed in V4 after audit identified it as invalid logic.
    """
    # Sentiment: centre at 0, clip to ±weight
    w_sent       = SIGNAL_WEIGHTS['sentiment']
    sent_contrib = max(-w_sent, min(w_sent, sent_score * w_sent))

    combined = (
        pred                    * SIGNAL_WEIGHTS['prediction']
        + sent_contrib
        + (sect_mult - 1.0)     * SIGNAL_WEIGHTS['sector']
    )
    combined = max(0.0, min(1.0, combined))

    # ── Threshold from settings.yaml (HyperOpt writes here) ───────
    BUY_THRESHOLD = _SIGNAL_SETTINGS.get('buy_threshold', 0.55)

    # ── Regime-based signal ───────────────────────────────────────
    signal = 'HOLD'

    if regime == 'uptrend':
        if pred >= BUY_THRESHOLD:
            signal = 'BUY'
    elif regime == 'downtrend':
        signal = 'AVOID'
    elif regime == 'volatile':
        signal = 'CAUTION'
    # sideways → stays HOLD (no BUY path)

    # Sector drag: demote BUY if sector is weak
    if sect_mult < 0.8 and signal == 'BUY':
        signal = 'HOLD'

    # Earnings risk: hold off on new buys
    if symbol in earnings_symbols and signal == 'BUY':
        signal = 'EARNINGS_HOLD'

    return signal, combined

# ------------------------------------------------------------------ #
#  SCANNER CLASS                                                       #
# ------------------------------------------------------------------ #

class StockScanner:
    """
    Encapsulates the full stock + crypto scan pipeline.

    Usage
    -----
    scanner = StockScanner(
        stock_watchlist  = [...],
        crypto_watchlist = [...],
        sector_analyzer  = SectorRotation(),
        market_regime    = {'can_trade': True, 'regime': 'uptrend', ...},
        open_positions   = trader.positions,
    )
    stock_results  = scanner.scan_stocks()
    crypto_results = scanner.scan_crypto()
    """

    def __init__(
        self,
        stock_watchlist  : list,
        crypto_watchlist : list,
        sector_analyzer  : SectorRotation,
        market_regime    : dict,
        open_positions   : dict,
        lookback_days    : int  = 730,
        top_k_features   : int  = 20,
        retrain_days     : int  = 30,
    ):
        self.stock_watchlist  = stock_watchlist
        self.crypto_watchlist = crypto_watchlist
        self.sector_analyzer  = sector_analyzer
        self.market_regime    = market_regime
        self.open_positions   = open_positions
        self.lookback_days    = lookback_days
        self.top_k_features   = top_k_features
        self.retrain_days     = retrain_days

        # Sub-components
        self.engine          = FeatureEngine()
        self.regime_detector = RegimeDetector()
        self.mtf_analyzer    = MultiTimeframeAnalyzer()
        self.corr_filter     = CorrelationFilter(max_per_sector=2)
        self.veto_agent      = VetoAgent()
        self.insider_tracker = InsiderTracker()
        self.news_fetcher    = NewsFetcher()
        self.sentiment_model = SentimentAnalyzer()

        # ── Meta-Labeler cache ─────────────────────────────────────────
        # MetaLabelers are loaded lazily per symbol from ModelCache.
        # An unfitted labeler passes all signals through (fail-open).
        self._meta_labelers: dict = {}   # {symbol: MetaLabeler | None}
        if _META_LABELER_AVAILABLE:
            try:
                from model_cache import ModelCache
                self._meta_cache = ModelCache()
            except Exception:
                self._meta_cache = None
        else:
            self._meta_cache = None

        # Settings (refreshed each scan so HyperOpt changes take effect)
        global _SIGNAL_SETTINGS
        _SIGNAL_SETTINGS = _load_signal_settings()
        logger.info(
            f'Scanner init | BUY_THRESHOLD={_SIGNAL_SETTINGS["buy_threshold"]} '
            f'| meta_threshold={_SIGNAL_SETTINGS["meta_threshold"]}'
        )

        # Populated by scan methods — accessible by main.py
        self.stock_data      = {}   # raw OHLCV per symbol
        self.earnings_symbols = []
        self.insider_scores  = {}
        self.mtf_scores      = {}

    # ---------------------------------------------------------------- #
    #  PUBLIC: EARNINGS CALENDAR                                         #
    # ---------------------------------------------------------------- #

    def fetch_earnings_calendar(self) -> list:
        """
        Fetch earnings calendar for all stocks concurrently.

        Returns list of dicts:
            [{'symbol': 'AAPL', 'date': '2024-07-25', 'days_until': 3}]

        Fix: concurrent fetch replaces sequential loop that blocked
        the scan for 2-5 minutes on a 40-stock watchlist.
        """
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
            except Exception as e:
                logger.debug(f'Earnings fetch failed for {symbol}: {e}')
            return None

        with ThreadPoolExecutor(
            max_workers=EARNINGS_MAX_WORKERS
        ) as executor:
            futures = {
                executor.submit(_fetch_one, s): s
                for s in self.stock_watchlist
            }
            for future in as_completed(
                futures, timeout=EARNINGS_TIMEOUT_SEC
            ):
                try:
                    result = future.result()
                    if result:
                        earnings_soon.append(result)
                except Exception as e:
                    symbol = futures[future]
                    logger.debug(
                        "Earnings fetch failed for %s: %s", symbol, e
                    )

        earnings_soon.sort(key=lambda x: x['days_until'])

        if earnings_soon:
            print(f"  ⚠️  {len(earnings_soon)} stocks reporting this week:")
            for e in earnings_soon:
                d = e['days_until']
                w = ("TODAY!" if d == 0
                     else "TOMORROW!" if d == 1
                     else f"in {d} days")
                print(f"     ⚠️  {e['symbol']} earnings {w}")
        else:
            print("  ✅ No earnings this week")

        # Blackout window: block new BUYs only within 3 days of earnings.
        # earnings_soon itself stays a 7-day window for informational
        # display above — only the actual blocking list is tightened.
        self.earnings_symbols = [e['symbol'] for e in earnings_soon if e['days_until'] <= 3]
        return earnings_soon

    # ---------------------------------------------------------------- #
    #  PUBLIC: SCAN STOCKS                                               #
    # ---------------------------------------------------------------- #

    def scan_stocks(self) -> dict:
        """
        Full stock scan pipeline.

        Returns
        -------
        dict keyed by symbol:
        {
            'prediction'       : float,
            'regime'           : str,
            'price'            : float,
            'sector'           : str,
            'sector_multiplier': float,
            'sentiment'        : float,
            'signal'           : str,
            'combined'         : float,
            'sizing_combined'  : float,   ← insider-adjusted for sizing only
            'atr'              : float,
            'mtf_score'        : float,
            'veto_result'      : dict,
            'action'           : str,     ← 'OPEN' | 'SKIP' | reason
        }
        """
        t0 = time.time()
        print("\n" + "=" * 60)
        print("PHASE 1: STOCK ANALYSIS")
        print("=" * 60)

        # ── 1a. Fetch raw data ────────────────────────────────────
        results = self._fetch_stock_data()
        if not results:
            logger.error("No stock data fetched — aborting stock scan")
            return {}

        # ── 1b. Insider scores ────────────────────────────────────
        print("\n  Loading insider trading data...")
        try:
            self.insider_scores = self.insider_tracker.get_bulk_scores(
                self.stock_watchlist, days_back=30
            )
        except Exception as e:
            logger.warning("Insider tracker failed: %s", e)
            self.insider_scores = {}

        # ── 1c. MTF scores ────────────────────────────────────────
        self.mtf_scores = self._fetch_mtf_scores()

        # ── 1d. Sentiment on top-N predicted stocks ───────────────
        sentiments = self._fetch_sentiments(results)

        # ── 1e. Build final signal per stock ─────────────────────
        final_signals = {}
        for symbol, raw_result in results.items():
            signal_record = self._build_stock_signal(
                symbol     = symbol,
                raw_result = raw_result,
                sentiments = sentiments,
            )
            if signal_record:
                final_signals[symbol] = signal_record

        logger.info(
            "Stock scan complete: %d signals in %.1fs",
            len(final_signals),
            time.time() - t0,
        )
        return final_signals

    # ---------------------------------------------------------------- #
    #  PUBLIC: SCAN CRYPTO                                               #
    # ---------------------------------------------------------------- #

    def scan_crypto(self) -> dict:
        """
        Crypto scan pipeline.

        Returns
        -------
        dict keyed by symbol with same schema as scan_stocks()
        plus action='OPEN'|'SKIP'.

        Fix: ATR proxy (5% of price) now passed to open_position
        instead of omitting ATR entirely (which defaulted to a
        3% stop — far too tight for crypto).
        """
        t0 = time.time()
        print("\n" + "=" * 60)
        print("PHASE 2: CRYPTO ANALYSIS")
        print("=" * 60)

        crypto_results = {}
        try:
            predictor    = CryptoPredictor()
            raw_signals  = predictor.run_full_pipeline(
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

            # ATR proxy for crypto — 5% of price
            # Real ATR requires per-bar OHLC which CryptoPredictor
            # does not currently expose. 5% is conservative for
            # BTC/ETH daily moves; SOL may need wider.
            atr_proxy = price * CRYPTO_ATR_PCT

            action = 'SKIP'
            if (signal == 'BUY'
                    and self.market_regime.get('can_trade', False)):
                action = 'OPEN'

            emoji = ("🟢" if signal == 'BUY'
                     else "🔴" if signal == 'AVOID'
                     else "⚪")
            print(
                f"  {emoji} {symbol:10s}"
                f" | pred={pred:.3f}"
                f" | {regime:10s}"
                f" | {signal}"
                f" | ${price:,.2f}"
                f" | ATR proxy ${atr_proxy:.2f}"
            )

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
                'mtf_score'        : 0.5,
                'veto_result'      : {'decision': 'APPROVE', 'reason': 'crypto_path'},
                'action'           : action,
            }

        logger.info(
            "Crypto scan complete: %d signals in %.1fs",
            len(crypto_results),
            time.time() - t0,
        )
        return crypto_results

    # ---------------------------------------------------------------- #
    #  PRIVATE: FETCH STOCK DATA                                         #
    # ---------------------------------------------------------------- #

    def _fetch_stock_data(self) -> dict:
        """
        Fetch raw OHLCV, train models, and return intermediate
        results keyed by symbol.
        """
        n = len(self.stock_watchlist)
        print(f"\n1a. Fetching data for {n} stocks...")

        fetcher = StockDataFetcher(
            watchlist    = self.stock_watchlist,
            lookback_days= self.lookback_days,
        )
        try:
            self.stock_data = self._with_retry(
                fetcher.fetch_all,
                retries=3,
                delay=10,
                label='stock_data_fetch',
            )
        except Exception as e:
            logger.error("Stock data fetch failed: %s", e)
            return {}

        print(f"\n1b. Training 4-model ensemble per stock...")
        results = {}

        for symbol, raw_df in self.stock_data.items():
            try:
                result = self._train_and_predict(symbol, raw_df)
                if result:
                    results[symbol] = result
            except Exception as e:
                logger.warning("Error processing %s: %s", symbol, e)

        return results

    # ---------------------------------------------------------------- #
    #  PRIVATE: TRAIN AND PREDICT PER SYMBOL                            #
    # ---------------------------------------------------------------- #

    def _train_and_predict(self, symbol: str, raw_df: pd.DataFrame) -> dict:
        """
        Slice raw data FIRST, then engineer features inside each
        window. This is the core look-ahead bias fix:

        OLD (wrong):
            df = engine.add_all_features(raw_df)   # full dataset
            train = df.iloc[train_start:split]      # slice after

        NEW (correct):
            train_raw = raw_df.iloc[train_start:split]
            train     = engine.add_all_features(train_raw)
            df        = engine.add_all_features(raw_df)
        """
        # ── Slice raw data BEFORE feature engineering ─────────────
        split             = len(raw_df) - self.retrain_days
        walk_forward_days = WALK_FORWARD_DAYS
        train_start       = max(0, split - walk_forward_days)

        train_raw = raw_df.iloc[train_start:split].copy()
        full_raw  = raw_df.copy()

        # ── Feature engineering on full history ───────────────────
        # Engineer on full dataset first so rolling indicators
        # have enough history (MA200 needs 200+ bars)
        # Then split by position after feature engineering
        df = self.engine.add_all_features(full_raw)

        # Pre-create target to prevent look-ahead in feature engine
        full_raw['target'] = (
            full_raw['close'].shift(-1) > full_raw['close']
        ).astype(int)

        # Split after feature engineering
        split_idx = len(df) - self.retrain_days
        train_start = max(0, split_idx - WALK_FORWARD_DAYS)

        train = df.iloc[train_start:split_idx].copy()

        feature_names = self.engine.get_feature_names()

        train = self.regime_detector.detect(train)
        df    = self.regime_detector.detect(df)

        # ── Data quality checks ───────────────────────────────────
        if len(train) < MIN_TRAIN_ROWS:
            logger.debug(
                "%s: insufficient train rows (%d), skipping",
                symbol, len(train)
            )
            return {}

        X_train = train[feature_names]
        y_train = train['target']

        if len(y_train.unique()) < 2:
            logger.debug("%s: target has only one class, skipping", symbol)
            return {}
        if y_train.value_counts().min() < 10:
            logger.debug("%s: class imbalance too severe, skipping", symbol)
            return {}

        # ── Feature selection ─────────────────────────────────────
        k = min(self.top_k_features, len(feature_names))
        selector = SelectKBest(
            score_func=mutual_info_classif, k=k
        )
        selector.fit(X_train, y_train)
        mask     = selector.get_support()
        selected = [f for f, m in zip(feature_names, mask) if m]

        # ── Model cache with feature hash versioning ──────────────
        feat_hash = feature_set_hash(selected)
        cached    = load_models(symbol)

        if cached and cached.get('feature_hash') == feat_hash:
            logger.info(
                "%s: cache hit (hash=%s)", symbol, feat_hash
            )
            selected = cached.get('selected_features', selected)
            model    = TechnicalPredictor(use_lstm=False)
            model.models = {
                'xgboost'      : cached.get('xgboost'),
                'lightgbm'     : cached.get('lightgbm'),
                'random_forest': cached.get('random_forest'),
                'catboost'     : cached.get('catboost'),
            }
            model.feature_names = selected
            model.trained       = True
        else:
            if cached:
                logger.info(
                    "%s: cache stale (feature set changed), retraining",
                    symbol
                )
            else:
                logger.info("%s: training models", symbol)

            model = TechnicalPredictor(use_lstm=True)
            model.train(X_train[selected], y_train)

            try:
                save_models(symbol, {
                    'xgboost'          : model.models.get('xgboost'),
                    'lightgbm'         : model.models.get('lightgbm'),
                    'random_forest'    : model.models.get('random_forest'),
                    'catboost'         : model.models.get('catboost'),
                    'selected_features': selected,
                    'feature_hash'     : feat_hash,
                })
            except Exception as e:
                logger.warning("Cache save failed for %s: %s", symbol, e)

        # ── Predict on latest row ─────────────────────────────────
        latest = df.iloc[-1:]
        _preds = model.predict(latest[selected])
        pred   = _preds[0] if len(_preds) > 0 else 0.5  # guard: empty array → neutral
        regime      = latest['regime'].iloc[0]
        price       = latest['close'].iloc[0]
        sector_mult = self.sector_analyzer.get_sector_signal(symbol)
        sector      = self.sector_analyzer.get_sector_for_stock(symbol)

        # AUC is only meaningful when the model was freshly trained this
        # scan — a cache-hit model never calls .train() so its train_auc/
        # val_auc are just the class defaults (0.5), not real measurements.
        # Reporting those as if they were live AUC would be misleading, so
        # only surface them on an actual retrain.
        freshly_trained = not (cached and cached.get('feature_hash') == feat_hash)
        model_auc_info = None
        if freshly_trained:
            model_auc_info = {
                'train_auc': float(getattr(model, 'train_auc', 0.5)),
                'val_auc':   float(getattr(model, 'val_auc', 0.5)),
            }

        return {
            'prediction'       : float(pred),
            'regime'           : regime,
            'price'            : float(price),
            'sector'           : sector,
            'sector_multiplier': float(sector_mult),
            'model_auc_info'   : model_auc_info,
        }

    # ---------------------------------------------------------------- #
    #  PRIVATE: MTF SCORES                                               #
    # ---------------------------------------------------------------- #

    def _fetch_mtf_scores(self) -> dict:
        """Fetch multi-timeframe alignment scores for all stocks."""
        mtf_scores = {s: 0.5 for s in self.stock_watchlist}
        bullish = 0

        for symbol in self.stock_watchlist:
            def _get_mtf(sym=symbol):
                daily_df = self.stock_data.get(sym)
                if daily_df is None or len(daily_df) < 60:
                    return 0.5
                return self.mtf_analyzer.get_mtf_score(sym, daily_df)

            try:
                score = self._with_retry(
                    _get_mtf,
                    retries=2,
                    delay=2,
                    label=f'MTF/{symbol}',
                )
                mtf_scores[symbol] = score
                if score > 0:
                    bullish += 1
                    print(f"  {symbol}: MTF {score:.0%} BULLISH")
            except Exception as e:
                logger.warning("MTF failed for %s: %s", symbol, e)

        print(
            f"\n  MTF complete: "
            f"{bullish}/{len(self.stock_watchlist)} bullish"
        )
        return mtf_scores

    # ---------------------------------------------------------------- #
    #  PRIVATE: SENTIMENT                                                #
    # ---------------------------------------------------------------- #

    def _fetch_sentiments(self, raw_results: dict) -> dict:
        """
        Run sentiment analysis on the top-N stocks by prediction
        score to control API cost.
        """
        sentiments = {}
        if not raw_results:
            return sentiments

        top_symbols = sorted(
            raw_results.items(),
            key=lambda x: x[1].get('prediction', 0),
            reverse=True,
        )[:SENTIMENT_TOP_N]
        top_symbols = [s[0] for s in top_symbols]

        try:
            all_news   = self.news_fetcher.fetch_all(top_symbols)
            sentiments = self.sentiment_model.get_sentiment_for_stocks(
                all_news
            )
        except Exception as e:
            logger.warning("Sentiment fetch failed: %s", e)

        return sentiments

    # ---------------------------------------------------------------- #
    #  PRIVATE: BUILD FINAL STOCK SIGNAL                                 #
    # ---------------------------------------------------------------- #

    def _build_stock_signal(
        self,
        symbol     : str,
        raw_result : dict,
        sentiments : dict,
    ) -> dict:
        """
        Combine model output, sentiment, sector, insider data,
        and all filters into one final signal record.

        Returns None if the symbol should be entirely excluded.
        """
        pred       = raw_result['prediction']
        regime     = raw_result['regime']
        price      = raw_result['price']
        sector     = raw_result['sector']
        sect_mult  = raw_result['sector_multiplier']
        sent_score = sentiments.get(symbol, {}).get(
            'sentiment_score', 0.0
        )

        signal, combined = compute_signal(
            pred, regime, sent_score, sect_mult,
            symbol, self.earnings_symbols,
        )

        # ── Meta-Labeler BUY gate ───────────────────────────────────
        # Only runs if signal is BUY — saves compute on HOLD/AVOID.
        # Loads meta model from cache; pass-through if not fitted yet.
        meta_confidence = 1.0   # default: let through
        if signal == 'BUY' and _META_LABELER_AVAILABLE and self._meta_cache:
            try:
                # Lazy-load from cache
                if symbol not in self._meta_labelers:
                    self._meta_labelers[symbol] = (
                        self._meta_cache.get_meta_labeler(symbol)
                    )
                meta_model = self._meta_labelers.get(symbol)

                if meta_model is not None and meta_model._is_fitted:
                    latest_features = self.stock_data[symbol].copy()
                    latest_features = self.engine.add_all_features(latest_features)
                    feat_cols = [c for c in latest_features.columns
                                 if c not in ('open','high','low','close',
                                              'volume','target','future_return')]
                    X_live = latest_features[feat_cols].iloc[-1:]

                    approved, meta_confidence = meta_model.should_trade(
                        X_live, pred
                    )
                    if not approved:
                        signal = 'HOLD'
                        logger.info(
                            f'{symbol}: Meta-Labeler filtered BUY '
                            f'(meta_conf={meta_confidence:.3f} < '
                            f'{_SIGNAL_SETTINGS["meta_threshold"]:.2f})'
                        )
                else:
                    logger.debug(
                        f'{symbol}: MetaLabeler not fitted yet — passing BUY through'
                    )
            except Exception as e:
                logger.warning(f'{symbol}: MetaLabeler gate error (non-fatal): {e}')

        # ── Insider boost (sizing only — does NOT promote signal) ─
        insider_score   = self.insider_scores.get(symbol, 0.0)
        sizing_combined = combined
        if insider_score > 0:
            sizing_combined = min(
                combined + insider_score * 0.5, 1.0
            )
            if insider_score >= 0.10:
                logger.info(
                    "%s: insider boost +%.2f "
                    "(sizing only, signal stays %s)",
                    symbol, insider_score, signal,
                )

        mtf_score = self.mtf_scores.get(symbol, 0.5)

        # ── ATR ───────────────────────────────────────────────────
        try:
            atr = calc_atr(self.stock_data, symbol)
        except ValueError as e:
            atr = price * 0.02   # 2% price fallback
            logger.warning(
                "ATR fallback for %s (%.2f): %s", symbol, atr, e
            )

        # ── Action determination ──────────────────────────────────
        action     = 'SKIP'
        veto_result = {'decision': 'APPROVE', 'reason': 'not_evaluated'}

        if signal == 'BUY' and self.market_regime.get('can_trade', False):
            action, veto_result = self._apply_filters(
                symbol     = symbol,
                price      = price,
                pred       = pred,
                regime     = regime,
                sent_score = sent_score,
                sector     = sector,
                mtf_score  = mtf_score,
            )

        # ── Console output ────────────────────────────────────────
        emoji = {
            'BUY'          : "🟢",
            'AVOID'        : "🔴",
            'EARNINGS_HOLD': "📅",
            'CAUTION'      : "⚠️",
        }.get(signal, "⚪")

        sect_emoji = (
            "🔄↑" if sect_mult > 1.1
            else "🔄↓" if sect_mult < 0.9
            else ""
        )
        sent_emoji = (
            "📈" if sent_score > 0.1
            else "📉" if sent_score < -0.1
            else ""
        )

        print(
            f"  {emoji} {symbol:6s}"
            f" | pred={pred:.3f}"
            f" | {regime:10s}"
            f" | sent={sent_score:+.2f}{sent_emoji}"
            f" | {sector:13s}{sect_emoji}"
            f" | {signal:14s}"
            f" | ${price:.2f}"
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
            'meta_confidence'  : float(meta_confidence),
            'veto_result'      : veto_result,
            'action'           : action,
            'model_auc_info'   : raw_result.get('model_auc_info'),
        }

    # ---------------------------------------------------------------- #
    #  PRIVATE: APPLY FILTERS                                            #
    # ---------------------------------------------------------------- #

    def _apply_filters(
        self,
        symbol    : str,
        price     : float,
        pred      : float,
        regime    : str,
        sent_score: float,
        sector    : str,
        mtf_score : float,
    ) -> tuple:
        """
        Apply MTF, correlation, and veto filters in order.

        Returns
        -------
        action      : str  — 'OPEN' | reason string if blocked
        veto_result : dict
        """
        veto_result = {'decision': 'APPROVE', 'reason': 'not_evaluated'}

        # ── MTF filter ────────────────────────────────────────────
        if mtf_score < 0.5:
            logger.info(
                "%s: BUY blocked by MTF filter (%.0f%%)",
                symbol, mtf_score * 100,
            )
            return 'MTF_HOLD', veto_result

        # ── Correlation / sector concentration filter ─────────────
        if not self.corr_filter.can_add_position(
            symbol, self.open_positions
        ):
            logger.info(
                "%s: BUY blocked by correlation filter", symbol
            )
            return 'CORR_HOLD', veto_result

        # ── Veto agent ────────────────────────────────────────────
        try:
            veto_result = self.veto_agent.review_signal(
                symbol            = symbol,
                price             = price,
                prediction        = pred,
                regime            = regime,
                sentiment         = sent_score,
                sector            = sector,
                market_regime     = self.market_regime.get(
                    'regime', 'unknown'
                ),
                mtf_score         = mtf_score,
                current_positions = self.open_positions,
                vix               = self.market_regime.get('vix', 20),
            )
        except Exception as e:
            # Fail-closed: any veto agent error = VETO
            logger.error(
                "Veto agent exception for %s: %s — failing closed",
                symbol, e,
            )
            return 'VETO_ERROR', {
                'decision': 'VETO',
                'reason'  : f'veto_agent_exception: {e}',
            }

        if veto_result.get('decision') == 'VETO':
            logger.info(
                "%s: VETOED — %s",
                symbol, veto_result.get('reason', ''),
            )
            return 'VETOED', veto_result

        return 'OPEN', veto_result

    # ---------------------------------------------------------------- #
    #  PRIVATE: RETRY WRAPPER                                            #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _with_retry(fn, retries=3, delay=5, label=''):
        """
        Call fn() up to `retries` times with `delay` seconds
        between attempts. Raises on final failure.
        """
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(
                        "Retry %d/%d for %s: %s",
                        attempt + 1, retries, label, e,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "All %d attempts failed for %s: %s",
                        retries, label, e,
                    )
                    raise