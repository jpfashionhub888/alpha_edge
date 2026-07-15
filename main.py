# main.py
"""
AlphaEdge Main Trading Scanner V4
Fixes applied:
  - dashboard_signals now captures ALL signals (not just BUY)
  - compute_signal: removed sideways+sector BUY, fixed sentiment formula
  - Atomic JSON writes (temp file + rename) to prevent corruption
  - Retry wrapper on data fetches
  - ATR None-guard before passing to open_position
  - VetoAgent import guard (won't crash if groq missing)
  - Model cache version key tied to feature set hash
"""

import warnings
import os
import hashlib
import json
import logging
import tempfile
import time
import pandas as pd
from datetime import datetime
from config import settings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from veto_agent import VetoAgent
from insider_tracker import InsiderTracker
from data.stock_data import StockDataFetcher
from data.news_data import NewsFetcher
from data.feature_engine import FeatureEngine
from models.technical_model import TechnicalPredictor
from models.sentiment_model import SentimentAnalyzer
from models.regime_detector import RegimeDetector
from market_regime import MarketRegimeFilter
from multi_timeframe import MultiTimeframeAnalyzer
from correlation_filter import CorrelationFilter
from models.crypto_predictor import CryptoPredictor
from models.sector_rotation import SectorRotation
from execution.paper_trader import PaperTrader
from monitoring.telegram_bot import TelegramBot
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from model_cache import save_models, load_models, is_cache_valid
from critic_agent import CriticAgent
from performance_analytics import PerformanceAnalytics
from risk_circuit_breaker import RiskCircuitBreaker

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Log rotation — cap at 5 MB × 5 backups so logs/ never fills the disk
from logging.handlers import RotatingFileHandler as _RFH
_rfh = _RFH('logs/alphaedge_daily.log', maxBytes=5_000_000, backupCount=5)
_rfh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logging.getLogger().addHandler(_rfh)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def atomic_json_write(filepath: str, data: object) -> None:
    """Write JSON atomically: write to temp file, then rename.
    Prevents corrupted JSON if process is killed mid-write."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(filepath) or '.', suffix='.tmp'
    )
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, filepath)  # atomic on POSIX and Windows
    except Exception:
        os.unlink(tmp_path)
        raise


def with_retry(fn, retries=3, delay=5, label=''):
    """Call fn() up to `retries` times with `delay` seconds between attempts."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Retry {attempt+1}/{retries} for {label}: {e}")
                time.sleep(delay)
            else:
                logger.error(f"All {retries} attempts failed for {label}: {e}")
                raise


def feature_set_hash(feature_names: list) -> str:
    """Hash of feature list used as cache version key."""
    key = ','.join(sorted(feature_names))
    return hashlib.md5(key.encode()).hexdigest()[:8]


def get_earnings_calendar(watchlist):
    """Check which stocks have earnings this week (with retry)."""
    import yfinance as yf
    print("\n📅 Checking earnings calendar...")
    earnings_soon = []
    for symbol in watchlist:
        try:
            def _fetch():
                ticker = yf.Ticker(symbol)
                return ticker.calendar
            cal = with_retry(_fetch, retries=2, delay=2, label=f'earnings/{symbol}')
            if cal is not None and len(cal) > 0:
                if isinstance(cal, dict):
                    ed = cal.get('Earnings Date', [None])
                    if ed:
                        if isinstance(ed, list):
                            ed = ed[0]
                        if ed is not None:
                            from datetime import timedelta
                            now = datetime.now()
                            if hasattr(ed, 'date'):
                                ed = ed.date()
                            today = now.date()
                            diff = (ed - today).days
                            if 0 <= diff <= 7:
                                earnings_soon.append({
                                    'symbol': symbol,
                                    'date': str(ed),
                                    'days_until': diff,
                                })
        except Exception as e:
            logger.debug(f'Earnings date parse skipped: {e}')  # Non-critical; skip silently

    if earnings_soon:
        n = len(earnings_soon)
        print(f"  ⚠️  {n} stocks reporting this week:")
        for e in earnings_soon:
            d = e['days_until']
            w = "TODAY!" if d == 0 else ("TOMORROW!" if d == 1 else f"in {d} days")
            print(f"     ⚠️  {e['symbol']} earnings {w}")
    else:
        print("  ✅ No earnings this week")
    return earnings_soon


def get_full_watchlist():
    """Return expanded stock watchlist."""
    return [
        # Mega-cap tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'AMD', 'NFLX',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'DIA',
        'XLF', 'XLK', 'XLE', 'XLV',
        # Financials
        'JPM', 'V', 'GS', 'BAC', 'MS',
        'WFC', 'C', 'AXP', 'BLK',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV', 'LLY', 'MRK',
        'MRNA', 'GILD', 'REGN',
        # Consumer / Retail
        'WMT', 'COST', 'HD', 'MCD',
        'KO', 'PEP', 'PG', 'NKE', 'TGT',
        # Energy
        'XOM', 'CVX', 'OXY',
        # Industrials
        'CAT', 'HON', 'BA', 'GE', 'LMT',
        # Semiconductors
        'TSM', 'INTC', 'QCOM', 'AVGO', 'MU',
        # SaaS / Cloud
        'CRM', 'SNOW', 'NET', 'DDOG', 'CRWD',
        'NOW', 'ADBE', 'ORCL', 'INTU',
        # Speculative / Growth
        'SOFI', 'PLTR', 'RIVN', 'HOOD', 'MARA',
    ]


def compute_signal(pred, regime, sent_score, sect_mult,
                   symbol, earnings_symbols):
    """
    Compute the final trading signal.

    FIX: sentiment formula corrected — sent_score is roughly -1..+1,
    so its contribution is (sent_score * 0.2), not (sent_score + 0.5) * 0.2
    which was artificially inflating combined scores.

    FIX: removed sideways+sector_boost -> BUY path.
    Sideways regime does NOT support a BUY without uptrend confirmation.

    combined is kept as a probability-like score 0..1 for position sizing.
    """
    # Normalise sentiment contribution: centre at 0, scale to [-0.2, +0.2]
    sent_contrib = max(-0.2, min(0.2, sent_score * 0.2))

    combined = (
        pred * 0.6
        + sent_contrib
        + (sect_mult - 1.0) * 0.2   # sector: 1.0 = neutral, >1 positive
    )
    combined = max(0.0, min(1.0, combined))

    # C3 FIX: use settings.BUY_THRESHOLD (0.63) not hardcoded 0.55/0.52.
    # bybit_live.py / gateio_live.py already use settings — now main.py matches.
    _buy_thr = settings.BUY_THRESHOLD          # e.g. 0.63 from settings.yaml
    _buy_thr_pred = _buy_thr - 0.11           # secondary pred-only gate

    signal = 'HOLD'

    if regime == 'uptrend' and combined > _buy_thr:
        signal = 'BUY'
    elif regime == 'uptrend' and pred > _buy_thr_pred:
        signal = 'BUY'
    elif regime == 'downtrend':
        signal = 'AVOID'
    elif regime == 'volatile':
        signal = 'CAUTION'
    # REMOVED: sideways + sector boost -> BUY (was invalid logic)

    # Sector drag: demote BUY if sector is weak
    if sect_mult < 0.8 and signal == 'BUY':
        signal = 'HOLD'

    # Earnings risk: hold off on executing new buys
    if symbol in earnings_symbols and signal == 'BUY':
        signal = 'EARNINGS_HOLD'

    return signal, combined


def calc_atr(stock_data, symbol):
    """Calculate 14-period ATR. Returns float or raises ValueError."""
    if symbol not in stock_data:
        raise ValueError(f"Symbol {symbol} not in stock_data")
    df_atr = stock_data[symbol].copy()
    high  = df_atr['high']
    low   = df_atr['low']
    close = df_atr['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = true_range.rolling(14).mean().iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        raise ValueError(f"ATR invalid for {symbol}: {atr_val}")
    return float(atr_val)


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def run_daily_scan():
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print("\n" + "🚀" * 25)
    print(f"ALPHA EDGE V4 - {now}")
    print("4-Model Ensemble + Sector Rotation + LSTM")
    print("🚀" * 25)

    trader   = PaperTrader(starting_capital=10000.0)
    trader.load_state()
    telegram = TelegramBot()

    stock_watchlist = get_full_watchlist()

    # ======================================================================
    # PHASE 0: EARNINGS + SECTOR ROTATION
    # ======================================================================
    print("\n" + "="*60)
    print("PHASE 0: EARNINGS + SECTOR ROTATION")
    print("="*60)

    earnings_soon    = get_earnings_calendar(stock_watchlist)
    # Blackout window: block new BUYs only within 3 days of earnings.
    # earnings_soon itself stays a 7-day window for informational
    # display/logging — only the actual blocking list is tightened.
    earnings_symbols = [e['symbol'] for e in earnings_soon if e.get('days_until', 99) <= 3]

    sector_analyzer = SectorRotation()
    sector_scores   = sector_analyzer.analyze()

    # ======================================================================
    # MARKET REGIME FILTER
    # ======================================================================
    print("\n" + "="*60)
    print("MARKET REGIME FILTER")
    print("="*60)

    regime_filter = MarketRegimeFilter()
    market_regime = regime_filter.analyze()

    # ======================================================================
    # RISK CIRCUIT BREAKER CHECK
    # ======================================================================
    print("\n" + "="*60)
    print("RISK CIRCUIT BREAKER")
    print("="*60)

    circuit_breaker = RiskCircuitBreaker()
    portfolio_value = trader.capital + sum(
        pos.get('shares', 0) * pos.get('current_price', pos.get('entry_price', 0))
        for pos in trader.positions.values()
    )
    circuit_triggered = circuit_breaker.check(
        current_value    = portfolio_value,
        starting_capital = trader.starting_capital,
        telegram         = telegram,
    )
    if circuit_triggered:
        market_regime['can_trade'] = False
        print("  Trading suspended by circuit breaker!")

    corr_filter    = CorrelationFilter(max_per_sector=2)
    veto_agent     = VetoAgent()
    insider_tracker = InsiderTracker()

    print("\n  Loading insider trading data...")
    insider_scores = insider_tracker.get_bulk_scores(stock_watchlist, days_back=30)

    if not market_regime['can_trade']:
        print(f"\n  CASH MODE ACTIVATED!")
        print(f"  Reason: {market_regime['reason']}")
        print(f"  No new BUY signals will be executed")

    # ======================================================================
    # MULTI-TIMEFRAME ANALYSIS
    # ======================================================================
    print("\n" + "="*60)
    print("MULTI-TIMEFRAME ANALYSIS")
    print("="*60)

    mtf_analyzer = MultiTimeframeAnalyzer()
    mtf_scores   = {}
    # S5 FIX: MTF loop moved to after stock_data fetch (see below)
    # so daily_df=stock_data.get(symbol) can be passed, eliminating
    # 41 redundant yfinance HTTP calls (~60s) per scan.

    if not market_regime['can_trade']:
        print("\n  Skipping MTF (CASH MODE active)")
        for symbol in stock_watchlist:
            mtf_scores[symbol] = 0.5

    # ======================================================================
    # PHASE 1: STOCK ANALYSIS
    # ======================================================================
    print("\n" + "="*60)
    print("PHASE 1: STOCK ANALYSIS")
    print("="*60)

    n = len(stock_watchlist)
    print(f"\n1a. Fetching data for {n} stocks...")

    stock_fetcher = StockDataFetcher(
        watchlist=stock_watchlist,
        lookback_days=730
    )
    stock_data = with_retry(stock_fetcher.fetch_all, retries=3, delay=10,
                             label='stock_data_fetch')

    # S5 FIX: compute MTF NOW that stock_data is available — reuse daily_df
    # to skip 41 redundant yfinance HTTP calls that were adding ~60s per scan.
    if market_regime['can_trade']:
        print("\n  Checking timeframe alignment (reusing downloaded data)...")
        for symbol in stock_watchlist:
            try:
                score = mtf_analyzer.get_mtf_score(
                    symbol, daily_df=stock_data.get(symbol)
                )
                mtf_scores[symbol] = score
                if score > 0:
                    print(f"  {symbol}: MTF {score:.0%} BULLISH")
            except Exception as e:
                mtf_scores[symbol] = 0.5
                logger.warning(f"MTF failed for {symbol}: {e}")
        bullish = sum(1 for s in mtf_scores.values() if s > 0)
        print(f"\n  MTF complete: {bullish}/{len(stock_watchlist)} bullish")

    print("\n1b. Training 4-model ensemble per stock...")
    engine          = FeatureEngine()
    regime_detector = RegimeDetector()
    stock_signals   = {}

    for symbol, raw_df in stock_data.items():
        try:
            df           = engine.add_all_features(raw_df)
            feature_names = engine.get_feature_names()
            df           = regime_detector.detect(df)

            if len(df) < 100:
                continue

            split              = len(df) - 30
            walk_forward_days  = 180
            train_start        = max(0, split - walk_forward_days)
            train              = df.iloc[train_start:split]

            if len(train) < 100:
                train = df.iloc[:split]

            X_train = train[feature_names]
            y_train = train['target']

            if len(y_train.unique()) < 2:
                continue
            if y_train.value_counts().min() < 10:
                continue

            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(20, len(feature_names))
            )
            selector.fit(X_train, y_train)
            mask     = selector.get_support()
            selected = [f for f, m in zip(feature_names, mask) if m]

            # Cache version key includes feature set hash to prevent stale models
            feat_hash = feature_set_hash(selected)
            # S4 FIX: pass feature_names so the hash check is active on reload
            cached    = load_models(symbol, feature_names=selected)

            if cached and cached.get('feature_hash') == feat_hash:
                print(f"  {symbol}: Loading from cache (hash={feat_hash})...")
                selected = cached.get('selected_features', selected)
                model    = TechnicalPredictor(use_lstm=False)
                model.models = {
                    'xgboost'     : cached.get('xgboost'),
                    'lightgbm'    : cached.get('lightgbm'),
                    'random_forest': cached.get('random_forest'),
                    'catboost'    : cached.get('catboost'),
                }
                model.feature_names = selected
                model.trained       = True
            else:
                if cached:
                    print(f"  {symbol}: Cache stale (feature set changed), retraining...")
                else:
                    print(f"  {symbol}: Training models...")
                model = TechnicalPredictor(use_lstm=True)
                model.train(X_train[selected], y_train)
                # C2 FIX: enforce overfit guard — skip this model's signals
                if getattr(model, 'overfit_flagged', False):
                    gap = getattr(model, 'overfit_gap', 0)
                    logger.warning(
                        f"{symbol}: SKIP — overfit gap {gap:.2f} > 0.20 "
                        f"(train/val AUC delta too large, signals unreliable)"
                    )
                    continue
                try:
                    # S4 FIX: pass feature_names so hash is stored and checked on reload
                    save_models(symbol, {
                        'xgboost'         : model.models.get('xgboost'),
                        'lightgbm'        : model.models.get('lightgbm'),
                        'random_forest'   : model.models.get('random_forest'),
                        'catboost'        : model.models.get('catboost'),
                        'selected_features': selected,
                        'feature_hash'    : feat_hash,
                    }, feature_names=selected)
                    print(f"  {symbol}: Models cached (hash={feat_hash})!")
                except Exception as e:
                    logger.warning(f"Cache save failed for {symbol}: {e}")

            latest     = df.iloc[-1:]
            _preds     = model.predict(latest[selected])
            pred       = _preds[0] if len(_preds) > 0 else 0.5

            # C4 FIX: MetaLabeler gate — filters false positives from primary model
            # load() returns unfitted (pass-through) if no cached meta model exists.
            try:
                from models.meta_labeler import MetaLabeler
                _meta_path = os.path.join('model_cache', f'meta_{symbol}.pkl')
                _meta      = MetaLabeler.load(_meta_path)
                _approved, _meta_conf = _meta.should_trade(latest[selected], pred)
                if not _approved:
                    logger.info(
                        f"{symbol}: MetaLabeler filtered "
                        f"(primary={pred:.3f}, meta_conf={_meta_conf:.3f})"
                    )
                    stock_signals[symbol] = {
                        'prediction': pred, 'regime': latest.iloc[0]['regime'],
                        'price': latest.iloc[0]['close'],
                        'sector': sector_analyzer.get_sector_for_stock(symbol),
                        'sector_multiplier': sector_analyzer.get_sector_signal(symbol),
                        'signal': 'META_FILTERED', 'combined': 0.0,
                        'meta_conf': _meta_conf, 'model_auc_info': None,
                    }
                    continue
            except Exception as _me:
                logger.debug(f"{symbol}: MetaLabeler skipped ({_me})")

            regime     = latest['regime'].iloc[0]
            price      = latest['close'].iloc[0]
            sector_mult = sector_analyzer.get_sector_signal(symbol)
            sector     = sector_analyzer.get_sector_for_stock(symbol)

            # AUC only meaningful when freshly trained this scan — a
            # cache-hit model never calls .train(), so train_auc/val_auc
            # would just be the class defaults (0.5), not real measurements.
            freshly_trained = not (cached and cached.get('feature_hash') == feat_hash)
            model_auc_info = None
            if freshly_trained:
                model_auc_info = {
                    'train_auc': float(getattr(model, 'train_auc', 0.5)),
                    'val_auc':   float(getattr(model, 'val_auc', 0.5)),
                }

            stock_signals[symbol] = {
                'prediction'       : pred,
                'regime'           : regime,
                'price'            : price,
                'sector'           : sector,
                'sector_multiplier': sector_mult,
                'model_auc_info'   : model_auc_info,
            }

        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")

    # ======================================================================
    # PHASE 2: CRYPTO
    # ======================================================================
    print("\n" + "="*60)
    print("PHASE 2: CRYPTO ANALYSIS")
    print("="*60)

    crypto_watchlist = ['BTC/USD', 'ETH/USD', 'SOL/USD']
    crypto_signals   = {}
    try:
        crypto_predictor = CryptoPredictor()
        crypto_signals   = crypto_predictor.run_full_pipeline(
            crypto_watchlist, lookback_days=365
        )
    except Exception as e:
        logger.warning(f"Crypto error: {e}")

    # ======================================================================
    # PHASE 3: SENTIMENT
    # ======================================================================
    print("\n" + "="*60)
    print("PHASE 3: SENTIMENT ANALYSIS")
    print("="*60)

    news_fetcher       = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    top_stocks         = sorted(
        stock_signals.items(),
        key=lambda x: x[1]['prediction'], reverse=True
    )[:7]
    top_symbols = [s[0] for s in top_stocks]
    sentiments  = {}
    try:
        all_news   = news_fetcher.fetch_all(top_symbols)
        sentiments = sentiment_analyzer.get_sentiment_for_stocks(all_news)
    except Exception as e:
        logger.warning(f"Sentiment error: {e}")

    # ======================================================================
    # PHASE 4: FINAL SIGNALS
    # ======================================================================
    print("\n" + "="*60)
    print("PHASE 4: FINAL SIGNALS")
    print("="*60)

    n_stocks = len(stock_signals)
    print(f"\n📊 Stock Signals ({n_stocks} stocks):")
    print("-"*72)

    dashboard_signals = {}  # FIX: populated for ALL signals, not just BUY

    for symbol, data in sorted(
        stock_signals.items(),
        key=lambda x: x[1]['prediction'], reverse=True
    ):
        pred      = data['prediction']
        regime    = data['regime']
        price     = data['price']
        sector    = data['sector']
        sect_mult = data['sector_multiplier']
        sent_score = sentiments.get(symbol, {}).get('sentiment_score', 0.0)

        signal, combined = compute_signal(
            pred, regime, sent_score, sect_mult,
            symbol, earnings_symbols
        )

        # Insider boost — only applies to combined score, not signal promotion
        insider_score = insider_scores.get(symbol, 0.0)
        if insider_score > 0:
            combined = min(combined + insider_score * 0.5, 1.0)  # FIX: halved weight
            if insider_score >= 0.10:
                print(f"  {symbol}: +{insider_score:.2f} insider boost (capped)")

        emoji = {
            'BUY': "🟢", 'AVOID': "🔴",
            'EARNINGS_HOLD': "📅", 'CAUTION': "⚠️"
        }.get(signal, "⚪")

        sect_emoji = "🔄↑" if sect_mult > 1.1 else ("🔄↓" if sect_mult < 0.9 else "")
        sent_emoji = "📈" if sent_score > 0.1 else ("📉" if sent_score < -0.1 else "")

        print(
            f"  {emoji} {symbol:6s}"
            f" | pred={pred:.3f}"
            f" | {regime:10s}"
            f" | sent={sent_score:+.2f}{sent_emoji}"
            f" | {sector:13s}{sect_emoji}"
            f" | {signal:14s}"
            f" | ${price:.2f}"
        )

        # S2 FIX: if market is in BEAR/CRASH, dashboard must show MARKET_HOLD
        # not BUY — the signal loop gates execution but not the dashboard label.
        if signal == 'BUY' and not market_regime['can_trade']:
            signal = 'MARKET_HOLD'

        # FIX: record ALL signals in dashboard, not just BUY
        dashboard_signals[symbol] = {
            'prediction': float(pred),
            'regime'    : regime,
            'price'     : float(price),
            'sentiment' : float(sent_score),
            'sector'    : sector,
            'signal'    : signal,
            'combined'  : float(combined),
        }

        if signal == 'BUY' and market_regime['can_trade']:

            mtf_score = mtf_scores.get(symbol, 0.5)
            if mtf_score < 0.5:
                print(f"    {symbol}: BUY blocked by MTF filter ({mtf_score:.0%})")
                dashboard_signals[symbol]['signal'] = 'MTF_HOLD'
                continue

            if not corr_filter.can_add_position(symbol, trader.positions):
                print(f"    {symbol}: BUY blocked by correlation filter")
                dashboard_signals[symbol]['signal'] = 'CORR_HOLD'
                continue

            veto_result = veto_agent.review_signal(
                symbol           = symbol,
                price            = price,
                prediction       = pred,
                regime           = regime,
                sentiment        = sent_score,
                sector           = sector,
                market_regime    = market_regime['regime'],
                mtf_score        = mtf_scores.get(symbol, 0.5),
                current_positions= trader.positions,
                vix              = market_regime.get('vix', 20),
            )

            if veto_result['decision'] == 'VETO':
                print(f"    {symbol}: VETOED — {veto_result['reason']}")
                dashboard_signals[symbol]['signal'] = 'VETOED'
                continue

            # ATR with explicit guard — never pass None to open_position
            try:
                atr = calc_atr(stock_data, symbol)
            except ValueError as e:
                logger.warning(f"ATR error for {symbol}: {e} — using price*0.02 fallback")
                atr = price * 0.02  # 2% of price as a safe fallback

            opened = trader.open_position(
                symbol, price, combined, reason=regime, atr=atr,
                regime_conf=market_regime.get('confidence', 1.0),
            )
            if opened:
                telegram.alert_buy_signal(symbol, price, pred, regime, sent_score)

    # ======================================================================
    # CRYPTO SIGNALS
    # ======================================================================
    print("\n🪙 Crypto Signals:")
    print("-"*60)
    if crypto_signals:
        for symbol, data in crypto_signals.items():
            pred   = data['prediction']
            regime = data['regime']
            price  = data['price']
            signal = 'HOLD'
            # C3 FIX: use settings.BUY_THRESHOLD for crypto too
            if regime == 'uptrend' and pred > settings.BUY_THRESHOLD:
                signal = 'BUY'
            elif regime == 'downtrend':
                signal = 'AVOID'

            emoji = "🟢" if signal == 'BUY' else ("🔴" if signal == 'AVOID' else "⚪")
            print(
                f"  {emoji} {symbol:10s}"
                f" | {pred:.3f}"
                f" | {regime:10s}"
                f" | {signal}"
                f" | ${price:,.2f}"
            )

            dashboard_signals[symbol] = {
                'prediction': float(pred),
                'regime'    : regime,
                'price'     : float(price),
                'sentiment' : 0.0,
                'sector'    : 'Crypto',
                'signal'    : signal,
                'combined'  : float(pred),
            }

            if signal == 'BUY':
                # Apply the same circuit-breaker gate as stocks — a portfolio
                # halt must stop crypto too, not just equities.
                if not market_regime['can_trade']:
                    dashboard_signals[symbol]['signal'] = 'MARKET_HOLD'
                    print(f"    {symbol}: crypto BUY blocked by circuit breaker (CASH MODE)")
                    continue
                opened = trader.open_position(symbol, price, pred, reason=regime)
                if opened:
                    telegram.alert_buy_signal(symbol, price, pred, regime, 0.0)
    else:
        print("  No crypto signals generated")

    # ======================================================================
    # PHASE 5: POSITION MANAGEMENT
    # ======================================================================
    print("\n" + "="*60)
    print("PHASE 5: POSITION MANAGEMENT")
    print("="*60)

    current_prices = {s: d['price'] for s, d in stock_signals.items()}
    current_prices.update({s: d['price'] for s, d in crypto_signals.items()})

    if trader.positions:
        print("\n  Checking stop loss / take profit...")
        for symbol in list(trader.positions.keys()):
            if symbol in current_prices:
                pos   = trader.positions.get(symbol, {})
                entry = pos.get('entry_price', 0)
                trader.update_position(symbol, current_prices[symbol])
                if symbol not in trader.positions:
                    exit_price = current_prices[symbol]
                    pnl = (exit_price - entry) * pos.get('shares', 0)
                    if pnl < 0:
                        telegram.alert_stop_loss(symbol, exit_price, pnl)
                    else:
                        telegram.alert_take_profit(symbol, exit_price, pnl)
    else:
        print("\n  No open positions to manage")

    # ======================================================================
    # SAVE FOR DASHBOARD — atomic writes only
    # ======================================================================
    atomic_json_write('logs/latest_signals.json', dashboard_signals)
    atomic_json_write('logs/earnings.json', earnings_soon)
    atomic_json_write('logs/sectors.json', {
        sector: {
            'score'       : float(d['score']),
            'flow'        : d['flow'],
            'momentum_21d': float(d['momentum_21d']),
        }
        for sector, d in sector_scores.items()
    })
    print("\n  💾 All data saved for dashboard (atomic)")

    # AUC tracking for walk-forward monitor / live-readiness check.
    # Kept as its own file rather than added to latest_signals.json,
    # since several consumers of that file iterate its top-level keys
    # assuming every one is a stock symbol.
    _auc_samples = [
        v['model_auc_info'] for v in stock_signals.values()
        if v.get('model_auc_info')
    ]
    if _auc_samples:
        atomic_json_write('logs/model_auc.json', {
            'training_auc': sum(s['train_auc'] for s in _auc_samples) / len(_auc_samples),
            'updated_at'  : now,   # FIX: now is already a formatted string
        })

    # ======================================================================
    # PORTFOLIO SUMMARY
    # ======================================================================
    trader.get_summary(current_prices)
    for symbol, pos in trader.positions.items():
        pos['current_price'] = current_prices.get(symbol, pos['entry_price'])
    trader.save_state()

    total_value = trader.capital + sum(
        pos.get('shares', 0) * current_prices.get(symbol, pos.get('entry_price', 0))
        for symbol, pos in trader.positions.items()
    )
    total_pnl = total_value - trader.starting_capital
    total_pct = total_pnl / trader.starting_capital

    positions_with_pnl = {}
    for symbol, pos in trader.positions.items():
        curr_price  = current_prices.get(symbol, pos.get('entry_price', 0))
        entry_price = pos.get('entry_price', 0)
        shares      = pos.get('shares', 0)
        pnl         = (curr_price - entry_price) * shares
        pnl_pct     = (curr_price - entry_price) / entry_price if entry_price > 0 else 0
        positions_with_pnl[symbol] = {
            **pos,
            'current_price': curr_price,
            'pnl'          : pnl,
            'pnl_pct'      : pnl_pct,
        }

    telegram.alert_daily_summary(total_value, total_pnl, total_pct,
                                  positions_with_pnl, dashboard_signals)

    print("\n" + "✅" * 25)
    print("ALPHA EDGE V4 SCAN COMPLETE")
    print("✅" * 25)
    print(f"\nScanned: {len(stock_signals)} stocks + {len(crypto_signals)} crypto")
    print(f"Earnings this week: {len(earnings_soon)} stocks")
    print(f"  Models: 5 per stock (XGB+LGB+RF+CatBoost+LSTM)")
    print("\nTo view dashboard:")
    print("  python run_dashboard.py")
    print("  Open http://localhost:8050\n")

    # ======================================================================
    # WEEKLY PERFORMANCE REPORT + CRITIC REVIEW
    # ======================================================================
    analytics = PerformanceAnalytics()
    if analytics.should_run_today():
        print("\n  Sending Weekly Performance Report...")
        analytics.send_report(telegram)

    critic = CriticAgent()
    critic.run_weekly_review(
        trade_history    = trader.trade_history,
        portfolio_value  = total_value,
        starting_capital = trader.starting_capital,
        telegram_bot     = telegram,
    )

    logger.info("run_daily_scan() complete — %d signals generated", len(stock_signals))
    return stock_signals


    run_daily_scan()
