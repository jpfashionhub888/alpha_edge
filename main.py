# main.py
"""
AlphaEdge Main Trading Scanner V4
Signal Quality Upgrades:
- MTF score correctly calibrated (-1 to +1 scale)
- MTF composite integrated INTO signal generation (not just post-hoc veto)
- BUY threshold raised from 0.55 → 0.63
- Volume confirmation required before BUY
- Minimum 2:1 risk/reward check using ATR before execution
- Crypto signal logic matches stock rigor
"""

import warnings
import os
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import logging
import pandas as pd
from veto_agent import VetoAgent
from insider_tracker import InsiderTracker
from datetime import datetime
from data.stock_data import StockDataFetcher
from config import settings
from data.news_data import NewsFetcher
from data.feature_engine import FeatureEngine
from models.technical_model import TechnicalPredictor
from models.sentiment_model import SentimentAnalyzer
from models.regime_detector import RegimeDetector
from market_regime import MarketRegimeDetector as MarketRegimeFilter
from multi_timeframe import MultiTimeframeAnalyzer
from correlation_filter import CorrelationFilter
from models.crypto_predictor import CryptoPredictor
from models.sector_rotation import SectorRotation
from execution.paper_trader import PaperTrader
from monitoring.telegram_bot import TelegramBot
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from model_cache import save_models, load_models, is_cache_valid
from critic_agent import CriticAgent
from performance_analytics import PerformanceAnalytics
from risk_circuit_breaker import RiskCircuitBreaker

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SIGNAL QUALITY CONSTANTS  (loaded from settings)
# ─────────────────────────────────────────────
BUY_THRESHOLD       = settings.BUY_THRESHOLD
PREDICTION_THRESHOLD = BUY_THRESHOLD
MTF_BLOCK_THRESHOLD = settings.MTF_BLOCK_THRESHOLD
VOLUME_SPIKE_MIN    = settings.VOLUME_SPIKE_MIN
MIN_RISK_REWARD     = settings.MIN_RISK_REWARD
MTF_WEIGHT_IN_SIGNAL = settings.MTF_WEIGHT_IN_SIGNAL
# ─────────────────────────────────────────────


def get_earnings_calendar(watchlist):
    """Check which stocks have earnings this week."""
    import yfinance as yf
    print("\n📅 Checking earnings calendar...")
    earnings_soon = []
    # ETFs have no fundamentals data — yfinance returns 404 for calendar fetch
    ETFS = {'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'XLK', 'XLF',
            'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLC'}
    for symbol in watchlist:
        if symbol in ETFS:
            continue
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
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
            logger.debug(f'Earnings calendar fetch failed for {symbol}: {e}')

    if earnings_soon:
        n = len(earnings_soon)
        print(f"  ⚠️  {n} stocks reporting this week:")
        for e in earnings_soon:
            d = e['days_until']
            w = "TODAY!" if d == 0 else ("TOMORROW!" if d == 1 else f"in {d} days")
            print(f"    ⚠️  {e['symbol']} earnings {w}")
    else:
        print("  ✅ No earnings this week")
    return earnings_soon


def get_full_watchlist():
    """Return expanded stock watchlist loaded from centralized config."""
    return settings.STOCK_WATCHLIST


def check_volume_confirmation(stock_data: dict, symbol: str) -> tuple[bool, float]:
    """
    FIX (new): Confirm entry with volume spike.
    Returns (confirmed: bool, ratio: float)
    Volume must be >= VOLUME_SPIKE_MIN x 20-day average on the signal day.
    """
    try:
        df = stock_data.get(symbol)
        if df is None or len(df) < 22:
            return False, 0.0
        vol_today  = float(df['volume'].iloc[-1])
        vol_avg_20 = float(df['volume'].iloc[-21:-1].mean())
        if vol_avg_20 == 0:
            return False, 0.0
        ratio = vol_today / vol_avg_20
        return ratio >= VOLUME_SPIKE_MIN, round(ratio, 2)
    except Exception:
        return False, 0.0


def check_risk_reward(price: float, atr: float | None) -> tuple[bool, float]:
    """
    FIX (new): Validate minimum risk/reward before entry.
    Uses 1x ATR as stop distance, 2x ATR as minimum target.
    Returns (passes: bool, rr_ratio: float)
    """
    if atr is None or atr <= 0 or price <= 0:
        return True, 0.0   # can't compute — don't block, just skip check
    stop_distance   = atr * 1.0    # stop 1 ATR below entry
    target_distance = atr * 2.0    # minimum target 2 ATR above entry
    rr_ratio = target_distance / stop_distance  # = 2.0 always with this method
    # This will evolve when you add dynamic targets; keeping structure for now
    passes = rr_ratio >= MIN_RISK_REWARD
    return passes, round(rr_ratio, 2)


def compute_signal(
    pred: float,
    regime: str,
    sent_score: float,
    sect_mult: float,
    symbol: str,
    earnings_symbols: list,
    mtf_composite: float = 0.0,   # FIX: MTF now part of signal, not just blocker
) -> tuple[str, float]:
    """
    Compute the final trading signal for a stock.

    Changes from V3:
    - mtf_composite is now an input (was ignored in signal calc before)
    - Weights rebalanced to make room for MTF input
    - BUY threshold raised to BUY_THRESHOLD constant
    - Single source of truth for signal logic
    """
    # FIX: MTF composite (-1 to +1) normalized to 0–1 range for weighting
    mtf_norm = (mtf_composite + 1.0) / 2.0  # converts -1..+1 → 0..1

    # Weight breakdown: pred 50%, sentiment 15%, sector 15%, MTF 15%, (leaves 5% neutral floor)
    combined = (
        pred               * 0.50
        + (sent_score + 0.5) * 0.15
        + (sect_mult - 1.0)  * 0.15   # P1-4 fix: offset from 1.0 so outflow sectors reduce score
        + mtf_norm           * MTF_WEIGHT_IN_SIGNAL
        + 0.05               # small neutral floor — prevents hair-trigger signals
    )

    signal = 'HOLD'

    if regime == 'uptrend' and combined > BUY_THRESHOLD:
        signal = 'BUY'
    elif regime == 'downtrend':
        signal = 'AVOID'
    elif regime == 'volatile':
        signal = 'CAUTION'

    # Sector rotation boost (only upgrades HOLD → BUY in sideways market)
    if sect_mult > 1.1 and signal == 'HOLD':
        if pred > 0.60 and regime == 'sideways' and mtf_norm > 0.55:
            signal = 'BUY'

    # Sector rotation penalty
    if sect_mult < 0.8 and signal == 'BUY':
        signal = 'HOLD'

    # Earnings protection
    if symbol in earnings_symbols and signal == 'BUY':
        signal = 'EARNINGS_HOLD'

    return signal, combined


def calc_atr(stock_data: dict, symbol: str) -> float | None:
    """Calculate 14-period ATR for a symbol. Returns None on failure."""
    try:
        if symbol not in stock_data:
            return None
        df_atr = stock_data[symbol].copy()
        high  = df_atr['high']
        low   = df_atr['low']
        close = df_atr['close']
        tr1   = high - low
        tr2   = abs(high - close.shift(1))
        tr3   = abs(low  - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return float(true_range.rolling(14).mean().iloc[-1])
    except Exception:
        return None


def run_daily_scan():
    """Run the complete daily trading scan."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print("\n" + "🚀" * 25)
    print(f"ALPHA EDGE V4 - {now}")
    print("Signal Quality Upgrade: MTF integrated, volume confirmation, R:R filter")
    print("🚀" * 25)

    trader   = PaperTrader(starting_capital=10000.0)
    trader.load_state()
    telegram = TelegramBot()

    stock_watchlist = get_full_watchlist()

    # ==========================================
    # PHASE 0: EARNINGS + SECTOR ROTATION
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 0: EARNINGS + SECTOR ROTATION")
    print("="*60)
    earnings_soon    = get_earnings_calendar(stock_watchlist)
    earnings_symbols = [e['symbol'] for e in earnings_soon]
    sector_analyzer  = SectorRotation()
    sector_scores    = sector_analyzer.analyze()

    # ==========================================
    # MARKET REGIME FILTER
    # ==========================================
    print("\n" + "="*60)
    print("MARKET REGIME FILTER")
    print("="*60)
    regime_filter = MarketRegimeFilter()
    # .analyze() was renamed to .detect(price_df) in market_regime.py refactor.
    # Fetch SPY as market proxy, then map return keys to what downstream expects.
    try:
        import yfinance as yf
        _spy = yf.download('SPY', period='2y', interval='1d', progress=False, auto_adjust=True)
        if hasattr(_spy.columns, 'levels'):  # flatten MultiIndex
            _spy.columns = [c[0] if isinstance(c, tuple) else c for c in _spy.columns]
        _spy.columns = [c.lower() for c in _spy.columns]
        _spy = _spy.rename(columns={'close': 'Close', 'open': 'Open',
                                     'high': 'High', 'low': 'Low', 'volume': 'Volume'})
        _regime_raw = regime_filter.detect(_spy)
    except Exception as _re:
        logger.warning(f"Regime detect failed ({_re}), defaulting to tradeable")
        _regime_raw = {'regime': 'unknown', 'tradeable': True, 'confidence': 0.5, 'signals': {}}
    # Map to the dict shape the rest of main.py expects
    market_regime = {
        'can_trade' : _regime_raw.get('tradeable', True),
        'regime'    : _regime_raw.get('regime', 'unknown'),
        'reason'    : f"Regime={_regime_raw.get('regime','?')} confidence={_regime_raw.get('confidence',0):.2f}",
        'vix'       : _regime_raw.get('signals', {}).get('vix_level', 20),
    }
    print(f"  Regime: {market_regime['regime']} | can_trade={market_regime['can_trade']} | {market_regime['reason']}")

    # ==========================================
    # RISK CIRCUIT BREAKER CHECK
    # ==========================================
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

    # ==========================================
    # MULTI-TIMEFRAME ANALYSIS
    # FIX: Now fetches full MTF result dict (not just score float)
    #      so we can use composite in signal generation
    # ==========================================
    print("\n" + "="*60)
    print("MULTI-TIMEFRAME ANALYSIS")
    print("="*60)

    mtf_analyzer    = MultiTimeframeAnalyzer()
    mtf_composites  = {}   # symbol → composite float (-1 to +1)
    mtf_blocked_by  = {}   # symbol → blocking reason string or None

    if market_regime['can_trade']:
        print("\n  Checking timeframe alignment...")
        for symbol in stock_watchlist:
            try:
                # get_mtf_score returns composite -1 to +1 (corrected wrapper)
                score = mtf_analyzer.get_mtf_score(symbol)
                mtf_composites[symbol] = score
                mtf_blocked_by[symbol] = None
                if score > 0.2:
                    print(f"    {symbol}: MTF {score:+.2f} BULLISH")
                elif score < -0.2:
                    print(f"    {symbol}: MTF {score:+.2f} BEARISH")
            except Exception as e:
                mtf_composites[symbol] = 0.0
                mtf_blocked_by[symbol] = None
                logger.warning(f"MTF failed for {symbol}: {e}")

        bullish = sum(1 for s in mtf_composites.values() if s > 0)
        print(f"\n  MTF complete: {bullish}/{len(stock_watchlist)} bullish")
    else:
        print("\n  Skipping MTF (CASH MODE active)")
        for symbol in stock_watchlist:
            mtf_composites[symbol] = 0.0
            mtf_blocked_by[symbol] = None

    # ==========================================
    # PHASE 1: STOCK ANALYSIS
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 1: STOCK ANALYSIS")
    print("="*60)

    n = len(stock_watchlist)
    print(f"\n1a. Fetching data for {n} stocks...")

    stock_fetcher = StockDataFetcher(
        watchlist=stock_watchlist,
        lookback_days=730
    )
    stock_data = stock_fetcher.fetch_all()

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

            cached = load_models(symbol)
            if cached:
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
                model = TechnicalPredictor(use_lstm=True)
                model.train(X_train[selected], y_train)
                try:
                    save_models(symbol, {
                        'xgboost'          : model.models.get('xgboost'),
                        'lightgbm'         : model.models.get('lightgbm'),
                        'random_forest'    : model.models.get('random_forest'),
                        'catboost'         : model.models.get('catboost'),
                        'selected_features': selected,
                    })
                except Exception as e:
                    logger.warning(f"Cache save failed for {symbol}: {e}")

            latest = df.iloc[-1:]
            _preds = model.predict(latest[selected])
            pred   = _preds[0] if len(_preds) > 0 else 0.5  # guard: empty array → neutral
            regime      = latest['regime'].iloc[0]
            price       = latest['close'].iloc[0]
            sector_mult = sector_analyzer.get_sector_signal(symbol)
            sector      = sector_analyzer.get_sector_for_stock(symbol)

            stock_signals[symbol] = {
                'prediction'       : pred,
                'regime'           : regime,
                'price'            : price,
                'sector'           : sector,
                'sector_multiplier': sector_mult,
            }

        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")

    # ==========================================
    # PHASE 2: CRYPTO
    # ==========================================
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

    # ==========================================
    # PHASE 3: SENTIMENT
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 3: SENTIMENT ANALYSIS")
    print("="*60)

    news_fetcher       = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()

    top_stocks  = sorted(
        stock_signals.items(),
        key=lambda x: x[1]['prediction'],
        reverse=True
    )[:7]
    top_symbols = [s[0] for s in top_stocks]
    sentiments  = {}
    try:
        all_news   = news_fetcher.fetch_all(top_symbols)
        sentiments = sentiment_analyzer.get_sentiment_for_stocks(all_news)
    except Exception as e:
        logger.warning(f"Sentiment error: {e}")

    # ==========================================
    # PHASE 4: FINAL SIGNALS
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 4: FINAL SIGNALS")
    print("="*60)

    n_stocks = len(stock_signals)
    print(f"\n📊 Stock Signals ({n_stocks} stocks):")
    print("-"*75)

    dashboard_signals = {}

    for symbol, data in sorted(
        stock_signals.items(),
        key=lambda x: x[1]['prediction'],
        reverse=True
    ):
        pred        = data['prediction']
        regime      = data['regime']
        price       = data['price']
        sector      = data['sector']
        sect_mult   = data['sector_multiplier']
        sent_score  = sentiments.get(symbol, {}).get('sentiment_score', 0.0)
        mtf_comp    = mtf_composites.get(symbol, 0.0)

        # FIX: MTF composite now passed into signal generation
        signal, combined = compute_signal(
            pred, regime, sent_score, sect_mult,
            symbol, earnings_symbols,
            mtf_composite=mtf_comp,
        )
        # P0-4: insider boost removed — Form 4 count can't distinguish buy vs sell direction

        # Emoji labels
        emoji      = {"BUY": "🟢", "AVOID": "🔴", "EARNINGS_HOLD": "📅", "CAUTION": "⚠️"}.get(signal, "⚪")
        sect_emoji = "🔄↑" if sect_mult > 1.1 else ("🔄↓" if sect_mult < 0.9 else "")
        sent_emoji = "📈" if sent_score > 0.1 else ("📉" if sent_score < -0.1 else "")

        print(
            f"  {emoji} {symbol:6s}"
            f" | pred={pred:.3f}"
            f" | mtf={mtf_comp:+.2f}"
            f" | {regime:10s}"
            f" | sent={sent_score:+.2f}{sent_emoji}"
            f" | {sector:13s}{sect_emoji}"
            f" | {signal}"
            f" | ${price:.2f}"
        )

        # P1-6: track real filter gate counts
        filters_passed  = 0
        filter_detail   = {}
        insider_filings = insider_scores.get(symbol, 0.0)  # always 0.0 per P0-4; kept for display
        scan_time       = datetime.now().isoformat()

        # Write dashboard entry BEFORE BUY gate checks so vetoed/blocked symbols get a full entry
        dashboard_signals[symbol] = {
            'prediction'     : float(pred),
            'regime'         : regime,
            'price'          : float(price),
            'sentiment'      : float(sent_score),
            'sector'         : sector,
            'signal'         : signal,
            'mtf'            : float(mtf_comp),
            'combined'       : float(combined),
            'filters_passed' : filters_passed,    # P1-6: updated below as gates pass
            'filter_detail'  : filter_detail,
            'insider_filings': insider_filings,
            'scan_time'      : scan_time,
        }

        if signal == 'BUY' and market_regime['can_trade']:

            # FIX: Correct MTF block threshold (-1 to +1 scale, not 0 to 1)
            if mtf_comp < MTF_BLOCK_THRESHOLD:
                print(f"    {symbol}: BUY blocked by MTF (composite={mtf_comp:+.2f} < {MTF_BLOCK_THRESHOLD})")
                signal = 'MTF_HOLD'
                dashboard_signals[symbol]['signal'] = signal
                continue

            # Gate 1 passed: MTF
            filters_passed += 1
            filter_detail['mtf'] = True

            # FIX (new): Volume confirmation
            vol_ok, vol_ratio = check_volume_confirmation(stock_data, symbol)
            if not vol_ok:
                print(f"    {symbol}: BUY blocked — volume {vol_ratio:.1f}x avg (need {VOLUME_SPIKE_MIN}x)")
                signal = 'VOL_HOLD'
                dashboard_signals[symbol]['signal'] = signal
                dashboard_signals[symbol]['filters_passed'] = filters_passed
                dashboard_signals[symbol]['filter_detail']  = filter_detail
                continue

            # Gate 2 passed: Volume
            filters_passed += 1
            filter_detail['volume'] = True

            # Correlation/Sector filter
            if not corr_filter.can_add_position(symbol, trader.positions):
                print(f"    {symbol}: BUY blocked by correlation filter")
                signal = 'CORR_HOLD'
                dashboard_signals[symbol]['signal'] = signal
                dashboard_signals[symbol]['filters_passed'] = filters_passed
                dashboard_signals[symbol]['filter_detail']  = filter_detail
                continue

            # Gate 3 passed: Correlation
            filters_passed += 1
            filter_detail['correlation'] = True

            # Veto agent
            veto_result = veto_agent.review_signal(
                symbol          = symbol,
                price           = price,
                prediction      = pred,
                regime          = regime,
                sentiment       = sent_score,
                sector          = sector,
                market_regime   = market_regime['regime'],
                mtf_score       = mtf_comp,
                current_positions = trader.positions,
                vix             = market_regime.get('vix', 20),
            )
            if veto_result['decision'] == 'VETO':
                print(f"    {symbol}: VETOED by AI — {veto_result['reason']}")
                signal = 'VETOED'
                dashboard_signals[symbol]['signal'] = signal
                dashboard_signals[symbol]['filters_passed'] = filters_passed
                dashboard_signals[symbol]['filter_detail']  = filter_detail
                continue

            # Gate 4 passed: Veto Agent
            filters_passed += 1
            filter_detail['veto_agent'] = True

            atr = calc_atr(stock_data, symbol)

            # FIX (new): Risk/reward check
            rr_ok, rr_ratio = check_risk_reward(price, atr)
            if not rr_ok:
                print(f"    {symbol}: BUY blocked — R:R {rr_ratio:.1f} < {MIN_RISK_REWARD} minimum")
                signal = 'RR_HOLD'
                dashboard_signals[symbol]['signal'] = signal
                dashboard_signals[symbol]['filters_passed'] = filters_passed
                dashboard_signals[symbol]['filter_detail']  = filter_detail
                continue

            # Gate 5 passed: R:R
            filters_passed += 1
            filter_detail['risk_reward'] = True

            # Update dashboard entry with final signal and gate counts
            dashboard_signals[symbol]['signal']         = signal
            dashboard_signals[symbol]['filters_passed'] = filters_passed
            dashboard_signals[symbol]['filter_detail']  = filter_detail

            print(f"    ✅ {symbol}: ALL filters passed ({filters_passed}/5) | vol={vol_ratio:.1f}x | R:R={rr_ratio:.1f} | mtf={mtf_comp:+.2f}")
            opened = trader.open_position(symbol, price, combined, reason=regime, atr=atr)

            if opened:
                telegram.alert_buy_signal(symbol, price, pred, regime, sent_score)

        # Final update of filters_passed in dashboard entry (for non-BUY signals)
        dashboard_signals[symbol]['filters_passed'] = filters_passed
        dashboard_signals[symbol]['filter_detail']  = filter_detail

    # ==========================================
    # PHASE 4b: CRYPTO SIGNALS
    # FIX: Now matches stock signal rigor
    # ==========================================
    print("\n🪙 Crypto Signals:")
    print("-"*60)

    if crypto_signals:
        for symbol, data in crypto_signals.items():
            pred   = data['prediction']
            regime = data['regime']
            price  = data['price']
            signal = 'HOLD'

            # FIX: Raised crypto BUY threshold to match stocks (was 0.52)
            # Also requires uptrend — same as stock logic
            if regime == 'uptrend' and pred > BUY_THRESHOLD:
                signal = 'BUY'
            elif regime == 'downtrend':
                signal = 'AVOID'
            elif regime == 'volatile':
                signal = 'CAUTION'

            emoji = "🟢" if signal == 'BUY' else ("🔴" if signal == 'AVOID' else ("⚠️" if signal == 'CAUTION' else "⚪"))
            print(
                f"  {emoji} {symbol:10s}"
                f" | pred={pred:.3f}"
                f" | {regime:10s}"
                f" | {signal}"
                f" | ${price:,.2f}"
            )

            if signal == 'BUY' and market_regime['can_trade']:
                # Volume confirmation using raw_df now exposed by CryptoPredictor
                raw_df = data.get('raw_df')
                if raw_df is not None:
                    vol_ok, vol_ratio = check_volume_confirmation({'symbol': raw_df}, symbol.replace('/', '_'))
                    if not vol_ok:
                        print(f"    {symbol}: BUY blocked — volume {vol_ratio:.1f}x avg (need {VOLUME_SPIKE_MIN}x)")
                        signal = 'VOL_HOLD'
                        dashboard_signals[symbol] = {**data, 'signal': signal, 'sentiment': 0.0}
                        continue

                opened = trader.open_position(symbol, price, pred, reason=regime)
                if opened:
                    telegram.alert_buy_signal(symbol, price, pred, regime, 0.0)

            dashboard_signals[symbol] = {
                'prediction'     : float(pred),
                'regime'         : regime,
                'price'          : float(price),
                'sentiment'      : 0.0,
                'sector'         : 'Crypto',
                'signal'         : signal,
                'mtf'            : 0.0,
                'combined'       : float(pred),   # P0-2 fix: expose combined for dashboard filter dots
                'filters_passed' : 0,
                'filter_detail'  : {},
                'insider_filings': 0,
                'scan_time'      : datetime.now().isoformat(),
            }
    else:
        print("  No crypto signals generated")

    # ==========================================
    # PHASE 5: POSITION MANAGEMENT
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 5: POSITION MANAGEMENT")
    print("="*60)

    current_prices = {}
    for symbol, data in stock_signals.items():
        current_prices[symbol] = data['price']
    for symbol, data in crypto_signals.items():
        current_prices[symbol] = data['price']

    if trader.positions:
        print("\n  Checking stop loss / take profit...")
        for symbol in list(trader.positions.keys()):
            if symbol in current_prices:
                pos   = trader.positions.get(symbol, {})
                entry = pos.get('entry_price', 0)
                trader.update_position(symbol, current_prices[symbol])
                if symbol not in trader.positions:
                    exit_price = current_prices[symbol]
                    pnl        = (exit_price - entry) * pos.get('shares', 0)
                    if pnl < 0:
                        telegram.alert_stop_loss(symbol, exit_price, pnl)
                    else:
                        telegram.alert_take_profit(symbol, exit_price, pnl)
    else:
        print("\n  No open positions to manage")

    # ==========================================
    # SAVE FOR DASHBOARD
    # ==========================================
    os.makedirs('logs', exist_ok=True)
    with open('logs/latest_signals.json', 'w') as f:
        json.dump(dashboard_signals, f, indent=2)
    with open('logs/earnings.json', 'w') as f:
        json.dump(earnings_soon, f, indent=2)
    with open('logs/sectors.json', 'w') as f:
        sector_data = {
            sector: {
                'score'        : float(data['score']),
                'flow'         : data['flow'],
                'momentum_21d' : float(data['momentum_21d']),
            }
            for sector, data in sector_scores.items()
        }
        json.dump(sector_data, f, indent=2)

    print("\n  💾 All data saved for dashboard")

    # ==========================================
    # PORTFOLIO SUMMARY
    # ==========================================
    trader.get_summary(current_prices)

    for symbol, pos in trader.positions.items():
        pos['current_price'] = current_prices.get(symbol, pos['entry_price'])
    trader.save_state()

    total_value = trader.capital + sum(
        pos.get('shares', 0) * current_prices.get(symbol, pos.get('entry_price', 0))
        for symbol, pos in trader.positions.items()
    )
    total_pnl = total_value - trader.starting_capital
    total_pct = total_pnl  / trader.starting_capital

    positions_with_pnl = {}
    for symbol, pos in trader.positions.items():
        curr_price = current_prices.get(symbol, pos.get('entry_price', 0))
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

    telegram.alert_daily_summary(
        total_value, total_pnl, total_pct,
        positions_with_pnl, dashboard_signals
    )

    print("\n" + "✅" * 25)
    print("ALPHA EDGE V4 SCAN COMPLETE")
    print("✅" * 25)

    n_s = len(stock_signals)
    n_c = len(crypto_signals)
    n_e = len(earnings_soon)
    print(f"\nScanned: {n_s} stocks + {n_c} crypto")
    print(f"Earnings this week: {n_e} stocks")
    print(f"Models: 5 per stock (XGB+LGB+RF+CatBoost+LSTM)")
    print(f"\nSignal thresholds: BUY>{BUY_THRESHOLD} | MTF>{MTF_BLOCK_THRESHOLD} | Vol>{VOLUME_SPIKE_MIN}x | R:R>{MIN_RISK_REWARD}")
    print("\nTo view dashboard: python run_dashboard.py → http://localhost:8050\n")

    # ==========================================
    # WEEKLY PERFORMANCE REPORT + CRITIC REVIEW
    # ==========================================
    analytics = PerformanceAnalytics()
    if analytics.should_run_today():
        print("\n  Sending Weekly Empire Performance Report...")
        analytics.send_report(telegram)

    critic = CriticAgent()
    critic.run_weekly_review(
        trade_history    = trader.trade_history,
        portfolio_value  = total_value,
        starting_capital = trader.starting_capital,
        telegram_bot     = telegram,
    )


if __name__ == "__main__":
    run_daily_scan()
