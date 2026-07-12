# alpaca_live.py
"""
AlphaEdge — Alpaca Stock Trading Loop

Fix 2.3: Runs exactly once per day at 16:15 ET (after market close).
A daily-bar model has no new signal information within the trading day;
running every 30 minutes wastes API quota and retrains models unnecessarily.

Modes:
  Paper (default): trades on Alpaca paper account
  Live:            trades on Alpaca live account

How to run:
  set ALPACA_API_KEY=your_key
  set ALPACA_SECRET_KEY=your_secret
  set ALPACA_BASE_URL=https://paper-api.alpaca.markets
  python alpaca_live.py

Scan schedule: once daily at 16:15 ET (15 min after market close)
"""

import os
from config import settings
import time
import logging
import warnings
import threading
from datetime import datetime, time as dtime
import pytz

from risk_circuit_breaker import RiskCircuitBreaker
from risk.position_sizer import PositionSizer, get_trade_stats_for_sizing

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────
# Fix 2.3: Once-daily scan at 16:15 ET (after all daily bars are final)
DAILY_SCAN_HOUR     = 16      # 4 PM ET
DAILY_SCAN_MIN      = 15      # 16:15 ET — 15 min after market close
RISK_PER_TRADE_PCT  = settings.MAX_RISK_PER_TRADE   # was hardcoded 0.015, now matches settings.yaml / paper_trader.py
MAX_POSITIONS       = 5       # max open positions at once
MARKET_TZ           = pytz.timezone('America/New_York')
MARKET_OPEN         = dtime(9, 30)
MARKET_CLOSE        = dtime(16, 0)
# ─────────────────────────────────────────────────────────────────────


def _seconds_until_next_scan() -> float:
    """
    Fix 2.3: Calculate seconds until the next 16:15 ET scan window.
    If it's already past 16:15 today, schedule for tomorrow at 16:15.
    """
    now_et    = datetime.now(MARKET_TZ)
    target_et = now_et.replace(
        hour=DAILY_SCAN_HOUR, minute=DAILY_SCAN_MIN, second=0, microsecond=0
    )
    if now_et >= target_et:
        # Already past today's scan time — wait for tomorrow
        from datetime import timedelta
        target_et += timedelta(days=1)
    delta = (target_et - now_et).total_seconds()
    return max(delta, 1.0)


def is_market_hours() -> bool:
    """Return True if current time is within NYSE market hours."""
    now_et = datetime.now(MARKET_TZ).time()
    return MARKET_OPEN <= now_et <= MARKET_CLOSE


def get_mode() -> str:
    """Detect paper vs live from ALPACA_BASE_URL env var."""
    url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    return 'PAPER' if 'paper' in url.lower() else 'LIVE'


class AlpacaLiveTrader:
    """
    Stock trading loop using Alpaca as execution backend.
    Uses existing main.py signal engine — no duplicate logic.
    """

    def __init__(self):
        from execution.alpaca_broker        import AlpacaBroker
        from monitoring.telegram_bot        import TelegramBot
        from monitoring.heartbeat           import HeartbeatMonitor

        self.broker          = AlpacaBroker()
        self.telegram        = TelegramBot()
        self.mode            = get_mode()
        self.circuit_breaker = RiskCircuitBreaker()

        # Phase 4: heartbeat monitor — writes logs/heartbeats/alpaca_bot.json
        self.heartbeat = HeartbeatMonitor(service_name='alpaca_bot')
        self.heartbeat.start()

        # Track our own stop/target levels since Alpaca paper
        # doesn't support bracket orders on all account types
        # symbol -> {entry, stop, target, shares, dollar_value}
        self.managed_positions: dict[str, dict] = {}

        # BUG 4 FIX: Restore stop/target levels that survived a bot restart.
        # Without this, managed_positions is always empty after systemd restarts
        # and the position monitor never fires stops/targets for existing positions.
        try:
            import json as _json
            _mp_file = 'logs/managed_positions.json'
            if os.path.exists(_mp_file):
                with open(_mp_file) as _f:
                    self.managed_positions = _json.load(_f)
                if self.managed_positions:
                    logger.info(
                        f'Restored {len(self.managed_positions)} managed positions '
                        f'from disk: {list(self.managed_positions.keys())}'
                    )
        except Exception as _e:
            logger.warning(f'Could not restore managed_positions from disk: {_e}')

        # Price monitor thread
        self._monitor_running = False

    # ── Startup ──────────────────────────────────────────────────────

    def start(self):
        print('\n' + '🚀' * 25)
        print(f'ALPHAEDGE ALPACA STOCKS  —  {datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M ET")}')
        print(f'Mode: {self.mode}  |  Scan interval: 16:15 ET daily')
        print(f'Max positions: {MAX_POSITIONS}  |  Risk per trade: {RISK_PER_TRADE_PCT*100:.1f}%')
        print('🚀' * 25)

        if not self.broker.connected:
            print('\n❌ Alpaca not connected. Check API keys and try again.')
            print('   Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL')
            return

        # Show account status
        self._print_account()

        # Phase 4: startup reconciliation
        # Enforced, not advisory: a phantom/orphan position after a crash
        # is exactly the double-buy scenario this module exists to catch.
        # Logging it and trading anyway (the previous behavior) makes the
        # check pointless — the Telegram alert literally says "do NOT
        # place new trades until resolved" while the code did precisely
        # that. Set ALPHAEDGE_FORCE_START=1 to override after manual review.
        try:
            from monitoring.reconciliation import reconcile_on_startup
            discrepancies = reconcile_on_startup(
                broker  = self.broker,
                log_file= 'logs/paper_trades_stocks_only.json',
                service = 'alpaca_bot',
            )
            if discrepancies:
                logger.error(
                    f'Reconciliation found {len(discrepancies)} discrepancies — '
                    f'review logs/reconciliation.log before trading'
                )
                if os.getenv('ALPHAEDGE_FORCE_START') != '1':
                    print(
                        f'\n❌ HALTED: {len(discrepancies)} position discrepancies '
                        f'found on startup.\n'
                        f'   Review logs/reconciliation.log, resolve manually, '
                        f'then either fix the local state file or set\n'
                        f'   ALPHAEDGE_FORCE_START=1 to proceed anyway.\n'
                    )
                    return
                else:
                    logger.warning(
                        'ALPHAEDGE_FORCE_START=1 set — proceeding despite '
                        'unresolved reconciliation discrepancies'
                    )
        except Exception as e:
            logger.warning(f'Reconciliation skipped: {e}')

        # Load any existing Alpaca positions into managed_positions
        self._sync_positions()

        # Start price monitor for stop/target management
        self._monitor_running = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        print('\n  Position monitor started')

        print(f'\n✅ System ready. Running first scan now...\n')

        # Fix 2.3: Once-daily scan loop at 16:15 ET
        # For the very first run, scan immediately then schedule subsequent
        # runs at 16:15 ET. This gives instant feedback on startup.
        try:
            first_run = True
            while True:
                if first_run:
                    # Always run immediately on startup for operator feedback
                    self._run_scan()
                    self.heartbeat.ping()
                    first_run = False
                else:
                    # Schedule: wait until next 16:15 ET
                    wait_sec = _seconds_until_next_scan()
                    next_et  = datetime.now(MARKET_TZ)
                    from datetime import timedelta
                    next_et  = (next_et +
                        timedelta(seconds=wait_sec)).strftime('%Y-%m-%d %H:%M ET')
                    print(
                        f'\n  ⏰ Fix 2.3: Next scan at {next_et} '
                        f'({wait_sec/3600:.1f}h away)\n'
                    )
                    time.sleep(wait_sec)
                    self._run_scan()
                    self.heartbeat.ping()

        except KeyboardInterrupt:
            print('\n\nShutting down...')
            self._monitor_running = False

    # ── Scan ─────────────────────────────────────────────────────────

    def _run_scan(self):
        """
        Run main.py signal scanner and route BUY signals to Alpaca.
        Intercepts signals before they hit PaperTrader.
        """
        now = datetime.now(MARKET_TZ).strftime('%Y-%m-%d %H:%M ET')
        print(f'\n{"="*60}')
        print(f'SCAN  —  {now}  |  Mode: {self.mode}')
        print(f'{"="*60}')

        try:
            # Import signal generation components from existing main.py
            from data.stock_data        import StockDataFetcher
            from data.news_data         import NewsFetcher
            from data.feature_engine    import FeatureEngine
            from models.technical_model import TechnicalPredictor
            from models.sentiment_model import SentimentAnalyzer
            from models.regime_detector import RegimeDetector
            from models.sector_rotation import SectorRotation
            from models.meta_labeler    import MetaLabeler
            from market_regime          import MarketRegimeDetector            
            from multi_timeframe        import MultiTimeframeAnalyzer
            from correlation_filter     import CorrelationFilter
            from veto_agent             import VetoAgent
            from main import (
                get_full_watchlist,
                compute_signal,
                calc_atr,
                get_earnings_calendar,
            )
            from scanner import (
                check_volume_confirmation,
                check_risk_reward,
            )
            from model_cache import load_models, save_models
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            import pandas as pd

        except ImportError as e:
            logger.error(f'Import error: {e}')
            return

        # ── Circuit breaker ────────────────────────────────────────────
        # FIX: Alpaca had zero circuit breaker wiring. Fail-closed:
        # any exception in check() blocks the scan, not allows it.
        try:
            account      = self.broker.get_account()
            portfolio_v  = float(account.get('portfolio_value', 10000)) if account else 10000.0
            # Persist real starting capital once (never derive from current value)
            if not self.circuit_breaker.state.get('starting_capital'):
                self.circuit_breaker.state['starting_capital'] = portfolio_v
                self.circuit_breaker._save_state()
            starting_capital = self.circuit_breaker.state['starting_capital']
            if self.circuit_breaker.check(
                current_value    = portfolio_v,
                starting_capital = starting_capital,
                telegram         = self.telegram,
            ):
                print('  🚫 Circuit breaker active — scan aborted, no new entries')
                return
        except Exception as e:
            logger.error(f'Circuit breaker check FAILED — aborting scan as precaution: {e}')
            if self.telegram:
                try:
                    self.telegram.send_message(
                        f'⚠️ Alpaca circuit breaker check errored — scan blocked, investigate: {e}'
                    )
                except Exception as e:
                    logger.warning(f'Telegram circuit-breaker alert failed: {e}')
            return

        # ── Market regime ─────────────────────────────────────────────
        try:
            regime_detector = MarketRegimeDetector()
            market_regime   = regime_detector.analyze() if hasattr(regime_detector, 'analyze') else {'can_trade': True, 'regime': 'unknown', 'reason': ''}
        except Exception as e:
            logger.warning(f'Regime detect error: {e}')
            market_regime = {'can_trade': True, 'regime': 'unknown', 'reason': ''}

        if not market_regime.get('can_trade', True):
            print(f'  CASH MODE — {market_regime.get("reason")}')
            return

        # ── Check position limit ──────────────────────────────────
        current_positions = self.broker.get_positions()
        n_open = len(current_positions)
        if n_open >= MAX_POSITIONS:
            print(f'  Max positions reached ({n_open}/{MAX_POSITIONS}) — no new entries')
            return

        # ── Watchlist + earnings ──────────────────────────────────
        watchlist        = get_full_watchlist()
        earnings_soon    = get_earnings_calendar(watchlist)
        # Blackout window: block new BUYs only within 3 days of earnings.
        # earnings_soon itself stays a 7-day window for informational
        # display — only the actual blocking list is tightened.
        earnings_symbols = [e['symbol'] for e in earnings_soon if e.get('days_until', 99) <= 3]

        sector_analyzer = SectorRotation()
        sector_scores   = sector_analyzer.analyze()
        mtf_analyzer    = MultiTimeframeAnalyzer()
        corr_filter     = CorrelationFilter(max_per_sector=2)
        veto_agent      = VetoAgent()

        # Layer 10: Options Flow Intelligence (free, non-blocking)
        try:
            from options_analyzer import OptionsAnalyzer
            options_analyzer = OptionsAnalyzer(cache_minutes=30)
        except Exception as e:
            logger.warning(f'OptionsAnalyzer import failed — Layer 10 disabled: {e}')
            options_analyzer = None

        # ── Fetch + score stocks ──────────────────────────────────
        stock_fetcher = StockDataFetcher(watchlist=watchlist, lookback_days=730)
        stock_data    = stock_fetcher.fetch_all()
        engine        = FeatureEngine()
        reg_detector  = RegimeDetector()
        stock_signals = {}

        print(f'\n  Scoring {len(stock_data)} stocks...')

        for symbol, raw_df in stock_data.items():
            try:
                df           = engine.add_all_features(raw_df)
                feature_names = engine.get_feature_names()
                df           = reg_detector.detect(df)

                if len(df) < 100:
                    continue

                split      = len(df) - 30
                train      = df.iloc[max(0, split-180):split]
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
                selected = [f for f, m in zip(feature_names, selector.get_support()) if m]

                cached = load_models(symbol)
                if cached:
                    selected = cached.get('selected_features', selected)
                    model    = TechnicalPredictor(use_lstm=False)
                    model.models = {k: cached.get(k) for k in ['xgboost','lightgbm','random_forest','catboost']}
                    model.feature_names = selected
                    model.trained = True
                else:
                    model = TechnicalPredictor(use_lstm=False)
                    model.train(X_train[selected], y_train)
                    # C2 FIX: block overfit models from generating signals
                    if getattr(model, 'overfit_flagged', False):
                        gap = getattr(model, 'overfit_gap', 0)
                        logger.warning(
                            f'{symbol}: SKIP — overfit gap {gap:.2f} > 0.20 '
                            f'(train/val AUC delta too large, signals unreliable)'
                        )
                        continue
                    try:
                        save_models(symbol, {
                            'xgboost'          : model.models.get('xgboost'),
                            'lightgbm'         : model.models.get('lightgbm'),
                            'random_forest'    : model.models.get('random_forest'),
                            'catboost'         : model.models.get('catboost'),
                            'selected_features': selected,
                        })
                    except Exception as e:
                        logger.warning(f'Model cache save failed for {symbol}: {e}')

                latest = df.iloc[-1:]
                _preds = model.predict(latest[selected])
                pred   = _preds[0] if len(_preds) > 0 else 0.5  # guard: empty array → neutral

                # C4 FIX: MetaLabeler gate — filters false positives from primary model.
                # load() returns unfitted instance (pass-through) if no cached model.
                try:
                    _meta_path = os.path.join('model_cache', f'meta_{symbol}.pkl')
                    _meta      = MetaLabeler.load(_meta_path)
                    _approved, _meta_conf = _meta.should_trade(latest[selected], pred)
                    if not _approved:
                        logger.info(
                            f'{symbol}: MetaLabeler filtered '
                            f'(primary={pred:.3f}, meta_conf={_meta_conf:.3f})'
                        )
                        # Record in signals so dashboard shows META_FILTERED
                        stock_signals[symbol] = {
                            'prediction': pred, 'regime': df.iloc[-1]['regime'],
                            'price': df.iloc[-1]['close'],
                            'sector': sector_analyzer.get_sector_for_stock(symbol),
                            'sector_multiplier': sector_analyzer.get_sector_signal(symbol),
                            'signal': 'META_FILTERED', 'combined': 0.0,
                            'meta_conf': _meta_conf,
                        }
                        continue
                except Exception as _me:
                    logger.debug(f'{symbol}: MetaLabeler skipped ({_me})')

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
                logger.warning(f'{symbol}: {e}')

        # ── Sentiment ─────────────────────────────────────────────
        top_symbols = [s for s, _ in sorted(
            stock_signals.items(),
            key=lambda x: x[1]['prediction'], reverse=True
        )[:7]]
        sentiments = {}
        try:
            news_fetcher       = NewsFetcher()
            sentiment_analyzer = SentimentAnalyzer()
            all_news   = news_fetcher.fetch_all(top_symbols)
            sentiments = sentiment_analyzer.get_sentiment_for_stocks(all_news)
        except Exception as e:
            logger.warning(f'Sentiment error: {e}')

        # ── Signal evaluation + Alpaca execution ──────────────────
        print(f'\n  Evaluating signals...\n')
        buy_count = 0
        trade_stats = get_trade_stats_for_sizing()

        for symbol, data in sorted(
            stock_signals.items(),
            key=lambda x: x[1]['prediction'], reverse=True
        ):
            if n_open + buy_count >= MAX_POSITIONS:
                break

            pred       = data['prediction']
            regime     = data['regime']
            price      = data['price']
            sector     = data['sector']
            sect_mult  = data['sector_multiplier']
            sent_score = sentiments.get(symbol, {}).get('sentiment_score', 0.0)

            # S2 FIX: skip full scoring for symbols blocked by market-wide regime.
            # Aligns dashboard signal output with execution reality.
            if not market_regime.get('can_trade', True):
                stock_signals[symbol].update({'signal': 'MARKET_HOLD', 'combined': 0.0})
                continue

            # MTF
            try:
                # S5 FIX: reuse already-downloaded stock_data (eliminates 41 redundant
                # yfinance HTTP calls that were adding ~60s per scan).
                mtf_comp = mtf_analyzer.get_mtf_score(
                    symbol, daily_df=stock_data.get(symbol)
                )
            except Exception as e:
                logger.warning(f'{symbol}: MTF error, using neutral: {e}')
                mtf_comp = 0.0

            signal, combined = compute_signal(
                pred, regime, sent_score, sect_mult,
                symbol, earnings_symbols,
            )

            # Always record the final signal verdict back into the dict
            # so it gets saved to latest_signals.json at end of scan.
            stock_signals[symbol].update({
                'signal'   : signal,
                'combined' : combined,
                'sentiment': sent_score,
            })

            if signal != 'BUY':
                continue

            # MTF filter — mirrors main.py's working pattern; compute_signal()
            # never accepted an mtf_composite kwarg (that was a leftover from
            # an incomplete refactor), so MTF is applied as a separate
            # post-signal block here instead, same threshold as main.py.
            if mtf_comp < 0.5:
                print(f'  {symbol}: SKIP — MTF filter ({mtf_comp:.0%})')
                continue

            # Skip if already in position
            if symbol in current_positions or symbol in self.managed_positions:
                continue

            # Volume confirmation
            vol_ok, vol_ratio = check_volume_confirmation(stock_data, symbol)
            if not vol_ok:
                print(f'  {symbol}: SKIP — volume {vol_ratio:.1f}x')
                continue

            # Correlation filter
            if not corr_filter.can_add_position(symbol, current_positions):
                print(f'  {symbol}: SKIP — correlation filter')
                continue

            # Veto agent
            try:
                veto = veto_agent.review_signal(
                    symbol=symbol, price=price, prediction=pred,
                    regime=regime, sentiment=sent_score, sector=sector,
                    market_regime=market_regime.get('regime', 'unknown'),
                    mtf_score=mtf_comp,
                    current_positions=current_positions,
                    vix=market_regime.get('vix', 20),
                )
                if veto.get('decision') == 'VETO':
                    print(f'  {symbol}: VETOED — {veto.get("reason")}')
                    continue
            except Exception as e:
                # Fail-closed: veto error = skip trade
                logger.warning(f'{symbol}: veto agent error (fail-closed): {e}')
                continue

            # ATR + R:R check
            atr = calc_atr(stock_data, symbol)
            rr_ok, rr_ratio = check_risk_reward(price, atr)
            if not rr_ok:
                print(f'  {symbol}: SKIP — R:R {rr_ratio:.1f}')
                continue

            # ── Layer 10: Options Flow ─────────────────────────────
            options_score = 0.0
            if options_analyzer is not None:
                try:
                    options_score = options_analyzer.get_options_score(symbol)
                    if options_score < -0.20:
                        print(
                            f'  {symbol}: SKIP — options bearish '
                            f'(score={options_score:+.2f})'
                        )
                        continue
                    if options_score != 0.0:
                        # Adjust combined score (informational, capped at 1.0)
                        combined = min(1.0, combined + options_score * 0.5)
                        print(
                            f'  {symbol}: options score {options_score:+.2f} '
                            f'→ adjusted combined={combined:.2f}'
                        )
                except Exception as e:
                    logger.warning(f'{symbol}: options layer error (non-blocking): {e}')

            # ── ALL PASSED — execute on Alpaca ────────────────────
            account     = self.broker.get_account()
            portfolio   = account.get('portfolio_value', 10000) if account else 10000
            stop_price  = price - (atr * 1.0) if atr else price * 0.97
            target_price = price + (atr * 2.5) if atr else price * 1.06

            sizer = PositionSizer(portfolio_value=portfolio, base_risk_pct=RISK_PER_TRADE_PCT)
            dollar_amount = sizer.calculate(
                symbol         = symbol,
                price          = price,
                atr            = atr if atr else price * 0.03,
                signal_score   = combined,
                win_rate       = trade_stats['win_rate'],
                avg_win        = trade_stats['avg_win'],
                avg_loss       = trade_stats['avg_loss'],
                regime_conf    = market_regime.get('confidence', 1.0),
                open_positions = current_positions,
                n_trades       = trade_stats['n_trades'],
            )

            if dollar_amount <= 0:
                print(f'  {symbol}: SKIP — position sizer returned $0 (below minimum size)')
                continue

            print(
                f'  ✅ {symbol} BUY | score={combined:.2f} | vol={vol_ratio:.1f}x'
                f' | R:R={rr_ratio:.1f} | ${dollar_amount:.0f}'
            )

            success = self.broker.buy(symbol, dollar_amount)
            if success:
                buy_count += 1
                self.managed_positions[symbol] = {
                    'entry_price' : price,
                    'stop'        : stop_price,
                    'target'      : target_price,
                    'dollar_value': dollar_amount,
                    'open_time'   : time.time(),
                    'score'       : combined,
                }
                self._save_managed_positions()  # BUG 4 FIX: persist to disk
                try:
                    self.telegram.alert_buy_signal(
                        symbol, price, combined,
                        f'{self.mode} stocks | R:R={rr_ratio:.1f}', sent_score
                    )
                except Exception as e:
                    logger.warning(f'Telegram buy alert failed for {symbol}: {e}')

        print(f'\n  Scan complete. New entries: {buy_count}')

        # ── Persist signals to disk (atomic write) ─────────────────
        # Stamps every stock with its final signal/combined/sentiment so
        # the dashboard SIGNALS tab always reflects the latest VPS scan.
        try:
            import tempfile
            os.makedirs('logs', exist_ok=True)
            scan_ts = datetime.now(MARKET_TZ).isoformat()
            signals_out = {
                sym: {
                    'prediction': d.get('prediction', 0),
                    'regime'    : d.get('regime', ''),
                    'price'     : d.get('price', 0),
                    'sentiment' : d.get('sentiment', 0.0),
                    'sector'    : d.get('sector', ''),
                    'signal'    : d.get('signal', 'HOLD'),
                    'combined'  : d.get('combined', 0),
                    'saved_at'  : scan_ts,
                }
                for sym, d in sorted(
                    stock_signals.items(),
                    key=lambda x: x[1].get('combined', 0), reverse=True
                )
            }
            tmp_fd, tmp_path = tempfile.mkstemp(dir='logs', suffix='.tmp')
            with os.fdopen(tmp_fd, 'w') as f:
                import json as _json
                _json.dump(signals_out, f, indent=2)
            os.replace(tmp_path, 'logs/latest_signals.json')
            logger.info(f'Saved {len(signals_out)} signals to logs/latest_signals.json')
        except Exception as e:
            logger.warning(f'Signal save failed (non-critical): {e}')

        # ── Persist sector scores (BUG 1 FIX) ────────────────────
        # sector_scores returned by sector_analyzer.analyze() above.
        # Dashboard SECTORS tab reads logs/sectors.json — save it now.
        try:
            import tempfile as _tf
            tmp_fd, tmp_path = _tf.mkstemp(dir='logs', suffix='.tmp')
            with os.fdopen(tmp_fd, 'w') as f:
                import json as _json
                _json.dump(sector_scores, f, indent=2)
            os.replace(tmp_path, 'logs/sectors.json')
            logger.info(f'Saved {len(sector_scores)} sectors to logs/sectors.json')
        except Exception as e:
            logger.warning(f'Sectors save failed (non-critical): {e}')

        # ── Persist earnings calendar (BUG 2 FIX) ─────────────────
        # earnings_soon fetched by get_earnings_calendar() above.
        # Dashboard EARNINGS tab reads logs/earnings.json — save it now.
        try:
            import tempfile as _tf
            tmp_fd, tmp_path = _tf.mkstemp(dir='logs', suffix='.tmp')
            with os.fdopen(tmp_fd, 'w') as f:
                import json as _json
                _json.dump(earnings_soon, f, indent=2)
            os.replace(tmp_path, 'logs/earnings.json')
            logger.info(f'Saved {len(earnings_soon)} earnings entries to logs/earnings.json')
        except Exception as e:
            logger.warning(f'Earnings save failed (non-critical): {e}')
