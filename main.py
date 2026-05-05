# main.py

"""
AlphaEdge Main Trading Scanner V3
With sector rotation, LSTM, Telegram,
expanded watchlist, and earnings calendar.
"""

import warnings
import os

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import logging
import pandas as pd  # FIX: was missing, needed for ATR calc
from datetime import datetime

from data.stock_data import StockDataFetcher
from data.news_data import NewsFetcher
from data.feature_engine import FeatureEngine
from models.technical_model import TechnicalPredictor
from models.sentiment_model import SentimentAnalyzer
from models.regime_detector import RegimeDetector
from models.crypto_predictor import CryptoPredictor
from models.sector_rotation import SectorRotation
from execution.paper_trader import PaperTrader
from monitoring.telegram_bot import TelegramBot
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from model_cache import save_models, load_models, is_cache_valid

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)


def get_earnings_calendar(watchlist):
    """Check which stocks have earnings this week."""

    import yfinance as yf

    print("\n📅 Checking earnings calendar...")

    earnings_soon = []

    for symbol in watchlist:
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
        except Exception:
            pass

    if earnings_soon:
        n = len(earnings_soon)
        print(f"   ⚠️ {n} stocks reporting this week:")
        for e in earnings_soon:
            d = e['days_until']
            if d == 0:
                w = "TODAY!"
            elif d == 1:
                w = "TOMORROW!"
            else:
                w = f"in {d} days"
            print(f"      ⚠️ {e['symbol']} earnings {w}")
    else:
        print("   ✅ No earnings this week")

    return earnings_soon


def get_full_watchlist():
    """Return expanded stock watchlist."""

    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'AMD', 'NFLX',
        'SPY', 'QQQ', 'IWM', 'DIA',
        'JPM', 'V', 'GS', 'BAC', 'MS',
        'JNJ', 'PFE', 'UNH', 'ABBV', 'LLY', 'MRK',
        'WMT', 'COST', 'HD', 'MCD',
        'XOM', 'CVX', 'OXY',
        'SOFI', 'PLTR', 'RIVN', 'HOOD', 'MARA',
        'CRM', 'SNOW', 'NET', 'DDOG', 'CRWD',
    ]


# FIX: extracted duplicate signal logic into a single helper
def compute_signal(pred, regime, sent_score, sect_mult,
                   symbol, earnings_symbols):
    """
    Compute the final trading signal for a stock.
    Single source of truth used for printing and dashboard JSON.
    """
    combined = (
        pred * 0.6
        + (sent_score + 0.5) * 0.2
        + (sect_mult - 0.5) * 0.2
    )

    signal = 'HOLD'
    if regime == 'uptrend' and combined > 0.55:
        signal = 'BUY'
    elif regime == 'uptrend' and pred > 0.52:
        signal = 'BUY'
    elif regime == 'downtrend':
        signal = 'AVOID'
    elif regime == 'volatile':
        signal = 'CAUTION'

    if sect_mult > 1.1 and signal == 'HOLD':
        if pred > 0.55 and regime == 'sideways':
            signal = 'BUY'

    if sect_mult < 0.8 and signal == 'BUY':
        signal = 'HOLD'

    if symbol in earnings_symbols and signal == 'BUY':
        signal = 'EARNINGS_HOLD'

    return signal, combined


def calc_atr(stock_data, symbol):
    """Calculate 14-period ATR for a symbol. Returns None on failure."""
    try:
        if symbol not in stock_data:
            return None
        df_atr = stock_data[symbol].copy()
        high  = df_atr['high']
        low   = df_atr['low']
        close = df_atr['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low  - close.shift(1))
        true_range = pd.concat(
            [tr1, tr2, tr3], axis=1
        ).max(axis=1)
        return float(true_range.rolling(14).mean().iloc[-1])
    except Exception:
        return None


def run_daily_scan():
    """Run the complete daily trading scan."""

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print("\n" + "🚀" * 25)
    print(f"ALPHA EDGE V3 - {now}")
    print("4-Model Ensemble + Sector Rotation + LSTM")
    print("🚀" * 25)

    trader = PaperTrader(starting_capital=10000.0)
    trader.load_state()

    telegram = TelegramBot()

    stock_watchlist = get_full_watchlist()

    # ==========================================
    # PHASE 0: EARNINGS + SECTOR ROTATION
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 0: EARNINGS + SECTOR ROTATION")
    print("="*60)

    earnings_soon = get_earnings_calendar(stock_watchlist)
    earnings_symbols = [e['symbol'] for e in earnings_soon]

    sector_analyzer = SectorRotation()
    sector_scores = sector_analyzer.analyze()

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
    engine = FeatureEngine()
    regime_detector = RegimeDetector()

    stock_signals = {}

    for symbol, raw_df in stock_data.items():
        try:
            df = engine.add_all_features(raw_df)
            feature_names = engine.get_feature_names()
            df = regime_detector.detect(df)

            if len(df) < 200:
                continue

            split = len(df) - 30
            train = df.iloc[:split]

            X_train = train[feature_names]
            y_train = train['target']

            if len(y_train.unique()) < 2:
                continue

            min_class = y_train.value_counts().min()
            if min_class < 10:
                continue

            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(20, len(feature_names))
            )
            selector.fit(X_train, y_train)
            mask = selector.get_support()
            selected = [
                f for f, m in zip(feature_names, mask) if m
            ]

            # Check model cache first
            cached = load_models(symbol)

            if cached:
                # Use cached models (instant!)
                print(f"   {symbol}: Loading from cache...")
                selected = cached.get('selected_features', selected)
                model = TechnicalPredictor(use_lstm=False)
                model.models = {
                    'xgboost'      : cached.get('xgboost'),
                    'lightgbm'     : cached.get('lightgbm'),
                    'random_forest': cached.get('random_forest'),
                    'catboost'     : cached.get('catboost'),
                }

                model.feature_names = selected
                model.trained = True
            else:
                # Train fresh models
                print(f"   {symbol}: Training models...")
                model = TechnicalPredictor(use_lstm=True)
                model.train(X_train[selected], y_train)

                # Save to cache
                try:
                    save_models(symbol, {
                        'xgboost'          : model.models.get('xgboost'),
                        'lightgbm'         : model.models.get('lightgbm'),
                        'random_forest'    : model.models.get('random_forest'),
                        'catboost'         : model.models.get('catboost'),
                        'selected_features': selected,
                    })
                    print(f"   {symbol}: Models cached!")
                except Exception as e:
                    logger.warning(f"Cache save failed for {symbol}: {e}")

            latest = df.iloc[-1:]
            pred = model.predict(latest[selected])[0]
            regime = latest['regime'].iloc[0]
            price = latest['close'].iloc[0]

            sector_mult = sector_analyzer.get_sector_signal(symbol)
            sector = sector_analyzer.get_sector_for_stock(symbol)

            stock_signals[symbol] = {
                'prediction': pred,
                'regime': regime,
                'price': price,
                'sector': sector,
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
    crypto_signals = {}

    try:
        crypto_predictor = CryptoPredictor()
        crypto_signals = crypto_predictor.run_full_pipeline(
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

    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()

    top_stocks = sorted(
        stock_signals.items(),
        key=lambda x: x[1]['prediction'],
        reverse=True
    )[:7]

    top_symbols = [s[0] for s in top_stocks]
    sentiments = {}

    try:
        all_news = news_fetcher.fetch_all(top_symbols)
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
    print("-"*65)

    dashboard_signals = {}

    for symbol, data in sorted(
        stock_signals.items(),
        key=lambda x: x[1]['prediction'],
        reverse=True
    ):
        pred      = data['prediction']
        regime    = data['regime']
        price     = data['price']
        sector    = data['sector']
        sect_mult = data['sector_multiplier']

        sent_score = 0.0
        if symbol in sentiments:
            sent_score = sentiments[symbol]['sentiment_score']

        # FIX: single helper replaces both duplicated signal blocks
        signal, combined = compute_signal(
            pred, regime, sent_score, sect_mult,
            symbol, earnings_symbols
        )

        if signal == 'BUY':
            emoji = "🟢"
        elif signal == 'AVOID':
            emoji = "🔴"
        elif signal == 'EARNINGS_HOLD':
            emoji = "📅"
        elif signal == 'CAUTION':
            emoji = "⚠️"
        else:
            emoji = "⚪"

        sect_emoji = ""
        if sect_mult > 1.1:
            sect_emoji = "🔄↑"
        elif sect_mult < 0.9:
            sect_emoji = "🔄↓"

        sent_emoji = ""
        if sent_score > 0.1:
            sent_emoji = "📈"
        elif sent_score < -0.1:
            sent_emoji = "📉"

        print(
            f"   {emoji} {symbol:6s}"
            f" | {pred:.3f}"
            f" | {regime:10s}"
            f" | {sent_score:+.2f}{sent_emoji}"
            f" | {sector:13s}{sect_emoji}"
            f" | {signal}"
            f" | ${price:.2f}"
        )

        # FIX: BUY action is now correctly inside the for-loop
        if signal == 'BUY':
            atr = calc_atr(stock_data, symbol)
            opened = trader.open_position(
                symbol, price, combined,
                reason=regime, atr=atr
            )
            if opened:
                telegram.alert_buy_signal(
                    symbol, price, pred, regime, sent_score
                )

        # Build dashboard entry here — no second loop needed
        dashboard_signals[symbol] = {
            'prediction': float(pred),
            'regime': regime,
            'price': float(price),
            'sentiment': float(sent_score),
            'sector': sector,
            'signal': signal,
        }

    print("\n🪙 Crypto Signals:")
    print("-"*60)

    if len(crypto_signals) > 0:
        for symbol, data in crypto_signals.items():
            pred   = data['prediction']
            regime = data['regime']
            price  = data['price']

            signal = 'HOLD'
            if regime == 'uptrend' and pred > 0.52:
                signal = 'BUY'
            elif regime == 'downtrend':
                signal = 'AVOID'

            emoji = "🟢" if signal == 'BUY' else ("🔴" if signal == 'AVOID' else "⚪")

            print(
                f"   {emoji} {symbol:10s}"
                f" | {pred:.3f}"
                f" | {regime:10s}"
                f" | {signal}"
                f" | ${price:,.2f}"
            )

            if signal == 'BUY':
                opened = trader.open_position(
                    symbol, price, pred, reason=regime
                )
                if opened:
                    telegram.alert_buy_signal(
                        symbol, price, pred, regime, 0.0
                    )

            dashboard_signals[symbol] = {
                'prediction': float(pred),
                'regime': regime,
                'price': float(price),
                'sentiment': 0.0,
                'sector': 'Crypto',
                'signal': signal,
            }
    else:
        print("   No crypto signals generated")

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

    if len(trader.positions) > 0:
        print("\n   Checking stop loss / take profit...")
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
        print("\n   No open positions to manage")

    # ==========================================
    # SAVE FOR DASHBOARD
    # ==========================================
    os.makedirs('logs', exist_ok=True)

    with open('logs/latest_signals.json', 'w') as f:
        json.dump(dashboard_signals, f, indent=2)

    with open('logs/earnings.json', 'w') as f:
        json.dump(earnings_soon, f, indent=2)

    with open('logs/sectors.json', 'w') as f:
        sector_data = {}
        for sector, data in sector_scores.items():
            sector_data[sector] = {
                'score': float(data['score']),
                'flow': data['flow'],
                'momentum_21d': float(data['momentum_21d']),
            }
        json.dump(sector_data, f, indent=2)

    print("\n   💾 All data saved for dashboard")

    # ==========================================
    # PORTFOLIO SUMMARY
    # ==========================================
    trader.get_summary(current_prices)

    for symbol, pos in trader.positions.items():
        pos['current_price'] = current_prices.get(
            symbol, pos['entry_price']
        )

    trader.save_state()

    total_value = trader.capital + sum(
        pos.get('shares', 0) * current_prices.get(
            symbol, pos.get('entry_price', 0)
        )
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
        pnl_pct     = (
            (curr_price - entry_price) / entry_price
            if entry_price > 0 else 0
        )
        positions_with_pnl[symbol] = {
            **pos,
            'current_price': curr_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
        }

    telegram.alert_daily_summary(
        total_value, total_pnl, total_pct,
        positions_with_pnl, dashboard_signals
    )

    print("\n" + "✅" * 25)
    print("ALPHA EDGE V3 SCAN COMPLETE")
    print("✅" * 25)

    n_s = len(stock_signals)
    n_c = len(crypto_signals)
    n_e = len(earnings_soon)

    print(f"\nScanned: {n_s} stocks + {n_c} crypto")
    print(f"Earnings this week: {n_e} stocks")
    print(f"Models: 4 per stock (XGB+LGB+RF+LSTM)")
    print("\nTo view dashboard:")
    print("   python run_dashboard.py")
    print("   Open http://localhost:8050\n")


if __name__ == "__main__":
    run_daily_scan()
