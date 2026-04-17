# live.py

"""
AlphaEdge Live System
Runs the scanner + dashboard together.
Scanner auto-updates every 30 minutes.
Dashboard auto-refreshes every 60 seconds.

Usage: python live.py
Then open http://localhost:8050
"""

import os
import sys
import json
import time
import logging
import warnings
import threading
from datetime import datetime

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)

# How often to re-scan markets (in seconds)
SCAN_INTERVAL = 1800  # 30 minutes


def run_scanner():
    """Run the market scanner once."""

    from data.stock_data import StockDataFetcher
    from data.feature_engine import FeatureEngine
    from models.technical_model import TechnicalPredictor
    from models.sentiment_model import SentimentAnalyzer
    from models.regime_detector import RegimeDetector
    from models.crypto_predictor import CryptoPredictor
    from data.news_data import NewsFetcher
    from execution.paper_trader import PaperTrader
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import mutual_info_classif

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print("\n" + "🚀" * 25)
    print(f"SCANNING MARKETS - {now}")
    print("🚀" * 25)

    # Initialize paper trader
    trader = PaperTrader(starting_capital=10000.0)
    trader.load_state()

    # ==========================================
    # STOCK SCANNING
    # ==========================================
    print("\n   Fetching stock data...")

    stock_watchlist = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'AMD', 'NFLX', 'SPY',
        'QQQ', 'JPM', 'V', 'JNJ', 'WMT'
    ]

    stock_fetcher = StockDataFetcher(
        watchlist=stock_watchlist,
        lookback_days=730
    )
    stock_data = stock_fetcher.fetch_all()

    print("\n   Generating signals...")
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

            model = TechnicalPredictor()
            model.train(X_train[selected], y_train)

            latest = df.iloc[-1:]
            pred = model.predict(latest[selected])[0]
            regime = latest['regime'].iloc[0]
            price = latest['close'].iloc[0]

            stock_signals[symbol] = {
                'prediction': pred,
                'regime': regime,
                'price': price,
            }

        except Exception as e:
            logger.warning(
                f"Error processing {symbol}: {e}"
            )

    # ==========================================
    # CRYPTO SCANNING
    # ==========================================
    print("\n   Scanning crypto...")

    crypto_watchlist = [
        'BTC/USD', 'ETH/USD', 'SOL/USD',
    ]

    crypto_signals = {}
    try:
        crypto_predictor = CryptoPredictor()
        crypto_signals = crypto_predictor.run_full_pipeline(
            crypto_watchlist,
            lookback_days=180
        )
    except Exception as e:
        logger.warning(f"Crypto error: {e}")

    # ==========================================
    # SENTIMENT
    # ==========================================
    print("\n   Analyzing sentiment...")

    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()

    top_stocks = sorted(
        stock_signals.items(),
        key=lambda x: x[1]['prediction'],
        reverse=True
    )[:5]

    top_symbols = [s[0] for s in top_stocks]
    sentiments = {}

    try:
        all_news = news_fetcher.fetch_all(top_symbols)
        sentiments = (
            sentiment_analyzer.get_sentiment_for_stocks(
                all_news
            )
        )
    except Exception as e:
        logger.warning(f"Sentiment error: {e}")

    # ==========================================
    # GENERATE SIGNALS AND TRADE
    # ==========================================
    print("\n   Generating final signals...")

    dashboard_signals = {}
    current_prices = {}

    for symbol, data in stock_signals.items():
        pred = data['prediction']
        regime = data['regime']
        price = data['price']
        current_prices[symbol] = price

        sent_score = 0.0
        if symbol in sentiments:
            sent_score = (
                sentiments[symbol]['sentiment_score']
            )

        combined = pred * 0.7 + (sent_score + 0.5) * 0.3

        signal = 'HOLD'
        if regime == 'uptrend' and combined > 0.55:
            signal = 'BUY'
        elif regime == 'uptrend' and pred > 0.52:
            signal = 'BUY'
        elif regime == 'downtrend':
            signal = 'AVOID'
        elif regime == 'volatile':
            signal = 'CAUTION'

        dashboard_signals[symbol] = {
            'prediction': float(pred),
            'regime': regime,
            'price': float(price),
            'sentiment': float(sent_score),
            'signal': signal,
        }

        if signal == 'BUY':
            emoji = "🟢"
        elif signal == 'AVOID':
            emoji = "🔴"
        else:
            emoji = "⚪"

        print(
            f"   {emoji} {symbol:6s}"
            f" | {signal:7s}"
            f" | Pred: {pred:.3f}"
            f" | {regime}"
        )

        if signal == 'BUY':
            trader.open_position(
                symbol, price, combined, reason=regime
            )

    for symbol, data in crypto_signals.items():
        pred = data['prediction']
        regime = data['regime']
        price = data['price']
        current_prices[symbol] = price

        signal = 'HOLD'
        if regime == 'uptrend' and pred > 0.52:
            signal = 'BUY'
        elif regime == 'downtrend':
            signal = 'AVOID'

        dashboard_signals[symbol] = {
            'prediction': float(pred),
            'regime': regime,
            'price': float(price),
            'sentiment': 0.0,
            'signal': signal,
        }

    # Update existing positions
    for symbol in list(trader.positions.keys()):
        if symbol in current_prices:
            trader.update_position(
                symbol,
                current_prices[symbol]
            )

    # Save everything
    os.makedirs('logs', exist_ok=True)

    with open('logs/latest_signals.json', 'w') as f:
        json.dump(dashboard_signals, f, indent=2)

    # Save scan timestamp
    scan_info = {
        'last_scan': datetime.now().isoformat(),
        'stocks_scanned': len(stock_signals),
        'crypto_scanned': len(crypto_signals),
        'open_positions': len(trader.positions),
        'next_scan_minutes': SCAN_INTERVAL // 60,
    }
    with open('logs/scan_info.json', 'w') as f:
        json.dump(scan_info, f, indent=2)

    trader.get_summary(current_prices)
    trader.save_state()

    print("\n   ✅ Scan complete. Dashboard updated.")
    return True


def scanner_loop():
    """Run scanner on a loop in background thread."""

    while True:
        try:
            run_scanner()
        except Exception as e:
            logger.error(f"Scanner error: {e}")

        next_scan = SCAN_INTERVAL // 60
        print(
            f"\n   ⏰ Next scan in {next_scan} minutes."
            f" Dashboard is live at http://localhost:8050"
        )
        time.sleep(SCAN_INTERVAL)


def start_dashboard():
    """Start the web dashboard."""

    from monitoring.dashboard import create_app

    app = create_app()
    app.run(
        debug=False,
        host='0.0.0.0',
        port=8050,
        use_reloader=False
    )


def main():
    print("\n" + "🌐" * 25)
    print("ALPHAEDGE LIVE SYSTEM")
    print("🌐" * 25)
    print("\nStarting scanner + dashboard...")
    print("Dashboard will be at: http://localhost:8050")
    print("Scanner runs every 30 minutes automatically")
    print("Press Ctrl+C to stop everything\n")

    # Run first scan immediately
    try:
        run_scanner()
    except Exception as e:
        logger.error(f"Initial scan failed: {e}")
        print("   ⚠️ Initial scan failed but dashboard starting anyway")

    # Start scanner in background thread
    scanner_thread = threading.Thread(
        target=scanner_loop,
        daemon=True
    )
    scanner_thread.start()

    # Start dashboard in main thread (blocks)
    start_dashboard()


if __name__ == "__main__":
    main()