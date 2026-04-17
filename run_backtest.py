# run_backtest.py

"""
Walk Forward Backtest V5
With proper risk management: stop loss, take profit,
trailing stop, daily loss limit.
"""

import logging
from data.stock_data import StockDataFetcher
from data.feature_engine import FeatureEngine
from backtest.walk_forward import WalkForwardBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

print("\n" + "🚀" * 25)
print("WALK FORWARD BACKTEST V5")
print("Risk Managed: Stop Loss + Take Profit + Trailing Stop")
print("🚀" * 25)

print("\n1. Fetching data (2 years, 15 stocks)")
fetcher = StockDataFetcher(
    watchlist=[
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'AMD', 'NFLX', 'SPY',
        'QQQ', 'JPM', 'V', 'JNJ', 'WMT'
    ],
    lookback_days=730
)

all_data = fetcher.get_combined_data()

print("\n2. Generating features")
engine = FeatureEngine()
df = engine.add_all_features(all_data)
feature_names = engine.get_feature_names()

print(f"   Total rows: {len(df)}")
print(f"   Total features: {len(feature_names)}")

print("\n3. Running Risk Managed Backtest")

backtester = WalkForwardBacktester(
    train_window_days=180,
    retrain_frequency_days=30,
    top_features=20,
    min_auc=0.54
)

performance = backtester.run(df, feature_names)

print("\n" + "🎉" * 20)
print("BACKTEST V5 COMPLETE")
print("🎉" * 20)

print("\nV5 risk management rules:")
print("   Stop Loss: -3% (cut losses fast)")
print("   Take Profit: +6% (2:1 reward/risk)")
print("   Trailing Stop: -2% from high")
print("   Daily Loss Limit: -2% (circuit breaker)")