# risk/manager.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Dynamic risk management.
    Stop loss and take profit adjust based on
    stock volatility instead of fixed percentages.
    """

    def __init__(self, base_stop_loss=0.03,
                 reward_risk_ratio=2.5,
                 trailing_stop_multiplier=0.8,
                 max_daily_loss=0.02,
                 max_portfolio_risk=0.06):
        self.base_stop_loss = base_stop_loss
        self.reward_risk_ratio = reward_risk_ratio
        self.trailing_multiplier = trailing_stop_multiplier
        self.max_daily_loss = max_daily_loss
        self.max_portfolio_risk = max_portfolio_risk

    def calculate_dynamic_stops(self, df):
        """
        Calculate stop loss based on ATR (volatility).
        Volatile stocks get wider stops.
        Calm stocks get tighter stops.
        """

        if 'atr_ratio' in df.columns:
            atr = df['atr_ratio'].iloc[-1]
        else:
            atr = self.base_stop_loss

        # Dynamic stop loss = 1.5x ATR
        stop_loss = max(atr * 1.5, 0.02)
        stop_loss = min(stop_loss, 0.08)

        # Take profit = stop loss * reward/risk ratio
        take_profit = stop_loss * self.reward_risk_ratio

        # Trailing stop = tighter than stop loss
        trailing = stop_loss * self.trailing_multiplier

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': trailing,
        }

    def apply(self, df):
        """Apply dynamic risk management."""

        df = df.copy()

        df['managed_signal'] = 0.0
        df['managed_return'] = 0.0
        df['exit_reason'] = 'none'

        # Calculate dynamic stops
        stops = self.calculate_dynamic_stops(df)
        stop_loss = stops['stop_loss']
        take_profit = stops['take_profit']
        trailing_stop = stops['trailing_stop']

        in_trade = False
        entry_price = 0.0
        highest = 0.0
        daily_loss = 0.0
        current_date = None
        trade_signal = 0.0

        for i in range(len(df)):
            idx = df.index[i]
            row = df.iloc[i]

            price = row['close']
            raw_signal = row['signal']
            daily_ret = row['returns']

            td = idx.date() if hasattr(idx, 'date') else idx
            if td != current_date:
                daily_loss = 0.0
                current_date = td

            if daily_loss < -self.max_daily_loss:
                df.iloc[i, df.columns.get_loc('managed_signal')] = 0.0
                df.iloc[i, df.columns.get_loc('exit_reason')] = 'daily_limit'
                in_trade = False
                continue

            if in_trade:
                if price > highest:
                    highest = price

                pnl = (price - entry_price) / entry_price

                if pnl <= -stop_loss:
                    ret = -stop_loss * trade_signal
                    df.iloc[i, df.columns.get_loc('managed_signal')] = 0.0
                    df.iloc[i, df.columns.get_loc('managed_return')] = ret
                    df.iloc[i, df.columns.get_loc('exit_reason')] = 'stop_loss'
                    daily_loss += ret
                    in_trade = False
                    continue

                if pnl >= take_profit:
                    ret = take_profit * trade_signal
                    df.iloc[i, df.columns.get_loc('managed_signal')] = 0.0
                    df.iloc[i, df.columns.get_loc('managed_return')] = ret
                    df.iloc[i, df.columns.get_loc('exit_reason')] = 'take_profit'
                    in_trade = False
                    continue

                drop = (highest - price) / highest
                if drop >= trailing_stop:
                    actual = pnl * trade_signal
                    df.iloc[i, df.columns.get_loc('managed_signal')] = 0.0
                    df.iloc[i, df.columns.get_loc('managed_return')] = actual
                    df.iloc[i, df.columns.get_loc('exit_reason')] = 'trailing_stop'
                    daily_loss += actual
                    in_trade = False
                    continue

                ret = trade_signal * daily_ret
                df.iloc[i, df.columns.get_loc('managed_signal')] = trade_signal
                df.iloc[i, df.columns.get_loc('managed_return')] = ret
                daily_loss += ret

            else:
                if raw_signal > 0:
                    in_trade = True
                    entry_price = price
                    highest = price
                    trade_signal = raw_signal
                    ret = raw_signal * daily_ret
                    df.iloc[i, df.columns.get_loc('managed_signal')] = raw_signal
                    df.iloc[i, df.columns.get_loc('managed_return')] = ret
                    df.iloc[i, df.columns.get_loc('exit_reason')] = 'entry'
                    daily_loss += ret

        # Stats
        exits = df[df['exit_reason'] != 'none']
        ecounts = exits['exit_reason'].value_counts()

        print("\n   Risk Management Stats:")
        print(
            f"      Dynamic Stop Loss: {stop_loss:.1%}"
        )
        print(
            f"      Dynamic Take Profit: {take_profit:.1%}"
        )
        print(
            f"      Dynamic Trailing: {trailing_stop:.1%}"
        )

        for reason, count in ecounts.items():
            print(f"      {reason}: {count}")

        sl = exits[exits['exit_reason'] == 'stop_loss']
        tp = exits[exits['exit_reason'] == 'take_profit']
        ts = exits[exits['exit_reason'] == 'trailing_stop']

        if len(sl) > 0:
            avg = sl['managed_return'].mean()
            print(f"      Avg stop loss: {avg:.2%}")

        if len(tp) > 0:
            avg = tp['managed_return'].mean()
            print(f"      Avg take profit: {avg:.2%}")

        if len(ts) > 0:
            avg = ts['managed_return'].mean()
            print(f"      Avg trailing stop: {avg:.2%}")

        return df