# execution/symbol_map.py
"""
Crypto symbol format converter.

Problem: Different parts of the system use different symbol formats:
  - CryptoPredictor / yfinance:  'BTC/USD', 'ETH/USD', 'SOL/USD'
  - Bybit API:                   'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
  - Bybit WebSocket topics:      'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
  - Display / Telegram:          'BTC', 'ETH', 'SOL'

This module is the single source of truth for all conversions.
Import it wherever you need to translate between formats.

Usage:
    from execution.symbol_map import SymbolMap

    SymbolMap.to_bybit('BTC/USD')    → 'BTCUSDT'
    SymbolMap.to_yfinance('BTCUSDT') → 'BTC/USD'
    SymbolMap.to_base('BTCUSDT')     → 'BTC'
    SymbolMap.bybit_list()           → ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    SymbolMap.yfinance_list()        → ['BTC/USD', 'ETH/USD', 'SOL/USD']
"""

from typing import Optional


# ── Master symbol registry ────────────────────────────────────────────
# Add new symbols here. Everything else derives from this table.
# Format: (base, quote, yfinance_suffix)
#   base:             e.g. 'BTC'
#   quote:            e.g. 'USDT'  (Bybit always USDT for linear)
#   yfinance_format:  e.g. 'BTC/USD' (what CryptoPredictor uses)

_REGISTRY = [
    # base    quote    yfinance
    ('BTC',   'USDT',  'BTC/USD'),
    ('ETH',   'USDT',  'ETH/USD'),
    ('SOL',   'USDT',  'SOL/USD'),
    ('BNB',   'USDT',  'BNB/USD'),
    ('XRP',   'USDT',  'XRP/USD'),
    ('DOGE',  'USDT',  'DOGE/USD'),
    ('ADA',   'USDT',  'ADA/USD'),
    ('AVAX',  'USDT',  'AVAX/USD'),
    ('LINK',  'USDT',  'LINK/USD'),
    ('DOT',   'USDT',  'DOT/USD'),
    ('MATIC', 'USDT',  'MATIC/USD'),
    ('UNI',   'USDT',  'UNI/USD'),
    ('ATOM',  'USDT',  'ATOM/USD'),
    ('LTC',   'USDT',  'LTC/USD'),
    ('NEAR',  'USDT',  'NEAR/USD'),
    ('OP',    'USDT',  'OP/USD'),
    ('ARB',   'USDT',  'ARB/USD'),
    ('SUI',   'USDT',  'SUI/USD'),
    ('TON',   'USDT',  'TON/USD'),
    ('PEPE',  'USDT',  'PEPE/USD'),
]

# Build lookup tables from registry
_BYBIT_TO_YFINANCE: dict[str, str] = {}   # 'BTCUSDT' → 'BTC/USD'
_YFINANCE_TO_BYBIT: dict[str, str] = {}   # 'BTC/USD' → 'BTCUSDT'
_BYBIT_TO_BASE:     dict[str, str] = {}   # 'BTCUSDT' → 'BTC'
_BASE_TO_BYBIT:     dict[str, str] = {}   # 'BTC'     → 'BTCUSDT'

for _base, _quote, _yf in _REGISTRY:
    _bybit = f'{_base}{_quote}'
    _BYBIT_TO_YFINANCE[_bybit] = _yf
    _YFINANCE_TO_BYBIT[_yf]    = _bybit
    _BYBIT_TO_BASE[_bybit]     = _base
    _BASE_TO_BYBIT[_base]      = _bybit
    # Also handle slash-free yfinance variants like 'BTCUSD'
    _yf_noslash = _yf.replace('/', '')
    _YFINANCE_TO_BYBIT[_yf_noslash] = _bybit


class SymbolMap:
    """
    Static utility class for crypto symbol format conversion.
    All methods are pure functions — no state, no side effects.
    """

    @staticmethod
    def to_bybit(symbol: str) -> Optional[str]:
        """
        Convert any known format to Bybit format.
        'BTC/USD' → 'BTCUSDT'
        'BTC'     → 'BTCUSDT'
        'BTCUSDT' → 'BTCUSDT' (passthrough)

        Returns None if symbol is unknown.
        """
        s = symbol.upper().strip()

        # Already Bybit format
        if s in _BYBIT_TO_YFINANCE:
            return s

        # yfinance format (with or without slash)
        if s in _YFINANCE_TO_BYBIT:
            return _YFINANCE_TO_BYBIT[s]

        # Base currency only
        if s in _BASE_TO_BYBIT:
            return _BASE_TO_BYBIT[s]

        return None

    @staticmethod
    def to_yfinance(symbol: str) -> Optional[str]:
        """
        Convert any known format to yfinance / CryptoPredictor format.
        'BTCUSDT' → 'BTC/USD'
        'BTC'     → 'BTC/USD'
        'BTC/USD' → 'BTC/USD' (passthrough)

        Returns None if symbol is unknown.
        """
        s = symbol.upper().strip()

        # Already yfinance format
        if s in _YFINANCE_TO_BYBIT:
            return s

        # Bybit format
        if s in _BYBIT_TO_YFINANCE:
            return _BYBIT_TO_YFINANCE[s]

        # Base currency only
        bybit = _BASE_TO_BYBIT.get(s)
        if bybit:
            return _BYBIT_TO_YFINANCE[bybit]

        return None

    @staticmethod
    def to_base(symbol: str) -> Optional[str]:
        """
        Extract base currency from any format.
        'BTCUSDT' → 'BTC'
        'BTC/USD' → 'BTC'
        'BTC'     → 'BTC'
        """
        s = symbol.upper().strip()

        if s in _BYBIT_TO_BASE:
            return _BYBIT_TO_BASE[s]

        # Try converting to bybit first then extracting
        bybit = SymbolMap.to_bybit(s)
        if bybit and bybit in _BYBIT_TO_BASE:
            return _BYBIT_TO_BASE[bybit]

        return None

    @staticmethod
    def bybit_list(yfinance_list: list[str] = None) -> list[str]:
        """
        Convert a list of yfinance symbols to Bybit symbols.
        If no list given, returns all known Bybit symbols.

        Example:
            SymbolMap.bybit_list(['BTC/USD', 'ETH/USD']) → ['BTCUSDT', 'ETHUSDT']
            SymbolMap.bybit_list()                       → ['BTCUSDT', 'ETHUSDT', ...]
        """
        if yfinance_list is None:
            return list(_BYBIT_TO_YFINANCE.keys())
        result = []
        for sym in yfinance_list:
            bybit = SymbolMap.to_bybit(sym)
            if bybit:
                result.append(bybit)
        return result

    @staticmethod
    def yfinance_list(bybit_list: list[str] = None) -> list[str]:
        """
        Convert a list of Bybit symbols to yfinance symbols.
        If no list given, returns all known yfinance symbols.

        Example:
            SymbolMap.yfinance_list(['BTCUSDT', 'ETHUSDT']) → ['BTC/USD', 'ETH/USD']
        """
        if bybit_list is None:
            return list(_YFINANCE_TO_BYBIT.keys())
        result = []
        for sym in bybit_list:
            yf = SymbolMap.to_yfinance(sym)
            if yf:
                result.append(yf)
        return result

    @staticmethod
    def is_known(symbol: str) -> bool:
        """Return True if symbol is in the registry in any format."""
        return SymbolMap.to_bybit(symbol) is not None

    @staticmethod
    def normalize_signal_dict(signals: dict) -> dict:
        """
        Takes a dict keyed by any symbol format and returns a new dict
        keyed by Bybit format. Unknown symbols are dropped.

        Useful when CryptoPredictor returns {'BTC/USD': {...}} and
        bybit_live.py needs {'BTCUSDT': {...}}.

        Example:
            signals = {'BTC/USD': {'prediction': 0.7}, 'ETH/USD': {'prediction': 0.6}}
            SymbolMap.normalize_signal_dict(signals)
            → {'BTCUSDT': {'prediction': 0.7}, 'ETHUSDT': {'prediction': 0.6}}
        """
        result = {}
        for sym, data in signals.items():
            bybit = SymbolMap.to_bybit(sym)
            if bybit:
                result[bybit] = data
        return result


# ── Convenience aliases ───────────────────────────────────────────────
# Import these directly if you only need one direction

def to_bybit(symbol: str) -> Optional[str]:
    return SymbolMap.to_bybit(symbol)

def to_yfinance(symbol: str) -> Optional[str]:
    return SymbolMap.to_yfinance(symbol)

def to_base(symbol: str) -> Optional[str]:
    return SymbolMap.to_base(symbol)
