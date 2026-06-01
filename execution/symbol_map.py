# execution/symbol_map.py
"""
Crypto symbol format converter — supports Bybit, Gate.io, yfinance.

Formats:
  Bybit:    'BTCUSDT'   (no separator)
  Gate.io:  'BTC_USDT'  (underscore)
  yfinance: 'BTC/USD'   (slash)
  Display:  'BTC'       (base only)

Usage:
  from execution.symbol_map import SymbolMap

  SymbolMap.to_bybit('BTC/USD')      → 'BTCUSDT'
  SymbolMap.to_gateio('BTCUSDT')     → 'BTC_USDT'
  SymbolMap.to_yfinance('BTC_USDT')  → 'BTC/USD'
  SymbolMap.to_base('BTC_USDT')      → 'BTC'
"""

from typing import Optional


# ── Master registry ───────────────────────────────────────────────────
# (base, quote, yfinance_format)
# Add new symbols here — everything else derives automatically.

_REGISTRY = [
    ('BTC',   'USDT', 'BTC/USD'),
    ('ETH',   'USDT', 'ETH/USD'),
    ('SOL',   'USDT', 'SOL/USD'),
    ('BNB',   'USDT', 'BNB/USD'),
    ('XRP',   'USDT', 'XRP/USD'),
    ('DOGE',  'USDT', 'DOGE/USD'),
    ('ADA',   'USDT', 'ADA/USD'),
    ('AVAX',  'USDT', 'AVAX/USD'),
    ('LINK',  'USDT', 'LINK/USD'),
    ('DOT',   'USDT', 'DOT/USD'),
    ('MATIC', 'USDT', 'MATIC/USD'),
    ('UNI',   'USDT', 'UNI/USD'),
    ('ATOM',  'USDT', 'ATOM/USD'),
    ('LTC',   'USDT', 'LTC/USD'),
    ('NEAR',  'USDT', 'NEAR/USD'),
    ('OP',    'USDT', 'OP/USD'),
    ('ARB',   'USDT', 'ARB/USD'),
    ('SUI',   'USDT', 'SUI/USD'),
    ('TON',   'USDT', 'TON/USD'),
    ('PEPE',  'USDT', 'PEPE/USD'),
]

# Build lookup tables
_BYBIT_TO_YF:    dict[str, str] = {}   # 'BTCUSDT'  → 'BTC/USD'
_YF_TO_BYBIT:    dict[str, str] = {}   # 'BTC/USD'  → 'BTCUSDT'
_GATEIO_TO_YF:   dict[str, str] = {}   # 'BTC_USDT' → 'BTC/USD'
_YF_TO_GATEIO:   dict[str, str] = {}   # 'BTC/USD'  → 'BTC_USDT'
_BYBIT_TO_GATEIO:dict[str, str] = {}   # 'BTCUSDT'  → 'BTC_USDT'
_GATEIO_TO_BYBIT:dict[str, str] = {}   # 'BTC_USDT' → 'BTCUSDT'
_TO_BASE:        dict[str, str] = {}   # any format → 'BTC'
_BASE_TO_BYBIT:  dict[str, str] = {}   # 'BTC'      → 'BTCUSDT'
_BASE_TO_GATEIO: dict[str, str] = {}   # 'BTC'      → 'BTC_USDT'

for _base, _quote, _yf in _REGISTRY:
    _bybit  = f'{_base}{_quote}'
    _gateio = f'{_base}_{_quote}'

    _BYBIT_TO_YF[_bybit]      = _yf
    _YF_TO_BYBIT[_yf]         = _bybit
    _GATEIO_TO_YF[_gateio]    = _yf
    _YF_TO_GATEIO[_yf]        = _gateio
    _BYBIT_TO_GATEIO[_bybit]  = _gateio
    _GATEIO_TO_BYBIT[_gateio] = _bybit
    _BASE_TO_BYBIT[_base]     = _bybit
    _BASE_TO_GATEIO[_base]    = _gateio

    # All formats → base
    _TO_BASE[_bybit]  = _base
    _TO_BASE[_gateio] = _base
    _TO_BASE[_yf]     = _base
    _TO_BASE[_base]   = _base

    # Slash-free yfinance variant e.g. 'BTCUSD'
    _yf_noslash = _yf.replace('/', '')
    _YF_TO_BYBIT[_yf_noslash]  = _bybit
    _YF_TO_GATEIO[_yf_noslash] = _gateio
    _TO_BASE[_yf_noslash]      = _base


class SymbolMap:
    """Static utility class for crypto symbol format conversion."""

    @staticmethod
    def to_bybit(symbol: str) -> Optional[str]:
        """Any format → Bybit format ('BTCUSDT')"""
        s = symbol.upper().strip()
        if s in _BYBIT_TO_YF:    return s
        if s in _YF_TO_BYBIT:    return _YF_TO_BYBIT[s]
        if s in _GATEIO_TO_BYBIT:return _GATEIO_TO_BYBIT[s]
        if s in _BASE_TO_BYBIT:  return _BASE_TO_BYBIT[s]
        return None

    @staticmethod
    def to_gateio(symbol: str) -> Optional[str]:
        """Any format → Gate.io format ('BTC_USDT')"""
        s = symbol.upper().strip()
        if s in _GATEIO_TO_YF:   return s
        if s in _BYBIT_TO_GATEIO:return _BYBIT_TO_GATEIO[s]
        if s in _YF_TO_GATEIO:   return _YF_TO_GATEIO[s]
        if s in _BASE_TO_GATEIO: return _BASE_TO_GATEIO[s]
        return None

    @staticmethod
    def to_yfinance(symbol: str) -> Optional[str]:
        """Any format → yfinance format ('BTC/USD')"""
        s = symbol.upper().strip()
        if s in _YF_TO_BYBIT:    return s
        if s in _BYBIT_TO_YF:    return _BYBIT_TO_YF[s]
        if s in _GATEIO_TO_YF:   return _GATEIO_TO_YF[s]
        bybit = _BASE_TO_BYBIT.get(s)
        if bybit:                 return _BYBIT_TO_YF[bybit]
        return None

    @staticmethod
    def to_base(symbol: str) -> Optional[str]:
        """Any format → base currency ('BTC')"""
        return _TO_BASE.get(symbol.upper().strip())

    @staticmethod
    def is_known(symbol: str) -> bool:
        return SymbolMap.to_base(symbol) is not None

    @staticmethod
    def bybit_list(symbols: list[str] = None) -> list[str]:
        """Convert list to Bybit format. Returns all known if no list given."""
        if symbols is None:
            return list(_BYBIT_TO_YF.keys())
        return [b for s in symbols if (b := SymbolMap.to_bybit(s))]

    @staticmethod
    def gateio_list(symbols: list[str] = None) -> list[str]:
        """Convert list to Gate.io format. Returns all known if no list given."""
        if symbols is None:
            return list(_GATEIO_TO_YF.keys())
        return [g for s in symbols if (g := SymbolMap.to_gateio(s))]

    @staticmethod
    def yfinance_list(symbols: list[str] = None) -> list[str]:
        """Convert list to yfinance format. Returns all known if no list given."""
        if symbols is None:
            return list(_YF_TO_BYBIT.keys())
        return [y for s in symbols if (y := SymbolMap.to_yfinance(s))]

    @staticmethod
    def normalize_signal_dict(signals: dict, target: str = 'bybit') -> dict:
        """
        Re-key a signals dict to target format.
        target: 'bybit', 'gateio', or 'yfinance'

        Example:
            normalize_signal_dict({'BTC/USD': {...}}, target='gateio')
            → {'BTC_USDT': {...}}
        """
        convert = {
            'bybit'   : SymbolMap.to_bybit,
            'gateio'  : SymbolMap.to_gateio,
            'yfinance': SymbolMap.to_yfinance,
        }.get(target, SymbolMap.to_bybit)

        return {
            converted: data
            for sym, data in signals.items()
            if (converted := convert(sym))
        }


# ── Convenience functions ─────────────────────────────────────────────

def to_bybit(symbol: str) -> Optional[str]:
    return SymbolMap.to_bybit(symbol)

def to_gateio(symbol: str) -> Optional[str]:
    return SymbolMap.to_gateio(symbol)

def to_yfinance(symbol: str) -> Optional[str]:
    return SymbolMap.to_yfinance(symbol)

def to_base(symbol: str) -> Optional[str]:
    return SymbolMap.to_base(symbol)
