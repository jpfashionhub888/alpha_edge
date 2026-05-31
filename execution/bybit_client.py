# execution/bybit_client.py
"""
Bybit V5 API Client — REST
Handles: auth, order placement, position management, account balance.

Setup:
  export BYBIT_API_KEY=your_key
  export BYBIT_API_SECRET=your_secret
  export BYBIT_TESTNET=true   # optional, defaults to live

Supports linear (USDT perps) and spot categories.
"""

import os
import time
import hmac
import hashlib
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────
BYBIT_API_KEY    = os.getenv('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')
USE_TESTNET      = os.getenv('BYBIT_TESTNET', 'false').lower() == 'true'

MAINNET_URL      = 'https://api.bybit.com'
TESTNET_URL      = 'https://api-testnet.bybit.com'

DEFAULT_CATEGORY = 'linear'   # USDT perpetual futures
MAX_RISK_PCT     = 0.015      # 1.5% account risk per trade
RECV_WINDOW      = 5000       # ms


class BybitClient:
    """Bybit V5 REST client with order execution and position management."""

    def __init__(
        self,
        api_key:    str  = BYBIT_API_KEY,
        api_secret: str  = BYBIT_API_SECRET,
        category:   str  = DEFAULT_CATEGORY,
        testnet:    bool = USE_TESTNET,
    ):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.category   = category
        self.base_url   = TESTNET_URL if testnet else MAINNET_URL
        self.session    = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        self.connected  = self._check_connection()

        env = 'TESTNET' if testnet else 'LIVE'
        status = '✅ connected' if self.connected else '❌ no API keys'
        logger.info(f"BybitClient [{env}] {status}")

    # ── Signing ──────────────────────────────────────────────────────

    def _sign(self, payload: str) -> str:
        return hmac.new(
            self.api_secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

    def _headers(self, ts: str, sign: str) -> dict:
        return {
            'X-BAPI-API-KEY'    : self.api_key,
            'X-BAPI-SIGN'       : sign,
            'X-BAPI-TIMESTAMP'  : ts,
            'X-BAPI-RECV-WINDOW': str(RECV_WINDOW),
            'Content-Type'      : 'application/json',
        }

    # ── HTTP ─────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        params = params or {}
        ts     = str(int(time.time() * 1000))
        qs     = '&'.join(f'{k}={v}' for k, v in sorted(params.items()))
        sign   = self._sign(f'{ts}{self.api_key}{RECV_WINDOW}{qs}')
        try:
            r = self.session.get(
                f'{self.base_url}{endpoint}',
                headers=self._headers(ts, sign),
                params=params, timeout=10,
            )
            d = r.json()
            if d.get('retCode') != 0:
                logger.warning(f'GET {endpoint}: {d.get("retMsg")}')
                return None
            return d.get('result')
        except Exception as e:
            logger.error(f'GET {endpoint} error: {e}')
            return None

    def _post(self, endpoint: str, body: dict) -> Optional[dict]:
        import json as _json
        ts       = str(int(time.time() * 1000))
        body_str = _json.dumps(body)
        sign     = self._sign(f'{ts}{self.api_key}{RECV_WINDOW}{body_str}')
        try:
            r = self.session.post(
                f'{self.base_url}{endpoint}',
                headers=self._headers(ts, sign),
                data=body_str, timeout=10,
            )
            d = r.json()
            if d.get('retCode') != 0:
                logger.warning(f'POST {endpoint}: {d.get("retMsg")} | {body}')
                return None
            return d.get('result')
        except Exception as e:
            logger.error(f'POST {endpoint} error: {e}')
            return None

    # ── Connection ───────────────────────────────────────────────────

    def _check_connection(self) -> bool:
        if not self.api_key or not self.api_secret:
            return False
        try:
            return self.get_wallet_balance() is not None
        except Exception:
            return False

    # ── Account ──────────────────────────────────────────────────────

    def get_wallet_balance(self, account_type: str = 'UNIFIED') -> Optional[dict]:
        r = self._get('/v5/account/wallet-balance', {'accountType': account_type})
        if r is None:
            return None
        try:
            coins = r['list'][0]['coin']
            return {
                c['coin']: {
                    'available'     : float(c.get('availableToWithdraw', 0)),
                    'wallet_balance': float(c.get('walletBalance', 0)),
                    'unrealised_pnl': float(c.get('unrealisedPnl', 0)),
                }
                for c in coins
            }
        except (KeyError, IndexError, TypeError):
            return None

    def get_usdt_balance(self) -> float:
        b = self.get_wallet_balance()
        return float((b or {}).get('USDT', {}).get('available', 0.0))

    def get_summary(self):
        balance   = self.get_usdt_balance()
        positions = self.get_all_positions()
        print(f"\n{'='*50}\nBYBIT ACCOUNT  |  USDT: ${balance:,.2f}")
        if positions:
            print(f"Open positions ({len(positions)}):")
            for p in positions:
                emoji = '🟢' if float(p.get('unrealisedPnl', 0)) >= 0 else '🔴'
                print(
                    f"  {emoji} {p['symbol']} {p['side']}"
                    f" | size={p['size']} | entry=${p['avgPrice']}"
                    f" | PnL=${p['unrealisedPnl']} ({p.get('pnl_pct', 0):.1%})"
                )
        else:
            print("  No open positions")
        print('='*50)

    # ── Market data (public) ─────────────────────────────────────────

    def get_last_price(self, symbol: str) -> Optional[float]:
        try:
            r = self.session.get(
                f'{self.base_url}/v5/market/tickers',
                params={'category': self.category, 'symbol': symbol},
                timeout=5,
            )
            items = r.json()['result']['list']
            return float(items[0]['lastPrice']) if items else None
        except Exception:
            return None

    def get_klines(self, symbol: str, interval: str = '15', limit: int = 200) -> Optional[list]:
        """
        Get klines. interval: '1','5','15','30','60','240','D','W'
        Returns list newest-first: [ts, open, high, low, close, volume]
        """
        try:
            r = self.session.get(
                f'{self.base_url}/v5/market/kline',
                params={'category': self.category, 'symbol': symbol,
                        'interval': interval, 'limit': limit},
                timeout=10,
            )
            d = r.json()
            return d['result']['list'] if d.get('retCode') == 0 else None
        except Exception as e:
            logger.warning(f'get_klines({symbol}) failed: {e}')
            return None

    # ── Orders ───────────────────────────────────────────────────────

    def place_order(
        self,
        symbol:      str,
        side:        str,         # 'Buy' or 'Sell'
        qty:         float,
        order_type:  str   = 'Market',
        price:       float = None,
        stop_loss:   float = None,
        take_profit: float = None,
        reduce_only: bool  = False,
        comment:     str   = '',
    ) -> Optional[dict]:
        if not self.connected:
            logger.warning(f'place_order: not connected')
            return None

        body = {
            'category'   : self.category,
            'symbol'     : symbol,
            'side'       : side,
            'orderType'  : order_type,
            'qty'        : str(qty),
            'timeInForce': 'GTC',
        }
        if order_type == 'Limit' and price:
            body['price'] = str(price)
        if stop_loss:
            body['stopLoss'] = str(stop_loss)
        if take_profit:
            body['takeProfit'] = str(take_profit)
        if reduce_only:
            body['reduceOnly'] = True

        result = self._post('/v5/order/create', body)
        if result:
            logger.info(f"✅ {side} {qty} {symbol} | {order_type} | id={result.get('orderId')} | {comment}")
        return result

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        return self._post('/v5/order/cancel', {
            'category': self.category, 'symbol': symbol, 'orderId': order_id,
        }) is not None

    # ── Positions ────────────────────────────────────────────────────

    def get_position(self, symbol: str) -> Optional[dict]:
        r = self._get('/v5/position/list', {'category': self.category, 'symbol': symbol})
        if r is None:
            return None
        for pos in r.get('list', []):
            if float(pos.get('size', 0)) > 0:
                entry = float(pos.get('avgPrice', 0))
                curr  = float(pos.get('markPrice', entry))
                pos['pnl_pct'] = (curr - entry) / entry if entry > 0 else 0
                return pos
        return None

    def get_all_positions(self) -> list:
        r = self._get('/v5/position/list', {'category': self.category, 'settleCoin': 'USDT'})
        if r is None:
            return []
        result = []
        for pos in r.get('list', []):
            if float(pos.get('size', 0)) > 0:
                entry = float(pos.get('avgPrice', 0))
                curr  = float(pos.get('markPrice', entry))
                pos['pnl_pct'] = (curr - entry) / entry if entry > 0 else 0
                result.append(pos)
        return result

    def close_position(self, symbol: str) -> Optional[dict]:
        pos = self.get_position(symbol)
        if pos is None:
            return None
        size       = float(pos.get('size', 0))
        side       = pos.get('side', '')
        close_side = 'Sell' if side == 'Buy' else 'Buy'
        return self.place_order(
            symbol=symbol, side=close_side, qty=size,
            reduce_only=True, comment='close_position',
        )

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        return self._post('/v5/position/set-leverage', {
            'category': self.category, 'symbol': symbol,
            'buyLeverage': str(leverage), 'sellLeverage': str(leverage),
        }) is not None

    # ── Position sizing ──────────────────────────────────────────────

    def calculate_position_size(
        self,
        symbol:      str,
        entry_price: float,
        stop_price:  float,
        risk_pct:    float = MAX_RISK_PCT,
    ) -> float:
        """
        Fixed-risk position sizing.
        Risks risk_pct of USDT balance on the trade.
        qty = (balance * risk_pct) / abs(entry - stop)
        """
        balance       = self.get_usdt_balance()
        stop_distance = abs(entry_price - stop_price)
        if balance <= 0 or stop_distance <= 0:
            return 0.0
        qty = round((balance * risk_pct) / stop_distance, 3)
        logger.info(
            f'Size {symbol}: balance=${balance:.0f} | risk=${balance*risk_pct:.0f}'
            f' | stop_dist=${stop_distance:.2f} | qty={qty}'
        )
        return qty
