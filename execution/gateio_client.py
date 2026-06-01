# execution/gateio_client.py
"""
Gate.io API v4 Client — REST
Handles: auth, order placement, position management, account balance.

Setup:
  export GATEIO_API_KEY=your_key
  export GATEIO_API_SECRET=your_secret
  export GATEIO_TESTNET=true   # optional, defaults to live

Supports spot trading (default) and futures (USDT settled).
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
GATEIO_API_KEY    = os.getenv('GATEIO_API_KEY', '')
GATEIO_API_SECRET = os.getenv('GATEIO_API_SECRET', '')
USE_TESTNET       = os.getenv('GATEIO_TESTNET', 'false').lower() == 'true'

MAINNET_URL  = 'https://api.gateio.ws/api/v4'
TESTNET_URL  = 'https://fx-api-testnet.gateio.ws/api/v4'

MAX_RISK_PCT = 0.015   # 1.5% account risk per trade


class GateioClient:
    """Gate.io API v4 REST client."""

    def __init__(
        self,
        api_key:    str  = GATEIO_API_KEY,
        api_secret: str  = GATEIO_API_SECRET,
        testnet:    bool = USE_TESTNET,
    ):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.base_url   = TESTNET_URL if testnet else MAINNET_URL
        self.testnet    = testnet
        self.session    = requests.Session()
        self.connected  = self._check_connection()

        env    = 'TESTNET' if testnet else 'LIVE'
        status = '✅ connected' if self.connected else '❌ no API keys'
        logger.info(f"GateioClient [{env}] {status}")

    # ── Signing ──────────────────────────────────────────────────────

    def _sign(
        self,
        method:   str,
        path:     str,
        query:    str = '',
        body:     str = '',
    ) -> dict:
        """
        Gate.io v4 signature:
        sign = HMAC-SHA512(secret, method + \n + path + \n + query + \n + body_hash + \n + ts)
        """
        ts        = str(int(time.time()))
        body_hash = hashlib.sha512(body.encode()).hexdigest()
        msg       = f'{method}\n{path}\n{query}\n{body_hash}\n{ts}'
        sign      = hmac.new(
            self.api_secret.encode(), msg.encode(), hashlib.sha512
        ).hexdigest()
        return {
            'KEY'       : self.api_key,
            'Timestamp' : ts,
            'SIGN'      : sign,
            'Content-Type': 'application/json',
        }

    # ── HTTP ─────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        params = params or {}
        query  = '&'.join(f'{k}={v}' for k, v in sorted(params.items()))
        headers = self._sign('GET', path, query)
        try:
            r = self.session.get(
                f'{self.base_url}{path}',
                headers=headers,
                params=params,
                timeout=10,
            )
            if r.status_code not in (200, 201):
                logger.warning(f'GET {path}: {r.status_code} {r.text[:200]}')
                return None
            return r.json()
        except Exception as e:
            logger.error(f'GET {path} error: {e}')
            return None

    def _post(self, path: str, body: dict) -> Optional[dict]:
        import json as _json
        body_str = _json.dumps(body)
        headers  = self._sign('POST', path, '', body_str)
        try:
            r = self.session.post(
                f'{self.base_url}{path}',
                headers=headers,
                data=body_str,
                timeout=10,
            )
            if r.status_code not in (200, 201):
                logger.warning(f'POST {path}: {r.status_code} {r.text[:200]}')
                return None
            return r.json()
        except Exception as e:
            logger.error(f'POST {path} error: {e}')
            return None

    def _delete(self, path: str, params: dict = None) -> Optional[dict]:
        params = params or {}
        query  = '&'.join(f'{k}={v}' for k, v in sorted(params.items()))
        headers = self._sign('DELETE', path, query)
        try:
            r = self.session.delete(
                f'{self.base_url}{path}',
                headers=headers,
                params=params,
                timeout=10,
            )
            if r.status_code not in (200, 201):
                logger.warning(f'DELETE {path}: {r.status_code} {r.text[:200]}')
                return None
            return r.json()
        except Exception as e:
            logger.error(f'DELETE {path} error: {e}')
            return None

    # ── Connection ───────────────────────────────────────────────────

    def _check_connection(self) -> bool:
        if not self.api_key or not self.api_secret:
            return False
        try:
            return self.get_usdt_balance() >= 0
        except Exception:
            return False

    # ── Account ──────────────────────────────────────────────────────

    def get_spot_balances(self) -> Optional[list]:
        """Get all spot account balances."""
        return self._get('/spot/accounts')

    def get_usdt_balance(self) -> float:
        """Get available USDT balance."""
        balances = self.get_spot_balances()
        if not balances:
            return 0.0
        for b in balances:
            if b.get('currency') == 'USDT':
                return float(b.get('available', 0.0))
        return 0.0

    def get_summary(self):
        """Print account summary."""
        balance = self.get_usdt_balance()
        orders  = self.get_open_orders()
        print(f"\n{'='*50}\nGATE.IO ACCOUNT  |  USDT: ${balance:,.2f}")
        if orders:
            print(f"Open orders ({len(orders)}):")
            for o in orders:
                print(
                    f"  {o.get('side','').upper()} {o.get('currency_pair')} "
                    f"| amount={o.get('amount')} | price={o.get('price')}"
                )
        else:
            print("  No open orders")
        print('='*50)

    # ── Market data (public) ─────────────────────────────────────────

    def get_ticker(self, currency_pair: str) -> Optional[dict]:
        """Get ticker for a currency pair. e.g. 'BTC_USDT'"""
        try:
            r = self.session.get(
                f'{self.base_url}/spot/tickers',
                params={'currency_pair': currency_pair},
                timeout=5,
            )
            data = r.json()
            return data[0] if data else None
        except Exception as e:
            logger.warning(f'get_ticker({currency_pair}) failed: {e}')
            return None

    def get_last_price(self, currency_pair: str) -> Optional[float]:
        """Get last traded price."""
        ticker = self.get_ticker(currency_pair)
        if ticker is None:
            return None
        try:
            return float(ticker['last'])
        except (KeyError, TypeError, ValueError):
            return None

    def get_klines(
        self,
        currency_pair: str,
        interval:      str = '15m',
        limit:         int = 200,
    ) -> Optional[list]:
        """
        Get candlestick data.
        interval: '10s','1m','5m','15m','30m','1h','4h','8h','1d','7d','30d'
        Returns list oldest-first: [timestamp, volume, close, high, low, open]
        """
        try:
            r = self.session.get(
                f'{self.base_url}/spot/candlesticks',
                params={
                    'currency_pair': currency_pair,
                    'interval'     : interval,
                    'limit'        : limit,
                },
                timeout=10,
            )
            data = r.json()
            return data if isinstance(data, list) else None
        except Exception as e:
            logger.warning(f'get_klines({currency_pair}) failed: {e}')
            return None

    # ── Orders ───────────────────────────────────────────────────────

    def place_order(
        self,
        currency_pair: str,
        side:          str,        # 'buy' or 'sell'
        amount:        float,      # quote amount for market buy, base for sell
        order_type:    str = 'market',
        price:         float = None,
        comment:       str = '',
    ) -> Optional[dict]:
        """
        Place a spot order on Gate.io.

        For market buy:  amount = USDT to spend
        For market sell: amount = base currency to sell (e.g. BTC)
        For limit:       amount = base currency, price required
        """
        if not self.connected:
            logger.warning('place_order: not connected')
            return None

        body = {
            'currency_pair': currency_pair,
            'side'         : side,
            'amount'       : str(amount),
            'type'         : order_type,
            'time_in_force': 'ioc' if order_type == 'market' else 'gtc',
        }
        if order_type == 'limit' and price:
            body['price'] = str(price)

        result = self._post('/spot/orders', body)
        if result:
            logger.info(
                f"✅ {side.upper()} {amount} {currency_pair} "
                f"| {order_type} | id={result.get('id')} | {comment}"
            )
        return result

    def cancel_order(self, currency_pair: str, order_id: str) -> bool:
        """Cancel an open order."""
        result = self._delete(
            f'/spot/orders/{order_id}',
            {'currency_pair': currency_pair},
        )
        return result is not None

    def get_open_orders(self, currency_pair: str = None) -> list:
        """Get all open orders, optionally filtered by pair."""
        params = {'status': 'open'}
        if currency_pair:
            params['currency_pair'] = currency_pair
        result = self._get('/spot/orders', params)
        return result if isinstance(result, list) else []

    # ── Position tracking ─────────────────────────────────────────────
    # Gate.io spot has no "positions" like futures.
    # We track holdings by checking spot balances.

    def get_holding(self, base_currency: str) -> float:
        """
        Get current holding of a base currency (e.g. 'BTC').
        Returns available amount as float.
        """
        balances = self.get_spot_balances()
        if not balances:
            return 0.0
        for b in balances:
            if b.get('currency') == base_currency:
                return float(b.get('available', 0.0))
        return 0.0

    def close_position(
        self,
        currency_pair: str,
        base_currency: str,
    ) -> Optional[dict]:
        """
        Sell entire holding of base_currency at market.
        e.g. close_position('BTC_USDT', 'BTC')
        """
        amount = self.get_holding(base_currency)
        if amount <= 0:
            logger.info(f'close_position({currency_pair}): no holding')
            return None
        return self.place_order(
            currency_pair = currency_pair,
            side          = 'sell',
            amount        = amount,
            comment       = 'close_position',
        )

    # ── Position sizing ──────────────────────────────────────────────

    def calculate_position_size(
        self,
        currency_pair: str,
        entry_price:   float,
        stop_price:    float,
        risk_pct:      float = MAX_RISK_PCT,
    ) -> float:
        """
        Fixed-risk position sizing.
        Returns USDT amount to spend (for market buy).
        qty_base = (balance * risk_pct) / abs(entry - stop)
        usdt_amount = qty_base * entry_price
        """
        balance       = self.get_usdt_balance()
        stop_distance = abs(entry_price - stop_price)
        if balance <= 0 or stop_distance <= 0 or entry_price <= 0:
            return 0.0
        qty_base    = (balance * risk_pct) / stop_distance
        usdt_amount = round(qty_base * entry_price, 2)
        logger.info(
            f'Size {currency_pair}: balance=${balance:.0f} | '
            f'risk=${balance*risk_pct:.0f} | stop_dist=${stop_distance:.2f} | '
            f'qty={qty_base:.6f} | usdt=${usdt_amount:.2f}'
        )
        return usdt_amount
