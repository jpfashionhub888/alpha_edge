# execution/crypto_broker.py (Coinbase-ready skeleton)

import ccxt
import logging
from config.settings import (
    CRYPTO_EXCHANGE, COINBASE_API_KEY, COINBASE_SECRET_KEY,
    COINBASE_PASSPHRASE, COINBASE_SANDBOX
)

logger = logging.getLogger(__name__)


class CoinbaseBroker:
    """
    Handles Coinbase execution (paper-first).
    - Paper mode: simulates orders, uses market data
    - Live mode: places real orders via ccxt
    """

    def __init__(self, mode: str = 'paper'):
        self.mode = mode
        self.exchange_name = CRYPTO_EXCHANGE

        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            params = {'enableRateLimit': True}
            if self.mode == 'live':
                params.update({
                    'apiKey': COINBASE_API_KEY,
                    'secret': COINBASE_SECRET_KEY,
                })
                if self.exchange_name == 'coinbasepro':
                    params.update({'password': COINBASE_PASSPHRASE})

            self.exchange = exchange_class(params)

            if COINBASE_SANDBOX and self.mode == 'paper':
                if hasattr(self.exchange, 'set_sandbox_mode'):
                    self.exchange.set_sandbox_mode(True)
                logger.info(f"Coinbase sandbox enabled (mode={self.mode})")

        except Exception as e:
            logger.error(f"Failed to initialize Coinbase broker: {e}")
            raise

    def get_balance(self) -> dict:
        """Get account balances (paper/live)."""
        if self.mode == 'paper':
            # Return a simulated balance for learning mode
            return {'USD': 1000.0, 'BTC': 0.0, 'ETH': 0.0, 'SOL': 0.0}
        try:
            balances = self.exchange.fetch_balance()
            return balances.get('total', {})
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}

    def get_ticker(self, symbol: str) -> dict:
        """Get latest price/volume."""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker {symbol}: {e}")
            return {}

    def place_order(self, symbol: str, side: str, amount: float,
                    order_type: str = 'market') -> dict:
        """
        Place an order.
        side = 'buy' or 'sell'
        amount = quantity (e.g., BTC amount)
        """
        if self.mode == 'paper':
            # Simulated fill using latest price
            ticker = self.get_ticker(symbol)
            price = ticker.get('last', 0)
            logger.info(f"PAPER ORDER: {side.upper()} {amount} {symbol} @ ${price}")
            return {
                'status': 'filled',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'cost': amount * price,
                'timestamp': datetime.now(),
                'id': f'paper_{int(time.time())}'
            }

        try:
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount)
            else:
                # For limit orders later (Phase 6+)
                order = self.exchange.create_limit_order(symbol, side, amount, price)
            logger.info(f"LIVE ORDER: {order}")
            return order
        except Exception as e:
            logger.error(f"Order error: {e}")
            return {'status': 'error', 'error': str(e)}


# Quick test (paper)
if __name__ == "__main__":
    broker = CoinbaseBroker(mode='paper')
    print("Balance:", broker.get_balance())
    print("BTC/USD ticker:", broker.get_ticker('BTC/USD'))
    print("Paper buy:", broker.place_order('BTC/USD', 'buy', 0.001))