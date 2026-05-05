# correlation_filter.py
# ALPHAEDGE - Correlation & Sector Filter
# Prevents overexposure to same sector
# Maximum 2 stocks per sector in portfolio

import logging

logger = logging.getLogger(__name__)

# Sector mapping for all watchlist stocks
STOCK_SECTORS = {
    # Technology
    'AAPL' : 'Technology',
    'MSFT' : 'Technology',
    'GOOGL': 'Technology',
    'AMZN' : 'Technology',
    'NVDA' : 'Technology',
    'META' : 'Technology',
    'AMD'  : 'Technology',
    'CRM'  : 'Technology',
    'SNOW' : 'Technology',
    'NET'  : 'Technology',
    'DDOG' : 'Technology',
    'CRWD' : 'Technology',
    'PLTR' : 'Technology',
    'MARA' : 'Technology',

    # Communications
    'NFLX' : 'Communications',

    # Financials
    'JPM'  : 'Financials',
    'V'    : 'Financials',
    'GS'   : 'Financials',
    'BAC'  : 'Financials',
    'MS'   : 'Financials',
    'SOFI' : 'Financials',
    'HOOD' : 'Financials',

    # Healthcare
    'JNJ'  : 'Healthcare',
    'PFE'  : 'Healthcare',
    'UNH'  : 'Healthcare',
    'ABBV' : 'Healthcare',
    'LLY'  : 'Healthcare',
    'MRK'  : 'Healthcare',

    # Consumer
    'WMT'  : 'Consumer',
    'COST' : 'Consumer',
    'HD'   : 'Consumer',
    'MCD'  : 'Consumer',
    'TSLA' : 'Consumer',
    'RIVN' : 'Consumer',

    # Energy
    'XOM'  : 'Energy',
    'CVX'  : 'Energy',
    'OXY'  : 'Energy',

    # Market ETFs
    'SPY'  : 'ETF',
    'QQQ'  : 'ETF',
    'IWM'  : 'ETF',
    'DIA'  : 'ETF',
}

# Maximum positions per sector
MAX_PER_SECTOR = 2


class CorrelationFilter:
    """
    Prevents overexposure to single sector.
    Ensures portfolio diversification.
    """

    def __init__(self, max_per_sector=MAX_PER_SECTOR):
        self.max_per_sector = max_per_sector

    def get_sector(self, symbol):
        """Get sector for a symbol."""
        return STOCK_SECTORS.get(symbol, 'Unknown')

    def count_sector_positions(self, positions):
        """
        Count how many open positions per sector.
        Returns dict: {sector: count}
        """
        sector_counts = {}
        for symbol in positions.keys():
            sector = self.get_sector(symbol)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return sector_counts

    def can_add_position(self, symbol, current_positions):
        """
        Check if we can add a position in this symbol.
        Returns True if sector limit not reached.
        """
        sector = self.get_sector(symbol)

        if sector == 'Unknown':
            return True

        sector_counts = self.count_sector_positions(
            current_positions
        )
        current_count = sector_counts.get(sector, 0)

        if current_count >= self.max_per_sector:
            logger.info(
                f"Correlation filter: {symbol} blocked "
                f"({sector} already has "
                f"{current_count}/{self.max_per_sector} positions)"
            )
            return False

        return True

    def get_portfolio_diversification(self, positions):
        """
        Get diversification score for current portfolio.
        1.0 = perfectly diversified
        0.0 = all in one sector
        """
        if not positions:
            return 1.0

        sector_counts = self.count_sector_positions(positions)
        unique_sectors = len(sector_counts)
        total_positions = len(positions)

        if total_positions == 0:
            return 1.0

        return unique_sectors / total_positions

    def print_portfolio_sectors(self, positions):
        """Print sector breakdown of portfolio."""
        if not positions:
            print("   No open positions")
            return

        sector_counts = self.count_sector_positions(positions)
        diversification = self.get_portfolio_diversification(positions)

        print(f"\n   Portfolio Sector Breakdown:")
        for sector, count in sorted(sector_counts.items()):
            bar = "█" * count
            limit = "⚠️" if count >= self.max_per_sector else "✅"
            print(f"   {limit} {sector:<15}: {bar} ({count})")

        print(f"\n   Diversification Score: {diversification:.0%}")


if __name__ == '__main__':
    print("\nTesting Correlation Filter...")
    cf = CorrelationFilter()

    # Test portfolio
    test_positions = {
        'AAPL': {'shares': 5},
        'MSFT': {'shares': 3},
        'GOOGL': {'shares': 2},
        'JPM' : {'shares': 4},
        'JNJ' : {'shares': 6},
    }

    print("\nCurrent portfolio:")
    cf.print_portfolio_sectors(test_positions)

    print("\nCan we add NVDA (Technology)?")
    result = cf.can_add_position('NVDA', test_positions)
    print(f"Result: {'YES' if result else 'NO - Sector limit reached!'}")

    print("\nCan we add XOM (Energy)?")
    result = cf.can_add_position('XOM', test_positions)
    print(f"Result: {'YES' if result else 'NO - Sector limit reached!'}")