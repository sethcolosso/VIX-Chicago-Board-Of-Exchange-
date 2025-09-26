Prototype: Forward-Looking Portfolio Optimizer using VIX and market inputs.
- Uses yfinance to fetch real prices and VIX (^VIX).
- Builds adjusted covariance matrix with VIX-driven implied volatility.
- Solves a mean-variance optimization with cvxpy.
- Produces volatility-targeted weights and simple VIX-hedge notional suggestion.

- VIX is an annualized percentage (e.g., 15 => 15% annualized vol).
- Replace ASSETS with your target tickers (ETF or single-stock )
- For production, swap yfinance with Bloomberg/Polygon data connectors and add robust error handling.
