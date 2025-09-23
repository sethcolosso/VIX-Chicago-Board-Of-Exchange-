"""
vix_opt.py

Prototype: Forward-Looking Portfolio Optimizer using VIX and market inputs.
- Uses yfinance to fetch real prices and VIX (^VIX).
- Builds adjusted covariance matrix with VIX-driven implied volatility.
- Solves a mean-variance optimization with cvxpy.
- Produces volatility-targeted weights and simple VIX-hedge notional suggestion.

Notes:
- VIX is an annualized percentage (e.g., 15 => 15% annualized vol).
- Replace ASSETS with your target tickers (ETF or single-stock )
- For production, swap yfinance with Bloomberg/Polygon data connectors and add robust error handling.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# User parameters / universe
# ----------------------------
ASSETS = ["SPY", "QQQ", "TLT", "IEF", "GLD", "DBC"]  # example tradables; change for your product
START_DATE = (datetime.utcnow() - timedelta(days=5*365)).strftime("%Y-%m-%d")  # 5 years history
END_DATE = datetime.utcnow().strftime("%Y-%m-%d")
RISK_FREE_RATE = 0.03  # annual risk-free (3%) - replace with current T-Bill yield for more accuracy

# Optimizer / model params
ALPHA = 0.6          # weight historic vs implied vol: sigma_adj = alpha*sigma_hist + (1-alpha)*sigma_impl
VIX_INFL_CORR_GAMMA = 0.7   # correlation inflation factor when VIX above threshold
VIX_THRESHOLD = 18.0  # VIX level above which correlations increase
TARGET_PORTFOLIO_VOL = 0.10  # target annual vol (10%) for volatility targeting
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.4
TARGET_RETURN_ANN = 0.07  # target annual return for MVO (7%)

# ----------------------------
# Utility functions
# ----------------------------
def annualize_vol(daily_vol):
    return daily_vol * np.sqrt(252)

def deannualize_vol(ann_vol):
    return ann_vol / np.sqrt(252)

def fetch_price_data(tickers, start, end):
    # uses yfinance; fetches Adj Close
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data

def fetch_vix(start, end):
    v = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    v.name = "VIX"
    return v

def compute_hist_stats(price_df):
    # daily returns
    ret = price_df.pct_change().dropna()
    mu_daily = ret.mean()
    sigma_daily = ret.std(ddof=1)
    cov_daily = ret.cov()
    corr = ret.corr()
    return ret, mu_daily, sigma_daily, cov_daily, corr

def compute_beta_to_spx(returns_df, spx_ticker="SPY"):
    # rolling beta - here we compute single beta over sample
    if spx_ticker not in returns_df.columns:
        raise ValueError("SPY must be in returns for beta calc, or pass correct market ticker")
    cov = returns_df.cov()
    var_spx = returns_df[spx_ticker].var()
    betas = cov.loc[:, spx_ticker] / var_spx
    return betas

# ----------------------------
# Main pipeline
# ----------------------------
def build_forward_covariance(price_df, vix_series, alpha=ALPHA, gamma=VIX_INFL_CORR_GAMMA, vix_thresh=VIX_THRESHOLD):
    """
    Build covariance matrix adjusted by VIX.
    Approach:
    - Compute hist vol (annualized) per asset.
    - Compute implied vol proxy per asset = beta_to_spy * VIX_annual (simple approach)
    - sigma_adj = alpha * sigma_hist + (1-alpha) * sigma_impl
    - Inflate correlation matrix when VIX > threshold: corr_adj = corr_hist * (1 + gamma * vix_excess_pct)
    - Build cov_adj = diag(sigma_adj) * corr_adj * diag(sigma_adj)
    """
    returns, mu_daily, sigma_daily, cov_daily, corr_hist = compute_hist_stats(price_df)
    sigma_hist_ann = annualize_vol(sigma_daily)

    # latest VIX (use last available)
    vix_latest = float(vix_series.dropna().iloc[-1])  # e.g., 15.0
    vix_ann = vix_latest / 100.0  # e.g., 0.15

    # compute betas to SPY (market)
    betas = compute_beta_to_spx(returns, spx_ticker="SPY")
    # implied vol proxy: scale beta by VIX (assets more correlated to market will have implied vol closer to VIX)
    sigma_impl_ann = (betas.abs() * vix_ann).values  # vector in annual terms
    # fallback: if zero or NaN, use historical vol
    for i, val in enumerate(sigma_impl_ann):
        if np.isnan(val) or val == 0:
            sigma_impl_ann[i] = sigma_hist_ann.iloc[i]

    # adjusted sigma (annual)
    sigma_adj_ann = alpha * sigma_hist_ann.values + (1 - alpha) * sigma_impl_ann
    # ensure min floor
    sigma_adj_ann = np.maximum(sigma_adj_ann, 0.0001)

    # correlation inflation based on VIX
    if vix_latest > vix_thresh:
        excess = (vix_latest - vix_thresh) / vix_thresh  # e.g., 0.2
        corr_adj = corr_hist * (1.0 + gamma * excess)
    else:
        corr_adj = corr_hist

    # cap correlations to reasonable bounds
    corr_adj = corr_adj.clip(lower=-0.95, upper=0.95)

    # assemble covariance (annual)
    D = np.diag(sigma_adj_ann)
    cov_adj_ann = D.dot(corr_adj.values).dot(D)

    # convert annual to daily cov if needed: cov_daily_adj = cov_adj_ann / 252
    return cov_adj_ann, sigma_adj_ann, sigma_hist_ann.values, mu_daily * 252.0, vix_latest

# ----------------------------
# Optimizer
# ----------------------------
def solve_mean_variance(mu_ann, cov_ann, target_return=TARGET_RETURN_ANN, lb=MIN_WEIGHT, ub=MAX_WEIGHT):
    """
    Solve min w^T Σ w s.t. mu^T w >= target_return, sum(w)=1, lb <= w <= ub
    Returns weights as numpy array
    """
    n = len(mu_ann)
    w = cp.Variable(n)
    Sigma = cp.Parameter((n, n), PSD=True)
    mu = cp.Parameter(n)

    Sigma.value = cov_ann
    mu.value = mu_ann

    obj = cp.quad_form(w, Sigma)
    constraints = [
        cp.sum(w) == 1,
        mu @ w >= target_return,
        w >= lb,
        w <= ub
    ]
    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if w.value is None:
        raise RuntimeError("Optimization failed. Try relaxing constraints or changing solver.")
    return np.array(w.value).flatten()

def compute_portfolio_stats(weights, mu_ann, cov_ann):
    port_ret = weights @ mu_ann
    port_vol = np.sqrt(weights @ cov_ann @ weights)
    # approximate sharpe
    sharpe = (port_ret - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0.0
    return port_ret, port_vol, sharpe

# ----------------------------
# Volatility targeting
# ----------------------------
def apply_vol_target(weights, cov_ann, target_vol=TARGET_PORTFOLIO_VOL, max_leverage=3.0):
    _, vol, _ = compute_portfolio_stats(weights, mu_ann=np.zeros(len(weights)), cov_ann=cov_ann)
    if vol == 0:
        return weights, 1.0
    leverage = target_vol / vol
    leverage = min(leverage, max_leverage)
    tgt_weights = weights * leverage
    return tgt_weights, leverage

# ----------------------------
# Hedge sizing (simple estimate)
# ----------------------------
def estimate_vix_hedge_notional(portfolio_value, cov_ann, weights, vix_latest, expected_vix_spike=0.20):
    """
    Simple heuristic:
    - assume a VIX spike raises portfolio var proportionally to sensitivity
    - We'll estimate notional of VIX futures needed to offset a targeted % of expected loss.
    This is heuristic and must be refined for production.
    """
    # current portfolio vol (annual)
    _, port_vol, _ = compute_portfolio_stats(weights, mu_ann=np.zeros(len(weights)), cov_ann=cov_ann)
    # expected loss in a VIX spike scenario (approx)
    expected_loss_pct = min(0.5, expected_vix_spike) * port_vol  # rough heuristic
    dollar_loss = portfolio_value * expected_loss_pct
    # one VIX future point move value depends on contract. We will provide notional in USD to buy VIX exposure,
    # assume 1 VIX futures notional equals 1 * portfolio_value_for_scaling (user must convert to contracts with futures multiplier)
    # So we return suggested VIX exposure (USD) to cover dollar_loss (conservative)
    suggested_vix_exposure = dollar_loss  # naive approach: buy VIX exposure equal to estimated loss
    return {
        "expected_loss_pct_estimate": expected_loss_pct,
        "dollar_loss_estimate": dollar_loss,
        "suggested_vix_notional_usd": suggested_vix_exposure,
        "vix_latest": vix_latest
    }

# ----------------------------
# Running pipeline example
# ----------------------------
if __name__ == "__main__":
    print("Fetching data...")
    tickers = ASSETS.copy()
    if "SPY" not in tickers:
        raise RuntimeError("Example uses SPY as market proxy - include SPY in ASSETS or change beta calc.")
    price_df = fetch_price_data(tickers, START_DATE, END_DATE)
    vix = fetch_vix(START_DATE, END_DATE)

    print(f"Data fetched: {price_df.shape[0]} rows, {len(price_df.columns)} assets. Latest date: {price_df.index[-1].date()}")
    cov_adj_ann, sigma_adj_ann, sigma_hist_ann, mu_ann, vix_latest = build_forward_covariance(price_df, vix)

    # run MVO
    print("Solving mean-variance optimization...")
    try:
        w_mvo = solve_mean_variance(mu_ann.values, cov_adj_ann, target_return=TARGET_RETURN_ANN)
    except Exception as e:
        print("MVO failed:", e)
        # fallback: maximize return for given bound constraints (no target)
        n = len(mu_ann)
        w_mvo = np.ones(n) / n
        print("Using equal weight fallback.")

    # compute stats
    port_ret, port_vol, sharpe = compute_portfolio_stats(w_mvo, mu_ann.values, cov_adj_ann)
    print("\n--- Optimized Portfolio (MVO) ---")
    df_out = pd.DataFrame({
        "ticker": tickers,
        "weight": w_mvo,
        "hist_ann_vol": sigma_hist_ann,
        "adj_ann_vol": sigma_adj_ann,
        "mu_ann": mu_ann.values
    })
    print(df_out)
    print(f"\nPortfolio expected annual return: {port_ret:.4f}, vol: {port_vol:.4f}, sharpe: {sharpe:.3f}")
    print(f"Current VIX: {vix_latest:.2f}")

    # apply volatility targeting
    print("\nApplying volatility target scaling...")
    tgt_weights, leverage = apply_vol_target(w_mvo, cov_adj_ann, target_vol=TARGET_PORTFOLIO_VOL)
    print(f"Leverage applied: {leverage:.3f}")
    df_out["weight_vol_target"] = tgt_weights
    print(df_out[["ticker", "weight", "weight_vol_target"]])

    # hedge suggestion
    portfolio_value_usd = 10_000_000  # example portfolio size; change as needed when offering product
    hedge = estimate_vix_hedge_notional(portfolio_value_usd, cov_adj_ann, tgt_weights, vix_latest)
    print("\nHEDGE SUGGESTION (heuristic):")
    for k, v in hedge.items():
        print(f"{k}: {v}")

    # quick plot of allocations
    plt.figure(figsize=(8, 4))
    plt.bar(df_out["ticker"], df_out["weight_vol_target"])
    plt.title("Volatility-targeted weights")
    plt.ylabel("Weight")
    plt.show()

