import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def load_price_data(price_file):
    prices = pd.read_csv(price_file)
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.sort_values(by=['date']).drop_duplicates(subset=['date', 'symbol'], keep='last')
    return prices.pivot(index='date', columns='symbol', values='close')

def calculate_daily_returns(prices):
    return prices.pct_change(fill_method=None).fillna(0)

def calculate_portfolio_returns(portfolio_weights, daily_returns):
    portfolio_returns = pd.Series(index=daily_returns.index, dtype=float)

    if not pd.api.types.is_datetime64_any_dtype(portfolio_weights['rebalance_date']):
        portfolio_weights['rebalance_date'] = pd.to_datetime(portfolio_weights['rebalance_date'])

    portfolio_weights = portfolio_weights.sort_values('rebalance_date')
    rebalance_dates = portfolio_weights['rebalance_date'].drop_duplicates().sort_values().tolist()
    n = len(rebalance_dates)

    for i, rebalance_date in enumerate(rebalance_dates):
        if i < n - 1:
            next_rebalance = rebalance_dates[i + 1]
            period_mask = (daily_returns.index > rebalance_date) & (daily_returns.index <= next_rebalance)
        else:
            period_mask = (daily_returns.index > rebalance_date)

        period_dates = daily_returns.index[period_mask]
        group = portfolio_weights[portfolio_weights['rebalance_date'] == rebalance_date]
        weights = group.set_index('symbol')['weight']
        valid_stocks = weights.index.intersection(daily_returns.columns)
        weights = weights[valid_stocks]

        if len(valid_stocks) == 0:
            continue

        weights = weights / weights.sum()

        if len(period_dates) > 0:
            returns_slice = daily_returns.loc[period_dates, valid_stocks]
            returns_slice = returns_slice[weights.index]  # reorder to match weights
            period_returns = returns_slice.dot(weights)
            portfolio_returns.loc[period_dates] = period_returns

    return portfolio_returns.dropna()

def calculate_metrics(returns, risk_free_rate=0.0):
    cumulative_returns = (1 + returns).cumprod() - 1
    total_return = cumulative_returns.iloc[-1]
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol != 0 else np.nan
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    return {
        'cumulative_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def calculate_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    return (cumulative - peak) / peak

def run_backtest(portfolio_file, price_file):
    portfolio_weights = pd.read_csv(portfolio_file)
    prices = load_price_data(price_file)
    daily_returns = calculate_daily_returns(prices)
    portfolio_returns = calculate_portfolio_returns(portfolio_weights, daily_returns)
    metrics = calculate_metrics(portfolio_returns)
    return portfolio_returns, metrics