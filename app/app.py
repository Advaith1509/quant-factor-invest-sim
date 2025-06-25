import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import scripts.backtester as backtester
import scripts.visualizer as visualizer


# Page configuration
st.set_page_config(page_title="Quantitative Factor Investing", layout="wide")
st.title("📊 Quantitative Factor Investing Simulator")

# Sidebar Navigation
tab = st.sidebar.radio("Select Strategy View", [
    "📈 Rule-Based Backtest",
    "🤖 ML-Based Backtest",
    "📊 Compare to S&P 500"
])

def read_portfolio(path):
    df = pd.read_csv(path)
    # Find date column (case-insensitive)
    date_col = next((col for col in df.columns if 'date' in col.lower()), None)
    if not date_col:
        raise ValueError(f"No date column found in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col)

# Cached data loader
@st.cache_data
def load_data():
    def read_portfolio(path):
        df = pd.read_csv(path)
        date_col = 'Date' if 'Date' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col])
        return df.set_index(date_col)
    
    rule_equal = pd.read_csv("data/processed_not_normalized/portfolio_weight/equal_weight_portfolio.csv", parse_dates=['Date'], index_col='Date')
    rule_risk = pd.read_csv("data/processed_not_normalized/portfolio_weight/risk_adjusted_portfolio.csv", parse_dates=['Date'], index_col='Date')
    ml_equal = pd.read_csv("data/processed_not_normalized/predicted_portfolio_scores/equal_weight_portfolio_ml.csv", parse_dates=['Date'], index_col='Date')
    ml_risk = pd.read_csv("data/processed_not_normalized/predicted_portfolio_scores/risk_adjusted_portfolio_ml.csv", parse_dates=['Date'], index_col='Date')
    sp500 = pd.read_csv("data/sp500.csv", parse_dates=True, index_col=0)
    return rule_equal, rule_risk, ml_equal, ml_risk, sp500

rule_equal, rule_risk, ml_equal, ml_risk, sp500 = load_data()

if tab == "📈 Rule-Based Backtest":
    st.header("📈 Rule-Based Strategy")
    visualizer.plot_cumulative_returns(rule_equal, rule_risk)
    visualizer.plot_drawdowns(rule_equal, rule_risk)
    visualizer.plot_histogram(rule_equal, rule_risk)
    visualizer.plot_rolling_sharpe(rule_equal, rule_risk)

elif tab == "🤖 ML-Based Backtest":
    st.header("🤖 ML-Based Strategy")
    visualizer.plot_cumulative_returns(ml_equal, ml_risk)
    visualizer.plot_drawdowns(ml_equal, ml_risk)
    visualizer.plot_histogram(ml_equal, ml_risk)
    visualizer.plot_rolling_sharpe(ml_equal, ml_risk)

elif tab == "📊 Compare to S&P 500":
    st.header("📊 Benchmark Comparison")
    visualizer.plot_benchmark_comparison(rule_equal, rule_risk, ml_equal, ml_risk, sp500)