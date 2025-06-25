import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scripts.backtester as backtester
import scripts.visualizer as visualizer

st.set_page_config(layout="wide")
st.title("📊 Quantitative Factor Investing Simulator")

# Sidebar selection
tab = st.sidebar.radio("Select View", [
    "📈 Rule-Based Strategy",
    "🤖 ML-Based Strategy",
    "📊 Benchmark Comparison"
])

@st.cache_data
def load_all_data():
    rule_equal = pd.read_csv("data/processed_not_normalized/portfolio_weight/equal_weight_portfolio.csv", parse_dates=['Date'], index_col='Date')
    rule_risk = pd.read_csv("data/processed_not_normalized/portfolio_weight/risk_adjusted_portfolio.csv", parse_dates=['Date'], index_col='Date')
    ml_equal = pd.read_csv("data/processed_not_normalized/predicted_portfolio_scores/equal_weight_portfolio_ml.csv", parse_dates=['Date'], index_col='Date')
    ml_risk = pd.read_csv("data/processed_not_normalized/predicted_portfolio_scores/risk_adjusted_portfolio_ml.csv", parse_dates=['Date'], index_col='Date')
    sp500 = pd.read_csv("data/sp500.csv", parse_dates=['Date'], index_col='Date')
    return rule_equal, rule_risk, ml_equal, ml_risk, sp500

rule_equal, rule_risk, ml_equal, ml_risk, sp500 = load_all_data()

if tab == "📈 Rule-Based Strategy":
    st.header("Rule-Based Portfolio Backtesting")
    visualizer.plot_cumulative_returns(rule_equal, rule_risk)
    visualizer.plot_drawdowns(rule_equal, rule_risk)
    visualizer.plot_histogram(rule_equal, rule_risk)
    visualizer.plot_rolling_sharpe(rule_equal, rule_risk)

elif tab == "🤖 ML-Based Strategy":
    st.header("ML-Based Portfolio Backtesting")
    visualizer.plot_cumulative_returns(ml_equal, ml_risk)
    visualizer.plot_drawdowns(ml_equal, ml_risk)
    visualizer.plot_histogram(ml_equal, ml_risk)
    visualizer.plot_rolling_sharpe(ml_equal, ml_risk)

elif tab == "📊 Benchmark Comparison":
    st.header("Portfolio vs S&P 500")
    visualizer.plot_benchmark_comparison(rule_equal, rule_risk, ml_equal, ml_risk, sp500)