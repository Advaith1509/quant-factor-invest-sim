import matplotlib.pyplot as plt
import streamlit as st

def plot_cumulative_returns(df1, df2):
    fig, ax = plt.subplots()
    ax.plot(df1['Cumulative'], label='Equal Weight')
    ax.plot(df2['Cumulative'], label='Risk-Adjusted')
    ax.set_title("Cumulative Returns Comparison")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)

def plot_drawdowns(df1, df2):
    fig, ax = plt.subplots()
    ax.plot(df1['Drawdown'], label='Equal Weight')
    ax.plot(df2['Drawdown'], label='Risk-Adjusted')
    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)

def plot_histogram(df1, df2):
    fig, ax = plt.subplots()
    ax.hist(df1['Daily Return'], bins=50, alpha=0.6, label='Equal Weight')
    ax.hist(df2['Daily Return'], bins=50, alpha=0.6, label='Risk-Adjusted')
    ax.set_title("Histogram of Daily Returns")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

def plot_rolling_sharpe(df1, df2):
    fig, ax = plt.subplots()
    ax.plot(df1['Rolling Sharpe'], label='Equal Weight')
    ax.plot(df2['Rolling Sharpe'], label='Risk-Adjusted')
    ax.set_title("Rolling Sharpe Ratio (window=63)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)

def plot_benchmark_comparison(re_eq, re_risk, ml_eq, ml_risk, sp500):
    fig, ax = plt.subplots()
    ax.plot(sp500['Close'] / sp500['Close'].iloc[0], label='S&P 500', linestyle='--', color='black')
    ax.plot(re_eq['Cumulative'], label='Equal Weight (Rule-Based)')
    ax.plot(re_risk['Cumulative'], label='Risk-Adjusted (Rule-Based)')
    ax.plot(ml_eq['Cumulative'], label='Equal Weight (ML-Based)', alpha=0.7)
    ax.plot(ml_risk['Cumulative'], label='Risk-Adjusted (ML-Based)', alpha=0.7)
    ax.set_title("Portfolio NAV vs S&P 500")
    ax.set_ylabel("NAV (normalized)")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)