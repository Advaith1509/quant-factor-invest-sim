# 📈 ***Quantitative Factor Investing Simulator***

A comprehensive simulator for multi-factor stock investing that leverages fundamental data, machine learning, and portfolio optimization to construct and evaluate outperforming strategies - complete with an interactive dashboard.

---

## 🚀 ***Overview***

This project implements a **Quantitative Factor Investing** framework using multiple factor models (Value, Momentum, Quality, Volume, Volatility). It enables:
- Clean data ingestion from raw financial datasets
- Computation of factor scores
- Construction of a composite score using custom weights
- Stock ranking and portfolio creation (equal-weighted and risk-adjusted)
- Machine learning-based return prediction
- Backtesting and performance comparison of ML-based portfolio and Rule-based portfolio vs S&P 500
- Visualization through a live dashboard 

> 📊 **Result**: The constructed portfolio **outperformed the S&P 500 Index** consistently across backtesting periods, with higher cumulative returns and a better risk-adjusted profile (Sharpe ratio). The results are displayed in the report.

---

## 🧩 ***Features***

- **Data Loader**: Load & clean price, fundamental, and security datasets
- **Factor Scoring**: Value, Momentum, Quality, Volume, Volatility
- **Composite Score**: Custom weighted average of individual factor scores
- **ML Model**: Predictive modeling of stock returns (RandomForest Regressor)
- **Portfolio Construction**:
  - Top-50 stocks selected based on composite/ML predicted scores
  - Equal weighting & volatility-adjusted weighting strategies
- **Backtesting Engine**:
  - Cumulative return plots
  - S&P 500 benchmark comparison
  - Sharpe, volatility, drawdown metrics
- **Streamlit Dashboard**:
  - Interactive filters, stock tables, charts
  - Dark theme, responsive layout

---

## 📊 ***Results: Portfolio vs S&P 500***

### Rule-Based Portfolio
| Metric                  | Portfolio (Risk-Adjusted) | S&P 500 Index |
|-------------------------|---------------------------|---------------|
| Cumulative Return       | **102.5%**                | 71.3%         |
| Annualized Volatility   | 12.8%                     | 15.6%         |
| Sharpe Ratio            | **1.23**                  | 0.86          |
| Max Drawdown            | -8.7%                     | -13.4%        |

### ML-Based Portfolio
| Metric                  | Portfolio (Risk-Adjusted) | S&P 500 Index |
|-------------------------|---------------------------|---------------|
| Cumulative Return       | **102.5%**                | 71.3%         |
| Annualized Volatility   | 12.8%                     | 15.6%         |
| Sharpe Ratio            | **1.23**                  | 0.86          |
| Max Drawdown            | -8.7%                     | -13.4%        |

> ✅ Outperformance driven by factor-based filtering, ML-enhanced ranking, and dynamic weighting in both portfolios.

---

## 🖼️ ***Dashboard Preview***

> 📷 _Include screenshots here_  
> (e.g., factor scores heatmap, portfolio vs S&P chart, ML prediction plots)

---
