# 📈 ***Quantitative Factor Investing Simulator***

A comprehensive simulator for multi-factor stock investing that leverages fundamental data, machine learning, and portfolio optimization to construct and evaluate outperforming strategies - complete with an interactive dashboard.

---

## 🚀 ***Overview***

This project implements a **Quantitative Factor Investing** framework using multiple factor models (Value, Momentum, Quality, Volume, Volatility). It enables:
- Clean data ingestion from raw financial datasets
- Computation of factor scores using both existing rules and an ML model.
- Construction of a composite score using custom weights
- Stock ranking and portfolio creation (equal-weighted and risk-adjusted)
- Machine learning-based return prediction
- Backtesting and performance comparison of ML-based portfolio and Rule-based portfolio vs S&P 500
- Visualization through a live dashboard 

> **Result**: The constructed portfolio **outperformed the S&P 500 Index** consistently across backtesting periods, with higher cumulative returns and a better risk-adjusted profile (Sharpe ratio). The results are displayed in the report.

---

## 🧩 ***Features***

- ***Data Loader***: Load & clean price, fundamental, and security datasets
- ***Factor Scoring***: Value, Momentum, Quality, Volume, Volatility
- ***Composite Score***: Custom weighted average of individual factor scores
- ***ML Model***: Predictive modeling of stock returns (RandomForest Regressor)
- ***Portfolio Construction:***
  - Top-50 stocks selected based on composite/ML predicted scores
  - Equal weighting & volatility-adjusted weighting strategies
- ***Backtesting Engine:***
  - Cumulative return plots
  - S&P 500 benchmark comparison
  - Sharpe, volatility, drawdown metrics
- ***Streamlit Dashboard:***
  - Interactive filters, stock tables, and charts for visualizations
  - Dark theme, responsive layout

---

## 📊 ***Results: Portfolio vs S&P 500***

### Rule-Based Portfolio
| Metric                  | Portfolio (Risk-Adjusted) | S&P 500 Index |
|-------------------------|---------------------------|---------------|
| Cumulative Return       | 211.41%                   | 71.3%         |
| Annualized Return       | 19.2%                     | 15.6%         |
| Sharpe Ratio            | 0.862                     | 0.86          |
| Max Drawdown            | -39.21%                   | -13.4%        |

### ML-Based Portfolio
| Metric                  | Portfolio (Risk-Adjusted) | S&P 500 Index |
|-------------------------|---------------------------|---------------|
| Cumulative Return       | 145.57%                   | 71.3%         |
| Annualized Return       | 28.86%                    | 15.6%         |
| Sharpe Ratio            | 1.385                     | 0.86          |
| Max Drawdown            | -32.07%                   | -13.4%        |

> Outperformance is driven by factor-based filtering, ML-enhanced ranking, and dynamic weighting in both portfolios when compared to the S&P 500 index over that period.

---

## 📄 ***Detailed Report***

For a complete walkthrough of:
- Data pipeline and methodology
- Feature engineering and factor design
- ML modeling decisions
- Portfolio logic and weighting strategies
- Graphs, visualizations, and dashboards

Please refer to the attached [`project_report.pdf`](./docs/project_report.pdf)

---

## 🛠️ ***Tech Stack***
	•	Python 3.10+
	•	Pandas, NumPy, Scikit-learn 
	•	Matplotlib, Seaborn, Plotly
	•	Streamlit (dashboard)
	•	PostgreSQL (data backend)
	•	SQLAlchemy / psycopg2 (DB connector)
	•	Random Forest Regressor (ML model)

---
## ⚙️ ***How to Run***

1. Clone the repo -
`git clone https://github.com/Advaith1509/quant-factor-simulator.git
cd quant-factor-simulator`

2. Install dependencies -
`pip install -r requirements.txt`

3. Set up .env with DB credentials.
   
4. Run Streamlit App -
`streamlit run app/app.py`

---
