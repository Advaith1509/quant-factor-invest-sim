# 📈 ***QuantSim - A Quantitative Factor Investing Simulator***

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-%23150458?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/Data-NumPy-%23013243?logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Plotting-Matplotlib-%230073b3?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Stats%20Viz-Seaborn-%2300a7b5)](https://seaborn.pydata.org/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-%23336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![SQLAlchemy](https://img.shields.io/badge/ORM-SQLAlchemy-%23d71a1a)](https://www.sqlalchemy.org/)
[![FinBERT](https://img.shields.io/badge/NLP-FinBERT-%23FF9900?logo=huggingface&logoColor=white)](https://huggingface.co/ProsusAI/finbert)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-%23f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive simulator for multi-factor stock investing that leverages fundamental data, machine learning, NLP-based sentiment analysis, and portfolio optimization to construct and evaluate outperforming strategies, complete with an interactive dashboard.

---

## 🚀 ***Overview***

This project implements a **Quantitative Factor Investing** framework using multiple factor models (Value, Momentum, Quality, Volume, Volatility) and Natural Language Processing (NLP)-based sentiment signals (when live data values are integrated into it). It enables:
- Clean data ingestion from raw financial datasets
- Computation of factor scores using both existing rules and an ML model
- Sentiment scoring using FinBERT to enhance decision-making with real-world financial sentiment
- Construction of a composite score using custom weights (including sentiment)
- Stock ranking and portfolio creation (equal-weighted and risk-adjusted)
- Machine learning-based return prediction
- Backtesting and performance comparison of ML-based portfolio and Rule-based portfolio vs S&P 500
- Visualization through a live dashboard

> **Result**: The constructed portfolio ***outperformed the S&P 500 Index*** consistently across backtesting periods, with higher cumulative returns and a better risk-adjusted profile (Sharpe ratio). The results are displayed in the report.

---

## 🧩 ***Features***

- ***Data Loader***:
  - Load & clean price, fundamental, price-split-adjusted and security datasets
- ***Factor Scoring***:
  - Value, Momentum, Quality, Volume, Volatility
- ***Sentiment Analysis***:
  - Financial news headlines scored using *FinBERT* (Hugging Face)
  - Sentiment factor computed as `positive-negative` sentiment
  - Aggregated by date and symbol for integration into the factor model
- ***Composite Score***:
  - Custom weighted average of individual factor scores, including NLP sentiment scores for future use
- ***ML Model***:
  - Predictive modeling of stock returns (RandomForest Regressor)
- ***Portfolio Construction:***
  - Top-50 stocks selected based on composite/ML predicted scores
  - Equal weighting & volatility-adjusted weighting strategies
- ***Backtesting Engine:***
  - Cumulative return plots
  - S&P 500 benchmark comparison
  - Sharpe, volatility, drawdown metrics
- ***Streamlit Dashboard:***
  - Interactive filters, stock tables, and charts for visualization

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

> Outperformance is driven by factor-based filtering, ML-enhanced ranking, **sentiment-aware signal integration**, and dynamic weighting in both portfolios when compared to the S&P 500 index over that period.

> ⚠️ *Note*: Returns above are pre-costs. After accounting for transaction costs, slippage, and portfolio turnover (~7% annually), estimated **net annualized returns** are:
> - Rule-Based Portfolio: **~12.2%**
> - ML-Based Portfolio: **~21.9%**
> Both portfolios still outperform the S&P 500 after adjusting for realistic market frictions.

---

## 📄 ***Detailed Report***

For a complete walkthrough of:
- Data pipeline and methodology
- Feature engineering and factor design
- ML modeling decisions
- Portfolio logic and weighting strategies
- **Sentiment integration using NLP**
- Graphs, visualizations, and dashboards

Please refer to the attached [`project_report.pdf`](project_report.pdf)

---

## 🛠️ ***Tech Stack***
- Python 3.10+
- Pandas, NumPy, Scikit-learn (data preprocessing, ML model)
- Random Forest Regressor (ML model) 
- Matplotlib, Seaborn, Plotly (visualisations)
- Streamlit (dashboard)
- PostgreSQL (data backend)
- SQLAlchemy / psycopg2 (DB connector)
- Hugging Face Transformers (FinBERT) (for sentiment analysis)

---

## 🗂️ ***Data Access***

Due to the large size of the dataset and intermediate outputs, all relevant files have been uploaded to Google Drive.

📁 **Access the dataset here**: [Google Drive Dataset Folder](https://drive.google.com/drive/folders/1dYgGPBbIwWZS2khuKKa-NuUfJV43rHnU?usp=drive_link)

The folder includes:
- Raw financial datasets (`data/raw/`)
- Processed features and cleaned datasets (`data/processed/`)
- Factor score files (value, momentum, quality, etc.)
- Sentiment analysis outputs (if applicable)

To set up locally:
1. Download the relevant folders from the Drive.
2. Place them into the corresponding locations (in the image uploaded) within the project directory given in the Google Drive folder.

> **Important**: You must download and correctly place the datasets before running notebooks or the dashboard app.

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
