import yfinance as yf
import pandas as pd

def fetch_and_save_sp500(start="2010-01-01", end="2016-12-30", output_path="data/sp500.csv"):
    ticker = "^GSPC"
    df = yf.download(tickers=ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)

    df = df[['Close']] 
    df.rename(columns={'Close': 'Close'}, inplace=True)
    df.to_csv(output_path, index_label='Date')

    print(f"[✓] S&P 500 data saved to: {output_path}")

if __name__ == "__main__":
    fetch_and_save_sp500()