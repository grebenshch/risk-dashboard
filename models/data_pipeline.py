import yfinance as yf
import pandas as pd
import sqlite3
import os

# Our 15 stocks
TICKERS = [
    "JPM", "GS", "HLI",        # Financials
    "AAPL", "PLTR", "META",     # Tech
    "JNJ", "MRK", "UNH",       # Healthcare
    "MCD", "KO",                # Consumer Staples
    "EL", "ULTA",               # Beauty
    "XOM", "CVX"                # Energy
]

# Create folders if not exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

def download_historical_data(tickers=TICKERS, start="2022-01-01", end="2025-01-01"):
    """Download historical closing prices"""
    print("Downloading historical data...")
    df = yf.download(tickers, start=start, end=end)["Close"]
    df.dropna(how="all", inplace=True)
    df.ffill(inplace=True)
    df.to_csv("data/raw/prices.csv")
    print(f"Saved {len(df)} rows to data/raw/prices.csv")
    return df

def compute_returns(prices_df):
    """Compute daily log returns"""
    import numpy as np
    returns = np.log(prices_df / prices_df.shift(1)).dropna()
    returns.to_csv("data/processed/returns_clean.csv")
    print(f"Saved returns to data/processed/returns_clean.csv")
    return returns

def save_to_sqlite(prices_df, returns_df):
    """Save data to SQLite database"""
    conn = sqlite3.connect("data/risk_data.db")
    prices_df.to_sql("prices", conn, if_exists="replace")
    returns_df.to_sql("returns", conn, if_exists="replace")
    conn.close()
    print("Saved to data/risk_data.db")

if __name__ == "__main__":
    prices = download_historical_data()
    returns = compute_returns(prices)
    save_to_sqlite(prices, returns)
    print("\nData pipeline complete!")
    print(f"Stocks: {len(prices.columns)}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Total trading days: {len(prices)}")