import yfinance as yf
import pandas as pd
from datetime import datetime

# Our 15 stocks
TICKERS = [
    "JPM", "GS", "HLI",        # Financials
    "AAPL", "PLTR", "META",     # Tech
    "JNJ", "MRK", "UNH",       # Healthcare
    "MCD", "KO",                # Consumer Staples
    "EL", "ULTA",               # Beauty
    "XOM", "CVX"                # Energy
]

def get_live_prices(tickers=TICKERS):
    """Fetch latest prices for all tickers"""
    prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d", interval="1m")
            if not data.empty:
                latest_price = data["Close"].iloc[-1]
                prices[ticker] = round(latest_price, 2)
            else:
                prices[ticker] = None
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            prices[ticker] = None
    return prices

def get_live_feed():
    """Get live prices and return as DataFrame"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fetching live prices...")
    prices = get_live_prices()
    df = pd.DataFrame(list(prices.items()), columns=["Ticker", "Price"])
    df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    df = get_live_feed()
    