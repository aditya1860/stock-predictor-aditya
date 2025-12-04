import yfinance as yf
import pandas as pd


def download_data(ticker="RELIANCE.NS", start="2015-01-01"):
    data = yf.download(ticker, start=start)
    data = data[["Close"]].copy()
    data.dropna(inplace=True)
    return data


def add_sma_signals(df, short_window=20, long_window=50):
    df = df.copy()

    # Moving averages
    df["SMA_short"] = df["Close"].rolling(short_window).mean()
    df["SMA_long"] = df["Close"].rolling(long_window).mean()

    # Strategy signals
    df["Position"] = 0
    df.loc[df["SMA_short"] > df["SMA_long"], "Position"] = 1

    # Daily returns
    df["Return"] = df["Close"].pct_change()

    # Strategy return = previous position * today's return
    df["StrategyReturn"] = df["Position"].shift(1) * df["Return"]

    df.dropna(inplace=True)
    return df


def summarize_performance(df):
    # Cum returns
    buy_hold_total = (1 + df["Return"]).prod() - 1
    strat_total = (1 + df["StrategyReturn"]).prod() - 1

    print("===== PERFORMANCE SUMMARY =====")
    print(f"Total Buy & Hold Return: {buy_hold_total * 100:.2f}%")
    print(f"Total Strategy Return   : {strat_total * 100:.2f}%")
    print("===============================")


def main():
    ticker = "RELIANCE.NS"
    print(f"Downloading data for {ticker}...")
    df = download_data(ticker)

    print("Running SMA Crossover Strategy...")
    df = add_sma_signals(df, short_window=20, long_window=50)

    summarize_performance(df)


if __name__ == "__main__":
    main()
