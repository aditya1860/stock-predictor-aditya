import yfinance as yf

def main():
    # 1. Kis stock ka data chahiye? (example: RELIANCE.NS)
    ticker = "RELIANCE.NS"

    print(f"Downloading data for {ticker} ...")

    # 2. Data download karo (pichle 10 saal ka approx)
    data = yf.download(ticker, start="2015-01-01")

    # 3. CSV file me save karo
    csv_name = "reliance_daily.csv"
    data.to_csv(csv_name)

    print(f"Data downloaded and saved to {csv_name}")
    print("Rows:", len(data))

if __name__ == "__main__":
    main()
