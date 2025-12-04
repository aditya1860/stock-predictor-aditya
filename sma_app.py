import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


# ------------------- STYLE --------------------

CUSTOM_CSS = """
<style>
/* Background */
.main {
    background-color: #0f0f0f;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111;
}

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff !important;
}

/* Normal text – scope only to main + sidebar (NOT global) */
.main p, .main label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #dddddd !important;
}

/* Metric boxes */
[data-testid="metric-container"] {
    background-color: #1a1a1a;
    border: 1px solid #333;
    padding: 12px;
    border-radius: 12px;
}

/* Divider */
hr {
    border: 1px solid #222;
}

/* Inputs */
input, textarea {
    background-color: #151515 !important;
    color: #eee !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton button {
    background-color: #ffffff10;
    color: #ddd;
    padding: 10px 20px;
    border: 1px solid #444;
    border-radius: 10px;
}
.stButton button:hover {
    background-color: #ffffff20;
}
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------- FUNCTIONS -------------------

def get_data(ticker, start="2015-01-01"):
    data = yf.download(ticker, start=start)
    data = data[["Close"]].copy()
    data.dropna(inplace=True)
    return data


def sma_strategy(df, short=20, long=50):
    df = df.copy()
    df["SMA_short"] = df["Close"].rolling(short).mean()
    df["SMA_long"] = df["Close"].rolling(long).mean()
    df["Position"] = (df["SMA_short"] > df["SMA_long"]).astype(int)
    df["Return"] = df["Close"].pct_change()
    df["StrategyReturn"] = df["Position"].shift(1) * df["Return"]
    df.dropna(inplace=True)
    return df


def stats(df):
    bh = (1 + df["Return"]).prod() - 1
    stg = (1 + df["StrategyReturn"]).prod() - 1

    bh_curve = (1 + df["Return"]).cumprod()
    stg_curve = (1 + df["StrategyReturn"]).cumprod()

    return bh, stg, bh_curve, stg_curve


# ------------------- UI -----------------------

st.title("SMA Crossover Backtest Dashboard")

st.caption(
    "A technical strategy based on moving averages. "
    "This tool lets you explore how different SMA settings perform historically."
)

st.divider()

# Sidebar
st.sidebar.title("Strategy Configuration")

symbols = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "NIFTYBEES.NS"
]

ticker = st.sidebar.selectbox("Select Stock", symbols)

short = st.sidebar.slider("Short SMA", 5, 100, 20)
long = st.sidebar.slider("Long SMA", 20, 300, 50)
start = st.sidebar.text_input("Data Start", "2015-01-01")

run = st.sidebar.button("Run Backtest")


# ------------------- LOGIC -----------------------

if run:
    with st.spinner("Fetching data..."):
        df = get_data(ticker, start)

    st.subheader(f"{ticker} — Price")
    st.line_chart(df["Close"], height=260)

    with st.spinner("Applying strategy..."):
        df2 = sma_strategy(df, short, long)
        bh, stg, bh_curve, stg_curve = stats(df2)

    st.divider()
    st.subheader("Performance Summary")

    c1, c2 = st.columns(2)
    c1.metric("Buy & Hold Return", f"{bh*100:.2f}%")
    c2.metric("SMA Strategy Return", f"{stg*100:.2f}%")

    st.divider()
    st.subheader("Equity Curve")
    eq = pd.DataFrame({"Buy & Hold": bh_curve, "SMA Strategy": stg_curve})
    st.line_chart(eq, height=260)

    st.caption(
        "Disclaimer: Historical performance does not guarantee future results."
    )
else:
    st.info("Use the sidebar to configure strategy, then click **Run Backtest**.")
