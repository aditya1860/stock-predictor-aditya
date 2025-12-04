import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- sklearn import guard (safe for deployments that can't build sklearn) ---
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except Exception:
    # sklearn not available in this environment (Streamlit Cloud). We'll disable training/prediction.
    SKLEARN_AVAILABLE = False
# -------------------------------------------------------------------------


# ===================== GLOBAL UI STYLE ===========================
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
/* ========== FIX SELECTBOX ONLY (NO PAGE DESIGN CHANGE) ========== */

/* Outer box (input) */
.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 8px !important;
    border: 1px solid #444444 !important;
}

/* Selected text */
.stSelectbox div[data-baseweb="select"] span {
    color: #000000 !important;
}

/* Dropdown panel */
.stSelectbox div[role="listbox"] {
    background-color: #ffffff !important;
    border: 1px solid #cccccc !important;
    border-radius: 8px !important;
}

/* Options */
.stSelectbox div[role="option"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    padding: 6px 10px !important;
}

/* Hover option */
.stSelectbox div[role="option"]:hover {
    background-color: #e9e9e9 !important;
    color: #000000 !important;
}

</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ===================== DATA ENGINE ==============================

@st.cache_data
def download_price_data(ticker: str, start: str = "2015-01-01") -> pd.DataFrame:
    data = yf.download(ticker, start=start)
    return data[["Open", "High", "Low", "Close", "Volume"]].dropna()


def build_features(df: pd.DataFrame):
    df = df.copy()
    df["ret1"] = df["Close"].pct_change()
    df["ret5"] = df["Close"].pct_change(5)
    df["ret10"] = df["Close"].pct_change(10)
    df["vol_chg"] = df["Volume"].pct_change()

    df["sma5"] = df["Close"].rolling(5).mean()
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()

    df["sma5_20"] = df["sma5"] / df["sma20"] - 1
    df["sma20_50"] = df["sma20"] / df["sma50"] - 1

    df["target"] = df["ret1"].shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    features = ["ret1", "ret5", "ret10", "vol_chg", "sma5_20", "sma20_50"]
    X = df[features]
    y = df["target"]
    return df, X, y, features


def train_model(X, y):
    idx = int(len(X) * 0.8)
    Xtr, Xte = X.iloc[:idx], X.iloc[idx:]
    ytr, yte = y.iloc[:idx], y.iloc[idx:]

    model = RandomForestRegressor(
        n_estimators=450,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    rmse = mean_squared_error(yte, pred) ** 0.5
    acc = (np.sign(pred) == np.sign(yte)).mean()
    return model, rmse, acc


def predict_next(model, df_feat, feat_cols, max_move_frac):
    row = df_feat.iloc[-1:][feat_cols]
    last = float(df_feat["Close"].iloc[-1])

    raw_ret = float(model.predict(row)[0])
    clipped_ret = float(np.clip(raw_ret, -max_move_frac, max_move_frac))

    next_price = last * (1 + clipped_ret)
    return last, clipped_ret, next_price


# ===================== SMA BACKTEST ==============================

def sma_backtest(df_price, short_win, long_win):
    d = df_price[["Close"]].copy()
    d["SMA_S"] = d["Close"].rolling(short_win).mean()
    d["SMA_L"] = d["Close"].rolling(long_win).mean()

    d["pos"] = (d["SMA_S"] > d["SMA_L"]).astype(int)
    d["ret"] = d["Close"].pct_change()
    d["str_ret"] = d["ret"] * d["pos"].shift(1)
    d.dropna(inplace=True)

    buy_hold = (1 + d["ret"]).prod() - 1
    strat = (1 + d["str_ret"]).prod() - 1

    bh_curve = (1 + d["ret"]).cumprod()
    st_curve = (1 + d["str_ret"]).cumprod()

    return d, bh_curve, st_curve, buy_hold, strat


# =================== UI LAYOUT ==============================

st.title("Market Intelligence Dashboard")

stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "HINDUNILVR.NS", "NIFTYBEES.NS",
]

st.sidebar.header("Parameters")
tk = st.sidebar.selectbox("Stock", stocks, index=0)
start_date = st.sidebar.text_input("Start date", "2015-01-01")

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction")
max_move = st.sidebar.slider("Clip move (%)", 2, 10, 6)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest")
sma_s = st.sidebar.slider("Short SMA", 5, 90, 20)
sma_l = st.sidebar.slider("Long SMA", 20, 300, 50)

run = st.sidebar.button("Run")


if run:
    # ----- PRICE DATA -----
    price_df = download_price_data(tk, start_date)

    st.subheader(f"{tk} • Price")

    # Altair line chart for price (bright cyan on dark bg)
    price_chart = (
        alt.Chart(price_df.reset_index())
        .mark_line(color="#4FC3F7")
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Close:Q", title="Close Price"),
        )
        .properties(height=260)
    )
    st.altair_chart(price_chart, use_container_width=True)

    tabs = st.tabs(["Prediction", "Backtest"])

    # ---------- PREDICTION TAB ----------
    with tabs[0]:
        st.markdown("### Model Based Next-Day Forecast")

        feat_df, X, y, feat_cols = build_features(price_df)
        if len(X) < 200:
            st.error("Data is very less so can not be predicted.")
        else:
            model, rmse, acc = train_model(X, y)
            last, move_frac, next_p = predict_next(
                model, feat_df, feat_cols, max_move / 100.0
            )

            change = next_p - last
            pct = move_frac * 100.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Last Close", f"₹{last:.2f}")
            c2.metric("Forecast Close", f"₹{next_p:.2f}", f"{change:+.2f}")
            c3.metric("Expected Move", f"{pct:+.2f}%")

            st.markdown("### Model Diagnostics")
            d1, d2 = st.columns(2)
            d1.metric("RMSE (returns)", f"{rmse:.4f}")
            d2.metric("Direction Accuracy", f"{acc*100:.1f}%")

            st.caption(
                f"Note: Raw prediction ko realistic range (±{max_move}%) ."
            )

    # ---------- BACKTEST TAB ----------
    with tabs[1]:
        st.markdown("### SMA Crossover Backtest")

        if sma_s >= sma_l:
            st.error("Keep Short SMA smaller than Long SMA(e.g. 20 & 50).")
        else:
            bt_df, bh_curve, st_curve, bh_ret, st_ret = sma_backtest(
                price_df, sma_s, sma_l
            )

            c1, c2 = st.columns(2)
            c1.metric("Buy & Hold Return", f"{bh_ret*100:.2f}%")
            c2.metric("SMA Strategy Return", f"{st_ret*100:.2f}%")

            st.markdown("#### Equity Curve (₹1 starting capital)")
            eq_df = pd.DataFrame(
                {"Buy & Hold": bh_curve, "SMA Strategy": st_curve},
                index=bt_df.index,
            )

            eq_chart = (
                alt.Chart(eq_df.reset_index())
                .mark_line()
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("value:Q", title="Growth (×)"),
                    color=alt.Color("variable:N", title="Strategy"),
                )
                .transform_fold(
                    ["Buy & Hold", "SMA Strategy"],
                    as_=["variable", "value"],
                )
                .properties(height=260)
            )
            st.altair_chart(eq_chart, use_container_width=True)

            st.caption(
                "Backtest transaction cost / slippage include nahi karta. "
                "Historical performance ≠ future results."
            )
else:
    st.info("ADITYA MARKET INTELLIGENCE DASHBOARD FOR STOCK ANALYSIS PLEASE PRESS RUN")
