import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle

FEATURE_COLS = ["Open", "High", "Low", "Volume", "Return1", "Return5", "VolChange"]

# per-day max move kitna allow karna hai (±)
MAX_DAILY_MOVE = 0.06  # 6% up ya down se zyada nahi


@st.cache_data
def load_history():
    """reliance_daily.csv file load karta hai."""
    df = pd.read_csv("reliance_daily.csv")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    return df


def load_model():
    """Trained RandomForest model load karta hai (jo return predict karta hai)."""
    with open("reliance_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def download_fresh_data():
    """YFinance se fresh data download karke CSV update karta hai."""
    ticker = "RELIANCE.NS"
    data = yf.download(ticker, start="2015-01-01")
    data.to_csv("reliance_daily.csv")
    return data


def make_features(df):
    """Training wale jaisa hi feature engineering."""
    df = df.copy()

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    df["Return1"] = df["Close"].pct_change(1)
    df["Return5"] = df["Close"].pct_change(5)
    df["VolChange"] = df["Volume"].pct_change(1)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS)

    return df


# ----------------- STREAMLIT UI -----------------

st.title("📈 Simple Stock Advisor – RELIANCE.NS (Return-based)")
st.write(
    "Yeh app next-day **percentage change (return)** predict karta hai, "
    "phir usse approximate next-day close price nikalta hai.\n\n"
    "**Note:** Ye sirf learning/demo ke liye hai, financial advice nahi hai."
)

st.markdown("---")

# ✅ Section 1: Data dekhna
st.header("1️⃣ Historical data & chart")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Fresh data download karo (YFinance se)"):
        with st.spinner("Downloading data from Yahoo Finance..."):
            df = download_fresh_data()
        st.success("Fresh data download ho gaya & reliance_daily.csv update ho gaya!")
        st.write(df.tail())

with col2:
    if st.button("📂 Existing CSV se data load karo"):
        try:
            df = load_history()
            st.success("reliance_daily.csv load ho gaya!")
            st.write(df.tail())
        except FileNotFoundError:
            st.error("reliance_daily.csv nahi mila. Pehle train_model.py run karo.")

# Chart
try:
    df_hist = load_history()
    if "Date" in df_hist.columns and "Close" in df_hist.columns:
        st.subheader("Closing price chart")
        st.line_chart(df_hist.set_index("Date")["Close"])
except Exception:
    pass

st.markdown("---")

# ✅ Section 2: Prediction
st.header("2️⃣ Next-day close price prediction (realistic range)")

if st.button("🔮 Predict next-day close"):
    try:
        # 1. Model + data load
        model = load_model()
        df = load_history()
        df_feat = make_features(df)

        if len(df_feat) == 0:
            st.error("Features banne ke baad koi row nahi bachi. CSV check karo.")
        else:
            last_row = df_feat.iloc[-1:][FEATURE_COLS]
            last_close = float(df_feat["Close"].iloc[-1])

            # 2. Model se NEXT-DAY RETURN predict
            pred_return = float(model.predict(last_row)[0])  # e.g. 0.02 = +2%

            # 3. Prediction ko realistic range me clip karo
            clipped_return = max(min(pred_return, MAX_DAILY_MOVE), -MAX_DAILY_MOVE)

            # 4. Next-day price estimate
            predicted_price = last_close * (1 + clipped_return)

            change = predicted_price - last_close
            pct = clipped_return * 100.0

            st.success(f"Predicted next-day close: **₹{predicted_price:.2f}**")
            st.write(f"Last close: ₹{last_close:.2f}")
            st.write(f"Expected move: **{pct:+.2f}%**  (₹{change:+.2f})")

            if abs(pred_return) > MAX_DAILY_MOVE:
                st.caption(
                    "Raw model bahut bada move predict kar raha tha, "
                    "isliye usse realistic range (±6%) me clip kiya gaya hai."
                )

    except FileNotFoundError:
        st.error("Model ya CSV nahi mila. Pehle train_model.py successfully run karo.")
    except Exception as e:
        st.error(f"Kuch error aa gaya: {e}")

st.markdown("---")
st.caption("Demo project – learning purpose only, financial advice nahi hai 🙂")
