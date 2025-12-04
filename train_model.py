import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Features same rakhenge, taaki app ke saath compatible rahe
FEATURE_COLS = ["Open", "High", "Low", "Volume", "Return1", "Return5", "VolChange"]


def main():
    # 1. Data load
    df = pd.read_csv("reliance_daily.csv")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

    # 2. Numeric columns ensure karo
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"Column '{col}' CSV me nahi mila.")

    df = df.dropna(subset=numeric_cols)

    # 3. Features banao
    df["Return1"] = df["Close"].pct_change(1)
    df["Return5"] = df["Close"].pct_change(5)
    df["VolChange"] = df["Volume"].pct_change(1)

    # 4. Target: NEXT-DAY RETURN (price nahi, % change)
    # aaj ka close -> kal ka close kitna % change
    df["ReturnTarget"] = df["Close"].pct_change(1).shift(-1)  # next day ka return

    # 5. Clean NaN / inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + ["ReturnTarget"])

    X = df[FEATURE_COLS]
    y = df["ReturnTarget"]  # yaha ab percentage change hai, e.g. 0.015 = 1.5%

    print("Total samples:", len(X))

    # 6. Train / test split (80 / 20), time order respect karte hue
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))

    # 7. Model define – thoda tuned params
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("Training model on returns...")
    model.fit(X_train, y_train)

    # 8. Evaluation on test
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5  # RMSE of returns
    print(f"Test RMSE on returns: {rmse:.4f}")

    # Approx rupee error idea (rough)
    last_price = float(df["Close"].iloc[-60:].mean())
    approx_rupee_rmse = rmse * last_price
    print(f"Approx rupee RMSE: {approx_rupee_rmse:.2f} INR")

    # 9. Final model ko full data par dubara train karo
    print("Training final model on FULL data...")
    model.fit(X, y)

    # 10. Save model – app isi file ko use karega
    with open("reliance_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Return-based model saved as reliance_model.pkl")


if __name__ == "__main__":
    main()
