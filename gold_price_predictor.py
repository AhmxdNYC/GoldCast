"""
GoldCast: A Machine Learning Application for Gold Price Prediction
==================================================================

This application predicts the next-day closing price of gold (XAU/USD,
COMEX Gold Futures ticker "GC=F") using a Random Forest Regressor trained
on historical price data and engineered technical-indicator features.

Author: Ahmad
Course: Machine Learning (Study.com / C949-adjacent)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1. DATA ACQUISITION
# ----------------------------------------------------------------------
def load_data(ticker="GC=F", start="2015-01-01", end="2025-06-01",
              csv_fallback="gold_data.csv"):
    """
    Download historical daily gold prices from Yahoo Finance.
    Falls back to a local CSV if the network is unavailable.
    """
    try:
        import yfinance as yf
        print(f"[INFO] Downloading {ticker} data from Yahoo Finance...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty:
            raise ValueError("Empty dataframe returned from yfinance.")
        # Flatten possible MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.to_csv(csv_fallback)
        print(f"[INFO] Downloaded {len(df)} records. Cached to {csv_fallback}.")
    except Exception as e:
        print(f"[WARN] yfinance download failed ({e}).")
        if os.path.exists(csv_fallback):
            print(f"[INFO] Loading cached data from {csv_fallback}.")
            df = pd.read_csv(csv_fallback, index_col=0, parse_dates=True)
        else:
            print("[ERROR] No data source available. Exiting.")
            sys.exit(1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING / PREPROCESSING
# ----------------------------------------------------------------------
def engineer_features(df):
    """
    Create technical-indicator features and the prediction target.
    Target = next day's closing price.
    """
    data = df.copy()

    # Lagged closing prices
    for lag in [1, 2, 3, 5, 10]:
        data[f"lag_{lag}"] = data["Close"].shift(lag)

    # Moving averages
    data["ma_5"] = data["Close"].rolling(window=5).mean()
    data["ma_10"] = data["Close"].rolling(window=10).mean()
    data["ma_20"] = data["Close"].rolling(window=20).mean()

    # Rolling volatility (std of returns)
    data["return_1d"] = data["Close"].pct_change()
    data["volatility_10"] = data["return_1d"].rolling(window=10).std()

    # High-Low daily range
    data["hl_range"] = data["High"] - data["Low"]

    # Relative Strength Index (RSI, 14-day)
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    # Target: next-day percentage return.
    # We predict the RETURN rather than the raw price level because tree-based
    # models cannot extrapolate beyond the value range seen in training. The
    # return series is approximately stationary, which is the standard, correct
    # framing for forecasting trending financial time series. The predicted
    # next-day price is then reconstructed as: today_close * (1 + predicted_return).
    data["target"] = data["Close"].pct_change().shift(-1)

    # Drop rows with NaN created by shifting/rolling
    data = data.dropna()
    return data


# ----------------------------------------------------------------------
# 3. MODEL TRAINING
# ----------------------------------------------------------------------
def train_model(data):
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
        "ma_5", "ma_10", "ma_20",
        "return_1d", "volatility_10", "hl_range", "rsi_14",
    ]
    X = data[feature_cols]
    y = data["target"]

    # Chronological split (no shuffle — time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    pred_returns = model.predict(X_test_s)

    # Reconstruct predicted PRICE from predicted return:
    # predicted_close[t+1] = close[t] * (1 + predicted_return[t])
    today_close = X_test["Close"].values
    pred_prices = today_close * (1 + pred_returns)
    actual_prices = today_close * (1 + y_test.values)

    actual_prices = pd.Series(actual_prices, index=y_test.index)

    # Directional accuracy: did we predict the sign of the move correctly?
    direction_correct = np.mean(
        np.sign(pred_returns) == np.sign(y_test.values)
    )

    metrics = {
        "MAE": mean_absolute_error(actual_prices, pred_prices),
        "RMSE": np.sqrt(mean_squared_error(actual_prices, pred_prices)),
        "R2": r2_score(actual_prices, pred_prices),
        "DirAcc": direction_correct,
    }
    return (model, scaler, feature_cols, X_test, actual_prices,
            pred_prices, metrics)


# ----------------------------------------------------------------------
# 4. VISUALIZATION
# ----------------------------------------------------------------------
def make_plots(y_test, preds, model, feature_cols, outdir="."):
    sns.set_style("whitegrid")

    # Actual vs Predicted
    plt.figure(figsize=(11, 5))
    plt.plot(y_test.index, y_test.values, label="Actual", linewidth=1.6)
    plt.plot(y_test.index, preds, label="Predicted", linewidth=1.6, alpha=0.8)
    plt.title("GoldCast: Actual vs. Predicted Gold Closing Price (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD / troy oz)")
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(outdir, "actual_vs_predicted.png")
    plt.savefig(p1, dpi=130)
    plt.close()

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)
    plt.figure(figsize=(9, 6))
    importances.plot(kind="barh", color="#C9A227")
    plt.title("GoldCast: Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    p2 = os.path.join(outdir, "feature_importance.png")
    plt.savefig(p2, dpi=130)
    plt.close()

    return p1, p2


# ----------------------------------------------------------------------
# 5. NEXT-DAY PREDICTION
# ----------------------------------------------------------------------
def predict_next_day(model, scaler, feature_cols, data):
    latest = data[feature_cols].iloc[[-1]]
    latest_s = scaler.transform(latest)
    pred_return = model.predict(latest_s)[0]
    last_close = data["Close"].iloc[-1]
    pred = last_close * (1 + pred_return)
    return last_close, pred


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GoldCast price predictor")
    parser.add_argument("--ticker", default="GC=F")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    print("=" * 60)
    print("  GoldCast — Gold Price Prediction (Random Forest)")
    print("=" * 60)

    df = load_data(args.ticker, args.start, args.end)
    data = engineer_features(df)
    print(f"[INFO] Feature matrix: {data.shape[0]} rows, "
          f"{data.shape[1]} columns (incl. target).")

    model, scaler, feature_cols, X_test, y_test, preds, metrics = train_model(data)

    print("\n----- MODEL PERFORMANCE (Test Set) -----")
    print(f"  MAE  : ${metrics['MAE']:.2f}")
    print(f"  RMSE : ${metrics['RMSE']:.2f}")
    print(f"  R^2  : {metrics['R2']:.4f}")
    print(f"  Directional Accuracy : {metrics['DirAcc']*100:.1f}%")

    p1, p2 = make_plots(y_test, preds, model, feature_cols, args.outdir)
    print(f"\n[INFO] Saved plot: {p1}")
    print(f"[INFO] Saved plot: {p2}")

    last_close, next_pred = predict_next_day(model, scaler, feature_cols, data)
    direction = "UP" if next_pred > last_close else "DOWN"
    print("\n----- NEXT-DAY PREDICTION -----")
    print(f"  Last known close : ${last_close:.2f}")
    print(f"  Predicted next   : ${next_pred:.2f}  ({direction})")
    print("=" * 60)


if __name__ == "__main__":
    main()
