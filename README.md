# GoldCast — Gold Price Prediction

A machine learning application that predicts the next-day closing price of gold
(COMEX Gold Futures, GC=F) using a Random Forest Regressor trained on engineered
technical-indicator features.

## Files
- `gold_price_predictor.py` — main application source code
- `gold_data.csv` — cached historical price data (offline fallback)
- `GoldCast_Report.docx` — full assignment report
- `AI_Attribution.docx` — required AI-usage documentation
- `actual_vs_predicted.png`, `feature_importance.png` — output plots

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python gold_price_predictor.py
```

Optional arguments:
```
python gold_price_predictor.py --ticker GC=F --start 2015-01-01 --end 2025-06-01
```

The app downloads live data from Yahoo Finance when online and automatically
falls back to the bundled `gold_data.csv` when offline. It prints model metrics
and a next-day prediction, and saves two plots to the working directory.

## Results (test set)
- MAE: $43.42
- RMSE: $62.98
- R²: 0.9950
- Directional accuracy: 41.7%
