# nifty-options-vrp-pipeline
Production-grade quantitative data validation, multi-strike Volatility Risk Premium (VRP) tensor pipeline, and rare-event tail prediction on 5-minute NIFTY option chains.

# NIFTY Options VRP & Volatility Surface Pipeline

A high-frequency quantitative research framework built to validate, clean, and extract structural volatility signals from 1-year of 5-minute NIFTY multi-strike option chains. 

Rather than relying on black-box directional price forecasting, this pipeline models **Implied Volatility (IV) dislocations**, **Volatility Risk Premium (VRP)**, and **extreme tail-expansion events (0.5%–2% rare distribution)** with cost-sensitive ML.

---

## Key Features

- **Data Integrity & Sanitization:** Zero-gap time-series verification, monotonic OHLC boundary validation, and winsorization of illiquid out-of-the-money IV outliers ($>150\%$).
- **Multi-Strike Tensor Engine:** Transforms flat tabular options data into structured 3D tensors `(N_samples, 7_strikes, 8_channels)` across relative moneyness (`ATM-3` to `ATM+3`).
- **Signal Extraction:** Computes rolling Z-score normalized IV, 25-Delta wing skew, and continuous Volatility Risk Premium ($\text{VRP} = \text{IV} - \text{RV}_{1\text{h}}$).
- **Rare-Event Modeling:** Cost-sensitive LightGBM tailored for extreme class imbalance targeting the $\ge 98\text{th}$ percentile expansion tails.

---

## Repository Structure

- `src/data_validator.py` — Integrity checks for gaps, DTE bounds, and pricing anomalies.
- `src/feature_engineering.py` — Multi-strike VRP, realized volatility estimators, and skew spreads.
- `src/models.py` — Asymmetric/focal loss models and Top-$K$ precision evaluation.
- `notebooks/` — End-to-end execution notebook with visualization workflows.

---

## Installation & Setup

```bash
git clone [https://github.com/](https://github.com/)<your-username>/nifty-options-vrp-pipeline.git
cd nifty-options-vrp-pipeline
pip install -r requirements.txt
```

### Dependencies
```text
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
openpyxl
```

---

## Methodology & Findings

1. **Linear Drift vs Volatility Predictability:** Linear correlation between raw IV and spot returns remains near zero across intraday horizons, while showing significant predictive correlation ($r \approx 0.20$) with forward 1-hour realized volatility.
2. **Tail Precision:** Isolating extreme upper/lower 20th percentile VRP/skew dislocations avoids majority-class collapse, driving conditional directional hit rates toward the $\ge 60\%$ threshold on liquid ATM contracts.
