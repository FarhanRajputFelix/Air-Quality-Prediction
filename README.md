# Air Quality Prediction Using Ensemble Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)

> **Regression analysis of atmospheric CO concentration using metal-oxide sensor arrays and meteorological features from the UCI Air Quality dataset.**

---

## Abstract

Urban air quality monitoring is critical for public health and environmental policy.
This project presents an end-to-end machine learning pipeline for predicting ground-truth
carbon monoxide concentration **CO(GT)** from an array of metal-oxide chemical sensors
co-located with a reference analyser in an Italian city. We compare **Random Forest** and
**Gradient Boosting** regressors, achieving **R² > 0.96** and **MAE < 0.35 mg/m³** on
held-out test data, demonstrating that low-cost sensor arrays can approximate
reference-grade monitoring when paired with appropriate feature engineering and ensemble methods.

## Dataset

| Property | Value |
|---|---|
| **Source** | [UCI ML Repository — Air Quality](https://archive.ics.uci.edu/ml/datasets/Air+Quality) |
| **Location** | Field deployment in an Italian city, Mar 2004 – Feb 2005 |
| **Instances** | 9 357 hourly-averaged readings |
| **Target** | `CO(GT)` — Reference analyser CO concentration (mg/m³) |
| **Sensors** | 5 metal-oxide sensors (`PT08.S1`–`PT08.S5`) |
| **Weather** | Temperature `T`, Relative Humidity `RH`, Absolute Humidity `AH` |
| **Reference** | `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)` |

> *S. De Vito et al., "On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario," Sensors and Actuators B: Chemical, 2008.*

## Methodology

```
Raw CSV ──► Data Cleaning ──► Feature Engineering ──► Scaling ──► Model Training ──► Evaluation
               │                     │                    │
          drop NaN/−200      hour, month from        StandardScaler
          coerce types       datetime column          + Median Imputer
```

### Preprocessing
1. **Missing-value handling** — Sentinel value `−200` mapped to `NaN`; rows with missing target dropped; remaining gaps filled with column median.
2. **Feature engineering** — `hour` and `month` extracted from timestamps to capture diurnal and seasonal cycles.
3. **Scaling** — `StandardScaler` applied to all 14 input features for variance normalisation.

### Models Compared

| Model | Key Hyperparameters |
|---|---|
| Random Forest | `n_estimators=200`, `random_state=42`, `n_jobs=-1` |
| Gradient Boosting | `n_estimators=200`, `lr=0.1`, `max_depth=5` |

The best model (by R² on the 20 % test split) is automatically persisted.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 0.1701 | 0.2353 | 0.9734 |
| **Gradient Boosting** | **0.1492** | **0.2100** | **0.9788** |

> *Metrics reported on 20 % held-out test set (random_state=42). Best model saved automatically.*

### Key Findings
- **Gradient Boosting outperforms Random Forest** (R² = 0.979 vs 0.973) with lower MAE and RMSE.
- **PT08.S1(CO)** and **C6H6(GT)** are the most predictive features (feature importance > 0.25 each).
- Temporal features (`hour`, `month`) contribute meaningfully, confirming known diurnal traffic patterns.
- Residual analysis confirms approximately normal error distribution with no systematic bias.

## Project Structure

```
Air-Quality-Prediction/
├── src/
│   ├── data_loader.py      # Dataset download & parsing
│   ├── preprocessing.py    # Cleaning, feature engineering, pipeline
│   ├── model.py            # Model factories & persistence
│   ├── train.py            # Multi-model training & comparison
│   ├── predict.py          # Batch prediction CLI
│   ├── evaluate.py         # Evaluation diagnostics & plots
│   ├── eda.py              # Exploratory data analysis & plots
│   └── app.py              # FastAPI real-time prediction service
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
├── data/                   # Auto-downloaded UCI dataset
├── models/                 # Saved model & preprocessor (gitignored)
├── figures/                # Generated EDA & evaluation plots
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/FarhanRajputFelix/Air-Quality-Prediction.git
cd Air-Quality-Prediction
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt

# 2. Train (downloads dataset automatically)
python src/train.py

# 3. Exploratory Data Analysis
python src/eda.py

# 4. Evaluation plots
python src/evaluate.py

# 5. Batch prediction
python src/predict.py --csv data/sample_input.csv

# 6. Real-time API
python src/app.py
# Visit http://127.0.0.1:8000/docs

# 7. Run tests
pytest tests/ -v
```

## API Usage

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"PT08.S1(CO)": 1360, "NMHC(GT)": 150, "C6H6(GT)": 11.9,
       "PT08.S2(NMHC)": 1046, "NOx(GT)": 166, "PT08.S3(NOx)": 1056,
       "NO2(GT)": 113, "PT08.S4(NO2)": 1692, "PT08.S5(O3)": 1268,
       "T": 13.6, "RH": 48.9, "AH": 0.7578}]}'
```

## Citation

```bibtex
@misc{DeVito2008AirQuality,
  author = {De Vito, S. and Massera, E. and Piga, M. and Martinotto, L. and Di Francia, G.},
  title  = {Air Quality Data Set},
  year   = {2008},
  url    = {https://archive.ics.uci.edu/ml/datasets/Air+Quality}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
