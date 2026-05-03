# Global Air Quality Analysis (2025-2026) 🌍💨

A research-grade analysis of global air quality trends based on the **IQAir 2025 World Air Quality Report** (released March 2026). This project provides a comparative analysis of the most polluted and cleanest regions globally, utilizing modern data science and ensemble learning techniques.

---

## 📊 Core Research Objectives
1. **Global Comparison:** Evaluate PM2.5 concentrations across 143 countries and thousands of cities.
2. **"Worst vs Best" Analysis:** Visualize the massive disparity between highly polluted regions (e.g., South Asia) and regions meeting WHO guidelines.
3. **Environmental Standards Assessment:** Quantifying the percentage of cities meeting the WHO annual PM2.5 guideline (5 µg/m³).

## 📁 Project Structure

```bash
├── data/                    # 2025 Global datasets (CSV)
├── figures/                 # Comparison plots & diagnostics
├── src/
│   ├── global_analysis.py   # Primary 2025 comparison script
│   ├── model.py             # Ensemble model definitions
│   ├── preprocessing.py     # Data cleaning & feature engineering
│   └── train.py             # Model training pipeline
├── tests/                   # Performance & logic validation
└── README.md                # Project documentation
```

## 📈 2025 Global Highlights

| Category | Top Polluted | Cleanest |
|---|---|---|
| **Country** | Pakistan (67.3 µg/m³) | French Polynesia (1.8 µg/m³) |
| **City** | Loni, India (112.5 µg/m³) | Nieuwoudtville, South Africa (1.0 µg/m³) |

### Key Findings (2025 Report):
- **WHO Guidelines:** Only **13 countries** (9%) successfully met the WHO annual PM2.5 guideline.
- **Regional Hotspots:** South Asia (Pakistan, India, Bangladesh) remains the most polluted region, with all Top 25 most polluted cities located there.
- **Data Gaps:** The 2025 report identified significant monitoring gaps in Africa and Central Asia.
- **Global Trend:** Only **14% of cities** met the WHO air quality standard in 2025, a decrease from 17% in the previous year.

---

## 🚀 Quick Start

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run Global Analysis
```bash
python src/global_analysis.py
```

## 📜 License
This project is licensed under the **MIT License**.

## 📝 Citation
Please cite this work as:
> "Global Air Quality Prediction and Comparative Analysis 2025", Farhan Rajput, 2026.
