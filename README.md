# Global Air Quality Mapping: An Advanced Applied ML Project 🌍💨

An end-to-end machine learning pipeline analyzing global atmospheric trends based on the **IQAir 2025 World Air Quality Report** (released March 2026). This project focuses on high-fidelity data engineering and predictive modeling to estimate regional environmental risks.

---

## 🏗️ Technical Architecture
This system is an **Advanced Applied Machine Learning** implementation designed for high-precision environmental analytics.

1.  **Iterative Modeling:** Built using a comparative framework between **Random Forest** and **Gradient Boosting** ensembles.
2.  **Performance Baseline:** The system achieves **strong predictive performance on held-out data** (Random Forest R² ≈ 0.97), effectively learning the regional structure of global pollution data.
3.  **Modern Data Stack:** Processes 2025-2026 reporting cycles to provide current insights into the global disparity between South Asian hotspots and pristine regions.

## 📁 Project Structure

```bash
├── data/                    # 2025-2026 Global datasets
├── figures/                 # Comparison plots & error distributions
├── src/
│   ├── global_analysis.py   # Regression baseline & Error diagnostics
│   ├── train_global.py      # Ensemble training pipeline
│   ├── predict_global.py    # Multi-pollutant inference engine
│   └── model.py             # Reusable model architectures
├── tests/                   # Reliability & logic validation
└── README.md                # Technical documentation
```

## 📊 2025 Key Observations
*   **Most Polluted:** Pakistan (67.3 µg/m³) and Loni, India (112.5 µg/m³).
*   **Cleanest:** French Polynesia (1.8 µg/m³) and Nieuwoudtville, South Africa (1.0 µg/m³).
*   **Guideline Status:** Only 9% of monitored countries met the WHO annual PM2.5 threshold.

## ⚠️ System Limitations & Generalization
As an applied ML project, it is critical to note the following:
*   **Correlation vs. Causality:** High R² scores reflect the model's ability to learn the internal structure of the IQAir report rankings.
*   **Uncertainty:** Predictions are most robust in regions with dense monitoring networks (e.g., South Asia, Europe, North America).
*   **Temporal Scope:** Findings are specialized to the 2025 reporting cycle.

---

## 🚀 Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate
pip install -r requirements.txt
python src/train_global.py   # Execute training
python src/global_analysis.py # Run diagnostics
```

## 📜 License
Licensed under the **MIT License**.
