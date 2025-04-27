# Urmia Lake Water Storage (LWE) Prediction using Climate Variables

This repository contains the code and sample data for predicting Lake Water Equivalent (LWE) changes in Urmia Lake using satellite-derived and climatic features.  
We applied machine learning models, feature engineering, and interpretability techniques to improve forecasting accuracy and extract insights into the drivers of lake water storage variation.

---

## Project Description

The study investigates how climatic indicators such as soil moisture, precipitation, temperature, and sea level pressure influence LWE changes over time.  
Key contributions of this work include:

- Preprocessing of climatic data (handling missing values, lag features, cyclic month encoding).
- Feature weighting using XGBoost feature importance.
- Application of Ridge Regression and Random Forest with cross-validation and hyperparameter tuning.
- Model interpretability via SHAP (SHapley Additive exPlanations) analysis.

---

## Repository Structure

```bash
├── data/
│   └── Urmia_Lake_Climate_Data_month.csv  # Example data file
├── code/
│   └── LWE_prediction.py                  # Main Python script
├── figures/
│   └── shap_summary_plot.png               # Example SHAP visualization
├── README.md                               # This file
└── LICENSE                                 # (optional)
