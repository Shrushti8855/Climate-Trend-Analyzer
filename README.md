# 🌍 ClimateIQ — Climate Trend Analyzer

An interactive Data Science dashboard for exploring historical climate patterns, identifying unusual temperature observations, and evaluating machine-learning models for climate forecasting.

> **Important:** This project uses simulated climate data for learning and demonstration. Its results should not be interpreted as real-world climate research or official climate forecasts.

## 📌 Project Overview

Climate data contains patterns and variations that can be explored using data analysis and machine learning. This project demonstrates a complete workflow, from generating a dataset and analyzing trends to evaluating predictive models and presenting results through an interactive dashboard.

## ✨ Key Features

- **Climate Trend Analysis:** Explore temperature, rainfall, humidity, and CO₂ patterns over time.
- **Seasonality Analysis:** Compare monthly climate patterns and identify seasonal variations.
- **Anomaly Detection:** Flag unusually high or low temperatures relative to typical values for the same calendar month.
- **Climate Forecasting:** Compare baseline forecasting approaches with a Random Forest regression model.
- **Model Evaluation:** Evaluate predictions using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Feature Importance:** Identify which input features contribute most to the trained Random Forest model.
- **Interactive Dashboard:** Explore visualizations and climate records through a multi-page Streamlit application.
- **Data Export:** Download data for further analysis.

## 🛠️ Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Plotly
- Streamlit
- CSV data processing
- Machine Learning and Exploratory Data Analysis

## 📊 Dataset

The project generates a simulated monthly climate dataset containing **552 observations** covering January 1980 to December 2025.

| Feature | Description |
|---|---|
| `date` | Monthly observation date |
| `temperature_c` | Temperature in degrees Celsius |
| `rainfall_mm` | Rainfall in millimetres |
| `humidity_pct` | Relative humidity percentage |
| `co2_ppm` | Simulated CO₂ concentration in parts per million |
| `data_source` | Dataset source indicator |

The data is simulated and is not a substitute for measurements from a verified meteorological source.

## 🤖 Forecasting Model Results

The models were evaluated on a chronological held-out test set of 108 observations.

| Model | MAE | RMSE |
|---|---:|---:|
| Mean baseline | 3.145 | 3.558 |
| Seasonal-naive baseline | 0.967 | 1.190 |
| Random Forest Regressor | 0.806 | 0.998 |

**Interpretation:** On this simulated dataset and test split, the Random Forest model achieved the lowest MAE and RMSE among the evaluated models.

These metrics describe performance on this particular dataset; they do not establish real-world forecasting accuracy.

## 📈 Feature Importance

The trained Random Forest model identified these leading features:

1. Current temperature — approximately 61.7%
2. Month — approximately 18.0%
3. Temperature from three observations earlier — approximately 15.3%

Feature importance describes the model's use of input variables. It does not prove that a feature causes climate changes.

## 📂 Project Structure

```text
Climate-Trend-Analyzer/
├── app.py
├── main.py
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── generate_data.py
│   ├── analyze_patterns.py
│   ├── forecast_climate.py
│   └── analyze_feature_importance.py
├── outputs/
│   ├── monthly_seasonality.csv
│   ├── temperature_anomalies.csv
│   ├── forecast_model_comparison.csv
│   ├── climate_forecast_predictions.csv
│   └── feature_importance.csv
├── images/
├── reports/
├── docs/
└── README.md
```

*The structure above highlights the main project files and outputs; additional generated files may be present.*

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Climate-Trend-Analyzer
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create and activate a virtual environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install streamlit plotly pandas scikit-learn matplotlib
```

### 4. Launch the dashboard

```powershell
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to explore the dashboard.

**Note:** The dashboard reads the project's dataset and generated output files. Ensure the required files are present before launching it.

## 🎯 Skills Demonstrated

- Data generation and preprocessing
- Exploratory Data Analysis (EDA)
- Time-series feature engineering
- Seasonality analysis
- Anomaly detection
- Regression modelling
- Chronological train/test evaluation
- Model comparison and feature interpretation
- Interactive dashboard development
- Data visualization and communication

## 🔮 Future Improvements

- Integrate verified historical climate datasets.
- Add more forecasting models and stronger validation.
- Compare predictions across different locations.
- Add uncertainty intervals to forecasts.
- Deploy the dashboard online.

## 👩‍💻 Author

**Shrushti Shewale**

Final-year B.Tech student in Electronics & Telecommunication Engineering, exploring Data Science, Machine Learning, and AI applications.

---

*Built as a hands-on Data Science portfolio project.*