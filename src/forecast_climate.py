
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "simulated_climate_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
IMAGE_DIR = PROJECT_DIR / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    return mae, rmse


def main():
    # 1. Load and sort the climate data
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 2. Create features and next-month temperature target
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df["temperature_lag_1"] = df["temperature_c"].shift(1)
    df["temperature_lag_2"] = df["temperature_c"].shift(2)
    df["temperature_lag_3"] = df["temperature_c"].shift(3)

    # Predict the next observation's temperature
    df["target_temperature"] = df["temperature_c"].shift(-1)

    # Seasonal-naive prediction: temperature from 12 months earlier
    df["seasonal_prediction"] = df["temperature_c"].shift(11)

    df = df.dropna().reset_index(drop=True)

    features = [
        "year",
        "month",
        "temperature_c",
        "temperature_lag_1",
        "temperature_lag_2",
        "temperature_lag_3",
        "rainfall_mm",
        "humidity_pct",
    ]

    X = df[features]
    y = df["target_temperature"]

    # 3. Chronological 80/20 train-test split
    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    seasonal_predictions = df["seasonal_prediction"].iloc[
        split_index:
    ].to_numpy()

    # 4. Mean baseline
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    # 5. Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    rf_predictions = model.predict(X_test)

    # 6. Evaluate all three approaches
    results = []

    for name, predictions in [
        ("Mean baseline", baseline_predictions),
        ("Seasonal-naive baseline", seasonal_predictions),
        ("Random Forest", rf_predictions),
    ]:
        mae, rmse = calculate_metrics(y_test, predictions)
        results.append(
            {"model": name, "MAE": mae, "RMSE": rmse}
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "forecast_model_comparison.csv",
        index=False,
    )

    # 7. Save held-out predictions
    predictions_df = pd.DataFrame(
        {
            "date": df["date"].iloc[split_index:].values,
            "actual_temperature_c": y_test.to_numpy(),
            "mean_baseline_prediction": baseline_predictions,
            "seasonal_naive_prediction": seasonal_predictions,
            "random_forest_prediction": rf_predictions,
        }
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "climate_forecast_predictions.csv",
        index=False,
    )

    # 8. Print results
    print("\n--- DATA SPLIT ---")
    print(f"Total usable records: {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    print("\n--- MODEL COMPARISON (LOWER IS BETTER) ---")
    print(results_df.round(3).to_string(index=False))

    print("\n--- FIRST 10 TEST PREDICTIONS ---")
    print(
        predictions_df.head(10).to_string(
            index=False,
            formatters={
                "actual_temperature_c": "{:.2f}".format,
                "mean_baseline_prediction": "{:.2f}".format,
                "seasonal_naive_prediction": "{:.2f}".format,
                "random_forest_prediction": "{:.2f}".format,
            },
        )
    )

    # 9. Plot actual temperatures versus model predictions
    plt.figure(figsize=(13, 6))
    plt.plot(
        predictions_df["date"],
        predictions_df["actual_temperature_c"],
        label="Actual",
        linewidth=2,
    )
    plt.plot(
        predictions_df["date"],
        predictions_df["mean_baseline_prediction"],
        label="Mean baseline",
        alpha=0.8,
    )
    plt.plot(
        predictions_df["date"],
        predictions_df["seasonal_naive_prediction"],
        label="Seasonal-naive",
        alpha=0.8,
    )
    plt.plot(
        predictions_df["date"],
        predictions_df["random_forest_prediction"],
        label="Random Forest",
        alpha=0.9,
    )

    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Climate Forecast: Actual vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        IMAGE_DIR / "climate_forecast_comparison.png",
        dpi=150,
    )
    plt.close()

    # 10. Plot model error comparison
    results_df.set_index("model")[["MAE", "RMSE"]].plot(
        kind="bar",
        figsize=(10, 5),
    )
    plt.ylabel("Error (°C)")
    plt.title("Forecast Model Error Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(
        IMAGE_DIR / "forecast_model_metrics.png",
        dpi=150,
    )
    plt.close()

    print("\nForecasting analysis completed successfully!")
    print(f"CSV outputs: {OUTPUT_DIR}")
    print(f"Charts saved to: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
