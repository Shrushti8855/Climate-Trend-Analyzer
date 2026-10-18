
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "simulated_climate_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
IMAGE_DIR = PROJECT_DIR / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load and sort climate observations
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Create calendar and historical features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["temperature_lag_1"] = df["temperature_c"].shift(1)
    df["temperature_lag_2"] = df["temperature_c"].shift(2)
    df["temperature_lag_3"] = df["temperature_c"].shift(3)
    df["target_temperature"] = df["temperature_c"].shift(-1)

    df = df.dropna().reset_index(drop=True)

    feature_columns = [
        "year",
        "month",
        "temperature_c",
        "temperature_lag_1",
        "temperature_lag_2",
        "temperature_lag_3",
        "rainfall_mm",
        "humidity_pct",
    ]

    X = df[feature_columns]
    y = df["target_temperature"]

    # Use only the first 80% of records for training
    split_index = int(len(df) * 0.80)
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Extract feature importance
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    print("\n--- FEATURE IMPORTANCE ---")
    print(importance_df.round(4).to_string(index=False))

    # Plot feature importance
    plot_df = importance_df.sort_values("importance")

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Climate Temperature Forecast: Feature Importance")
    plt.tight_layout()
    plt.savefig(
        IMAGE_DIR / "feature_importance.png",
        dpi=150,
    )
    plt.close()

    print("\nFeature importance analysis completed!")
    print(f"CSV saved to: {OUTPUT_DIR / 'feature_importance.csv'}")
    print(f"Chart saved to: {IMAGE_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
