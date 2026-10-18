
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "simulated_climate_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
IMAGE_DIR = PROJECT_DIR / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Monthly seasonality
    df["month"] = df["date"].dt.month

    monthly = (
        df.groupby("month")
        .agg(
            avg_temperature_c=("temperature_c", "mean"),
            avg_rainfall_mm=("rainfall_mm", "mean"),
            avg_humidity_pct=("humidity_pct", "mean"),
        )
        .reset_index()
    )

    monthly.to_csv(
        OUTPUT_DIR / "monthly_seasonality.csv", index=False
    )

    print("\n--- MONTHLY SEASONALITY ---")
    print(monthly.round(2).to_string(index=False))

    # Detect temperature anomalies within each calendar month
    df["month_avg_temp"] = df.groupby("month")[
        "temperature_c"
    ].transform("mean")

    df["month_std_temp"] = df.groupby("month")[
        "temperature_c"
    ].transform("std")

    df["temperature_z_score"] = (
        (df["temperature_c"] - df["month_avg_temp"])
        / df["month_std_temp"].replace(0, float("nan"))
    )

    anomalies = df[df["temperature_z_score"].abs() >= 2.5].copy()

    anomalies.to_csv(
        OUTPUT_DIR / "temperature_anomalies.csv", index=False
    )

    print("\n--- TEMPERATURE ANOMALIES ---")
    print(f"Total observations: {len(df)}")
    print(f"Anomalies detected: {len(anomalies)}")

    if not anomalies.empty:
        print(
    anomalies[
        ["date", "temperature_c", "temperature_z_score"]
    ].to_string(
        index=False,
        formatters={
            "temperature_c": "{:.2f}".format,
            "temperature_z_score": "{:.2f}".format,
        },
    )
)
    # Plot monthly temperature seasonality
    plt.figure(figsize=(10, 5))
    plt.plot(
        monthly["month"],
        monthly["avg_temperature_c"],
        marker="o",
    )
    plt.xticks(range(1, 13))
    plt.xlabel("Month")
    plt.ylabel("Average temperature (°C)")
    plt.title("Monthly Temperature Seasonality")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "monthly_seasonality.png", dpi=150)
    plt.close()

    # Plot temperature observations and highlight anomalies
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["temperature_c"], label="Temperature")

    if not anomalies.empty:
        plt.scatter(
            anomalies["date"],
            anomalies["temperature_c"],
            label="Detected anomalies",
            marker="x",
        )

    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "temperature_anomalies.png", dpi=150)
    plt.close()

    print("\nAnalysis complete!")
    print(f"CSV outputs: {OUTPUT_DIR}")
    print(f"Charts saved to: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
