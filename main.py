
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Find the project folder automatically.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "simulated_climate_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
IMAGE_DIR = PROJECT_DIR / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Load the dataset.
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    print("=" * 55)
    print("CLIMATE TREND ANALYZER - DATA EXPLORATION")
    print("=" * 55)

    # 2. Inspect the dataset.
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to "
          f"{df['date'].max().date()}")

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    # 3. Check data quality.
    print("\n--- DATA QUALITY CHECKS ---")
    print("\nMissing values per column:")
    print(df.isnull().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate dates: {df['date'].duplicated().sum()}")

    required_columns = [
        "date",
        "temperature_c",
        "rainfall_mm",
        "humidity_pct",
        "co2_ppm",
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    numeric_columns = [
        "temperature_c",
        "rainfall_mm",
        "humidity_pct",
        "co2_ppm",
    ]

    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values found. Clean the data before analysis.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found. Investigate before analysis.")

    if df[numeric_columns].isna().any().any():
        raise ValueError("Numeric climate columns contain missing values.")

    print("\nBasic validation passed!")

    # 4. Calculate descriptive statistics.
    print("\n--- DESCRIPTIVE STATISTICS ---")
    print(df[numeric_columns].describe().round(2))

    # 5. Summarize climate conditions by year.
    annual = (
        df.set_index("date")
        .resample("YS")
        .agg(
            temperature_c=("temperature_c", "mean"),
            rainfall_mm=("rainfall_mm", "sum"),
            humidity_pct=("humidity_pct", "mean"),
            co2_ppm=("co2_ppm", "mean"),
        )
        .reset_index()
    )

    annual["year"] = annual["date"].dt.year

    annual_path = OUTPUT_DIR / "annual_climate_summary.csv"
    annual.to_csv(annual_path, index=False)

    print(f"\nAnnual summary saved to: {annual_path}")
    print("\nFirst five annual summaries:")
    print(annual.head().round(2).to_string(index=False))

    # 6. Plot annual average temperature.
    years = annual["year"].to_numpy()
    temperatures = annual["temperature_c"].to_numpy()

    # Descriptive straight-line trend, not a climate prediction.
    slope, intercept = np.polyfit(years, temperatures, 1)
    trend_line = slope * years + intercept

    plt.figure(figsize=(12, 6))
    plt.plot(
        years,
        temperatures,
        marker="o",
        markersize=3,
        label="Annual average temperature",
    )
    plt.plot(
        years,
        trend_line,
        linestyle="--",
        label="Descriptive linear trend",
    )

    plt.title("Simulated Annual Average Temperature (1980–2025)")
    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    chart_path = IMAGE_DIR / "temperature_trend.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nTemperature chart saved to: {chart_path}")
    print(f"Simulated linear trend: {slope:.3f} °C per year")
    print("\nNote: this is simulated data, not real climate evidence.")


if __name__ == "__main__":
    main()
