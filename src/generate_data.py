
from pathlib import Path

import numpy as np
import pandas as pd


def generate_climate_data():
    """Generate simulated monthly climate data for project development."""

    # Fixed seed makes results reproducible.
    rng = np.random.default_rng(42)

    # One observation per month from January 1980 to December 2025.
    dates = pd.date_range(
        start="1980-01-01",
        end="2025-12-01",
        freq="MS",
    )

    month = dates.month.to_numpy()
    years_since_start = (
        (dates.year - dates.year.min())
        + (month - 1) / 12
    )

    # Simulated seasonal temperature cycle plus a small artificial trend.
    seasonal_temperature = 25 + 5 * np.sin(
        2 * np.pi * (month - 4) / 12
    )
    temperature = (
        seasonal_temperature
        + 0.025 * years_since_start
        + rng.normal(0, 0.8, len(dates))
    )

    # Simulated rainfall with seasonal variation.
    seasonal_rainfall = 70 + 55 * np.sin(
        2 * np.pi * (month - 6) / 12
    )
    rainfall = np.maximum(
        0,
        seasonal_rainfall + rng.normal(0, 20, len(dates)),
    )

    # Simulated humidity, constrained to a plausible percentage range.
    humidity = np.clip(
        65 + 12 * np.sin(2 * np.pi * (month - 6) / 12)
        + rng.normal(0, 5, len(dates)),
        20,
        100,
    )

    # Illustrative CO2 values, not measured atmospheric observations.
    co2 = (
        338
        + 1.8 * years_since_start
        + 2 * np.sin(2 * np.pi * month / 12)
        + rng.normal(0, 0.7, len(dates))
    )

    df = pd.DataFrame(
        {
            "date": dates,
            "temperature_c": temperature.round(2),
            "rainfall_mm": rainfall.round(2),
            "humidity_pct": humidity.round(2),
            "co2_ppm": co2.round(2),
            "data_source": "SIMULATED",
        }
    )

    # Add a few deliberately unusual temperatures for anomaly testing.
    anomaly_indices = [120, 250, 400]
    df.loc[anomaly_indices, "temperature_c"] += [5, -5, 6]

    output_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "raw"
        / "simulated_climate_data.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Simulated climate dataset created successfully!")
    print(f"Saved to: {output_path}")
    print(f"Number of records: {len(df)}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nColumn information:")
    print(df.dtypes)


if __name__ == "__main__":
    generate_climate_data()
