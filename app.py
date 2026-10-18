
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "simulated_climate_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Climate Intelligence | Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#10243A"
TEAL = "#13B8A6"
BLUE = "#5B8DEF"
BG = "#F4F7FB"
MUTED = "#66758A"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
    }}
    [data-testid="stSidebar"] {{
        background: {NAVY};
    }}
    [data-testid="stSidebar"] * {{
        color: #F4F7FB;
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }}
    .hero {{
        background: linear-gradient(120deg, {NAVY}, #1B4965);
        padding: 28px 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 20px;
    }}
    .hero h1 {{
        color: white;
        margin: 0;
        font-size: 2rem;
    }}
    .hero p {{
        color: #D4E4F1;
        margin: 8px 0 0 0;
    }}
    .kpi {{
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
        min-height: 120px;
        box-shadow: 0 2px 8px rgba(16,36,58,0.04);
    }}
    .kpi-label {{
        color: {MUTED};
        font-size: 0.85rem;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        color: {NAVY};
        font-size: 1.8rem;
        font-weight: 700;
    }}
    .kpi-note {{
        color: {MUTED};
        font-size: 0.78rem;
        margin-top: 6px;
    }}
    .section-note {{
        color: {MUTED};
        font-size: 0.9rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- DATA LOADING ----------------
@st.cache_data
def load_climate_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data
def load_output(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def kpi(label, value, note):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", color=NAVY),
        margin=dict(l=15, r=15, t=55, b=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        gridcolor="#E8EDF3",
        zeroline=False,
    )
    return fig


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Check that your simulated climate CSV exists in data/raw/.")
    st.stop()

df = load_climate_data()

required_columns = {
    "date",
    "temperature_c",
    "rainfall_mm",
    "humidity_pct",
}

if not required_columns.issubset(df.columns):
    st.error("The climate dataset is missing required columns.")
    st.stop()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🌍 ClimateIQ")
    st.caption("CLIMATE ANALYTICS PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Climate Explorer",
            "Anomaly Analysis",
            "Forecast Lab",
            "Model Insights",
            "Data Explorer",
        ],
    )

    st.divider()
    st.markdown("### Filters")

    start_date = st.date_input(
        "Start date",
        value=df["date"].min().date(),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date(),
    )

    end_date = st.date_input(
        "End date",
        value=df["date"].max().date(),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date(),
    )

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    st.caption("Dataset: simulated climate observations")
    st.caption(f"Available records: {len(df):,}")


filtered = df[
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
].copy()

if filtered.empty:
    st.warning("No records found for this date range.")
    st.stop()


# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <h1>Climate Intelligence Dashboard</h1>
        <p>Explore climate patterns, statistical anomalies,
        and machine learning forecast performance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Reporting period: {start_date:%d %b %Y} — "
    f"{end_date:%d %b %Y}  |  "
    f"{len(filtered):,} observations"
)


# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.subheader("Executive Overview")

    a, b, c, d = st.columns(4)

    with a:
        kpi(
            "AVG. TEMPERATURE",
            f"{filtered['temperature_c'].mean():.1f} °C",
            "Selected period",
        )
    with b:
        kpi(
            "TOTAL RAINFALL",
            f"{filtered['rainfall_mm'].sum():,.0f} mm",
            "Sum of recorded observations",
        )
    with c:
        kpi(
            "AVG. HUMIDITY",
            f"{filtered['humidity_pct'].mean():.1f}%",
            "Selected period",
        )
    with d:
        kpi(
            "OBSERVATIONS",
            f"{len(filtered):,}",
            "Records in selected period",
        )

    st.markdown("### Climate trends")

    left, right = st.columns([1.7, 1])

    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=filtered["date"],
                y=filtered["temperature_c"],
                name="Temperature",
                mode="lines",
                line=dict(color=TEAL, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(19,184,166,0.09)",
            )
        )
        fig.update_layout(
            title="Temperature over time",
            yaxis_title="Temperature (°C)",
            xaxis_title="",
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    with right:
        monthly = (
            filtered.assign(month=filtered["date"].dt.month)
            .groupby("month", as_index=False)
            .agg(
                temperature=("temperature_c", "mean"),
                rainfall=("rainfall_mm", "mean"),
            )
        )

        fig = px.bar(
            monthly,
            x="month",
            y="temperature",
            title="Average temperature by month",
            labels={
                "month": "Calendar month",
                "temperature": "Temperature (°C)",
            },
            color_discrete_sequence=[BLUE],
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    left, right = st.columns(2)

    with left:
        fig = px.line(
            filtered,
            x="date",
            y="rainfall_mm",
            title="Rainfall history",
            labels={
                "date": "Date",
                "rainfall_mm": "Rainfall (mm)",
            },
            color_discrete_sequence=[BLUE],
        )
        st.plotly_chart(
            style_chart(fig, 300),
            use_container_width=True,
        )

    with right:
        fig = px.line(
            filtered,
            x="date",
            y="humidity_pct",
            title="Humidity history",
            labels={
                "date": "Date",
                "humidity_pct": "Humidity (%)",
            },
            color_discrete_sequence=["#9B72CF"],
        )
        st.plotly_chart(
            style_chart(fig, 300),
            use_container_width=True,
        )

    st.info(
        "This project uses simulated climate data for educational "
        "purposes. Results are not official climate assessments."
    )


# ---------------- CLIMATE EXPLORER ----------------
elif page == "Climate Explorer":
    st.subheader("Climate Explorer")
    metric = st.selectbox(
        "Select climate variable",
        {
            "Temperature": "temperature_c",
            "Rainfall": "rainfall_mm",
            "Humidity": "humidity_pct",
        },
    )

    column = {
        "Temperature": "temperature_c",
        "Rainfall": "rainfall_mm",
        "Humidity": "humidity_pct",
    }[metric]

    fig = px.line(
        filtered,
        x="date",
        y=column,
        title=f"{metric} over time",
        markers=True,
        labels={"date": "Date", column: metric},
        color_discrete_sequence=[TEAL],
    )
    st.plotly_chart(
        style_chart(fig, 450),
        use_container_width=True,
    )

    monthly = (
        filtered.assign(month=filtered["date"].dt.month)
        .groupby("month", as_index=False)
        .agg(
            temperature_c=("temperature_c", "mean"),
            rainfall_mm=("rainfall_mm", "mean"),
            humidity_pct=("humidity_pct", "mean"),
        )
    )

    st.markdown("### Monthly summary")
    st.dataframe(
        monthly.round(2),
        use_container_width=True,
        hide_index=True,
    )


# ---------------- ANOMALIES ----------------
elif page == "Anomaly Analysis":
    st.subheader("Temperature Anomaly Analysis")
    anomalies = load_output("temperature_anomalies.csv")

    if anomalies is None:
        st.warning(
            "Run python src\\analyze_patterns.py first."
        )
    else:
        anomalies["date"] = pd.to_datetime(anomalies["date"])
        anomalies = anomalies[
            (anomalies["date"].dt.date >= start_date)
            & (anomalies["date"].dt.date <= end_date)
        ]

        st.metric("Flagged observations", len(anomalies))

        if not anomalies.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=filtered["date"],
                    y=filtered["temperature_c"],
                    mode="lines",
                    name="Temperature",
                    line=dict(color=BLUE),
                )
            )
            visible = anomalies[
                anomalies["date"].isin(filtered["date"])
            ]
            fig.add_trace(
                go.Scatter(
                    x=visible["date"],
                    y=visible["temperature_c"],
                    mode="markers",
                    name="Flagged anomaly",
                    marker=dict(color="#E45858", size=10, symbol="x"),
                )
            )
            fig.update_layout(
                title="Temperature observations and flagged anomalies",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
            )
            st.plotly_chart(
                style_chart(fig, 400),
                use_container_width=True,
            )

            st.dataframe(
                anomalies,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No flagged anomalies in this date range.")

        st.caption(
            "Anomalies are identified using calendar-month z-scores. "
            "A flagged point is not automatically a real climate extreme."
        )


# ---------------- FORECAST LAB ----------------
elif page == "Forecast Lab":
    st.subheader("Forecast Lab")
    metrics = load_output("forecast_model_comparison.csv")
    predictions = load_output("climate_forecast_predictions.csv")

    if metrics is not None:
        st.markdown("### Model evaluation")
        st.dataframe(
            metrics.round(3),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            metrics,
            x="model",
            y=["MAE", "RMSE"],
            barmode="group",
            title="Forecast model error comparison",
            labels={"value": "Error (°C)", "model": "Model"},
            color_discrete_sequence=[TEAL, BLUE],
        )
        st.plotly_chart(
            style_chart(fig, 400),
            use_container_width=True,
        )
    else:
        st.info(
            "Run python src\\forecast_climate.py first."
        )

    if predictions is not None:
        predictions["date"] = pd.to_datetime(predictions["date"])
        predictions = predictions[
            (predictions["date"].dt.date >= start_date)
            & (predictions["date"].dt.date <= end_date)
        ]

        prediction_cols = [
            c for c in [
                "actual_temperature_c",
                "mean_baseline_prediction",
                "seasonal_naive_prediction",
                "random_forest_prediction",
            ]
            if c in predictions.columns
        ]

        if not predictions.empty and prediction_cols:
            fig = px.line(
                predictions,
                x="date",
                y=prediction_cols,
                title="Actual vs predicted temperature",
                labels={
                    "date": "Date",
                    "value": "Temperature (°C)",
                    "variable": "Series",
                },
                color_discrete_sequence=[
                    NAVY, "#9B72CF", BLUE, TEAL
                ],
            )
            st.plotly_chart(
                style_chart(fig, 450),
                use_container_width=True,
            )

            st.download_button(
                "Download forecast predictions",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="climate_forecast_predictions.csv",
                mime="text/csv",
            )


# ---------------- MODEL INSIGHTS ----------------
elif page == "Model Insights":
    st.subheader("Model Insights")
    importance = load_output("feature_importance.csv")

    if importance is None:
        st.info(
            "Run python src\\analyze_feature_importance.py first."
        )
    else:
        importance = importance.sort_values(
            "importance", ascending=True
        )

        fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Random Forest feature importance",
            labels={
                "importance": "Relative importance",
                "feature": "Input feature",
            },
            color_discrete_sequence=[TEAL],
        )
        st.plotly_chart(
            style_chart(fig, 450),
            use_container_width=True,
        )

        st.dataframe(
            importance.sort_values(
                "importance", ascending=False
            ).round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Importance is model-specific and does not imply causation. "
            "Correlated inputs may share importance."
        )


# ---------------- DATA EXPLORER ----------------
elif page == "Data Explorer":
    st.subheader("Data Explorer")
    st.caption(
        "Inspect and download the selected climate observations."
    )

    search = st.text_input(
        "Search dataset columns or values",
        placeholder="Optional: search the displayed data",
    )

    display_df = filtered.copy()

    if search.strip():
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(
                search.strip(), case=False, na=False
            )
        ).any(axis=1)
        display_df = display_df[mask]

    st.write(f"Matching records: {len(display_df):,}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download filtered climate data",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_climate_data.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download complete climate dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="climate_dataset.csv",
        mime="text/csv",
    )


# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "CLIMATEIQ • Built with Python, Pandas, Scikit-learn, "
    "Streamlit and Plotly • Simulated data • Educational use only"
)
