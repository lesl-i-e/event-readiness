import streamlit as st
import pandas as pd

st.set_page_config(page_title="Event Readiness Index", layout="wide")

df = pd.read_csv("city_metrics.csv")

indicator_cols = [
    "road_density_km_km2_norm",
    "health_facilities_per_100k_pop_norm",
    "population_density_norm",
    "open_space_per_100k_pop_norm",
    "intersection_density_norm",
    "hotels_per_100k_norm",
    "airport_distance_km_norm"
]

st.title("Urban Event Readiness Index")

# Sidebar — weighting
st.sidebar.header("Indicator Weights")

weights = {}
for col in indicator_cols:
    weights[col] = st.sidebar.slider(col, 0.0, 1.0, 1/len(indicator_cols))

# Normalize weights
total = sum(weights.values())
weights = {k: v/total for k,v in weights.items()}

df["ERI_custom"] = sum(df[col] * w for col, w in weights.items())

# Ranking table
st.subheader("City Rankings")
st.dataframe(
    df[["city", "ERI_custom"]]
    .sort_values("ERI_custom", ascending=False)
    .reset_index(drop=True)
)

# City comparison
st.subheader("Indicator Breakdown")
city = st.selectbox("Select City", df["city"])
st.bar_chart(
    df[df["city"] == city][indicator_cols].T
)
