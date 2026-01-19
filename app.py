import streamlit as st
import pandas as pd
import json
from statistics import mean
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import os

# ───────────────────────────────────────────────
# Page config
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Event Readiness – African Cities",
    page_icon="🌍",
    layout="wide"
)

# Minimal styling – avoid pink surprises
st.markdown("""
    <style>
    .main { background-color: #f9fafb; }
    h1 { color: #1a3c34; }
    .stAlert { margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Title & intro
# ───────────────────────────────────────────────
st.title("🌍 Event Readiness – African Cities")
st.caption("Compare infrastructure readiness for hosting major events")
st.markdown("---")

# ───────────────────────────────────────────────
# Sidebar – weights & city selection
# ───────────────────────────────────────────────
st.sidebar.header("Controls")

cities = ["Nairobi", "Kampala", "Dar es Salaam", "Kigali", "Casablanca"]

selected_cities = st.sidebar.multiselect(
    "Select cities",
    cities,
    default=["Nairobi", "Kampala"]
)

# ───────────────────────────────────────────────
# Load data with loud debug messages
# ───────────────────────────────────────────────
DATA_FILE = "city_metrics.csv"

if not os.path.exists(DATA_FILE):
    st.sidebar.error(f"CRITICAL: {DATA_FILE} not found in app root")
    st.error(f"Cannot find **{DATA_FILE}** in the deployed app folder.")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
    st.sidebar.success(f"Loaded {len(df)} rows from {DATA_FILE}")
except Exception as e:
    st.error(f"Failed to read {DATA_FILE}\n{e}")
    st.stop()

required_cols = [
    "city",
    "road_density_km_km2_norm",
    "health_facilities_per_100k_pop_norm",
    "intersection_density_norm",
    "hotels_per_100k_norm",
    "airport_distance_km_norm",
    "open_space_per_100k_pop_norm",
    "population_density_norm"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}")
    st.stop()

# Filter to selected cities
df_sel = df[df["city"].isin(selected_cities)].copy()

if df_sel.empty:
    st.warning("None of the selected cities exist in the CSV. Check spelling/case in 'city' column.")
    st.info(f"Available cities in data: {', '.join(df['city'].unique())}")
else:
    st.success(f"Showing data for {len(df_sel)} cities")

# ───────────────────────────────────────────────
# Weights
# ───────────────────────────────────────────────
st.sidebar.header("ERI Weights")

indicators = required_cols[1:]  # exclude 'city'
weights = {}
for col in indicators:
    nice_name = col.replace("_norm", "").replace("_", " ").title()
    weights[col] = st.sidebar.slider(nice_name, 0.0, 1.0, 0.14, step=0.05)

total_w = sum(weights.values())
weights_norm = {k: v/total_w if total_w > 0 else 1/len(indicators) for k,v in weights.items()}

# Calculate ERI
df_sel["ERI"] = df_sel[indicators].mul(list(weights_norm.values()), axis=1).sum(axis=1)

# ───────────────────────────────────────────────
# Main content – simple & robust
# ───────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Rankings")
    if not df_sel.empty:
        rank_df = df_sel[["city", "ERI"]].sort_values("ERI", ascending=False).reset_index(drop=True)
        rank_df["Rank"] = rank_df.index + 1
        st.dataframe(
            rank_df[["Rank", "city", "ERI"]].style.format({"ERI": "{:.3f}"}),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No matching data – check city names in CSV")

with col_right:
    st.subheader("ERI Bar Chart")
    if not df_sel.empty:
        fig = px.bar(
            df_sel,
            x="city",
            y="ERI",
            color="city",
            text_auto=".3f",
            height=300
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to plot")

# ───────────────────────────────────────────────
# Maps – very simple version
# ───────────────────────────────────────────────
st.subheader("City Maps")

if selected_cities:
    map_cols = st.columns(3)
    for i, city in enumerate(selected_cities[:3]):  # max 3 maps
        with map_cols[i]:
            st.caption(city)

            feat_path = geojson_files.get(city)
            bound_path = boundary_files.get(city)

            if not (feat_path and bound_path and os.path.exists(feat_path) and os.path.exists(bound_path)):
                st.warning("Map files missing")
                continue

            try:
                with open(bound_path, 'r', encoding='utf-8') as f:
                    bound = json.load(f)

                # Very simple centroid
                coords = []
                def find_coords(g):
                    if isinstance(g, list) and len(g) == 2 and isinstance(g[0], (int,float)):
                        coords.append(g)
                    elif isinstance(g, list):
                        for x in g: find_coords(x)
                find_coords(bound.get('features',[{}])[0].get('geometry',{}).get('coordinates',[]))

                if coords:
                    lat = mean(c[1] for c in coords)
                    lon = mean(c[0] for c in coords)
                else:
                    lat, lon = 0, 0

                m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB positron")

                folium.GeoJson(bound, style_function=lambda _: {'color': 'darkgreen', 'weight': 3}).add_to(m)

                st_folium(m, width=300, height=300, returned_objects=[])

            except Exception as e:
                st.error(f"Map error: {str(e)[:80]}...")
else:
    st.info("Select cities to see maps")

st.markdown("---")
st.caption("Data: OSM / WorldPop / GADM / World Bank | App rebuilt Jan 19, 2026")
