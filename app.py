# Diagnostic wrapper: add at top of app.py (before other imports)
import sys, subprocess
import streamlit as st

def show_env_and_freeze(info_msg=None):
    st.write("Environment:", sys.executable, sys.version)
    st.write("PIP freeze:")
    st.text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    if info_msg:
        st.error(info_msg)

try:
    import folium
    import streamlit_folium
except Exception as e:
    show_env_and_freeze(f"Import error: {e}")
    # re-raise so the build log / runtime still shows a failure
    raise

import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import json
from statistics import mean
import os

# ───────────────────────────────────────────────
# Page configuration & basic styling
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="African Cities Event Readiness Dashboard",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light custom styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 6px; }
    h1, h2, h3 { color: #1b5e20; }
    .sidebar .sidebar-content { background-color: #e8f5e9; }
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Load city metrics
# ───────────────────────────────────────────────
@st.cache_data
def load_metrics():
    path = "city_metrics.csv"
    if not os.path.exists(path):
        st.error(f"Cannot find city_metrics.csv at {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

df_metrics = load_metrics()

# ───────────────────────────────────────────────
# City & file mappings (update paths if files are in subfolder)
# ───────────────────────────────────────────────
cities = ["Nairobi", "Kampala", "Dar es Salaam", "Kigali", "Casablanca"]

geojson_files = {
    "Nairobi": "nairobi_infrastructure_features.geojson",
    "Kampala": "kampala_infrastructure_features.geojson",
    "Dar es Salaam": "dar_es_salaam_infrastructure_features.geojson",
    "Kigali": "kigali_infrastructure_features.geojson",
    "Casablanca": "casablanca_infrastructure_features.geojson"
}

boundary_files = {
    "Nairobi": "nairobi_city_boundary.geojson",
    "Kampala": "kampala_city_boundary.geojson",
    "Dar es Salaam": "dar_es_salaam_city_boundary.geojson",
    "Kigali": "kigali_city_boundary.geojson",
    "Casablanca": "casablanca_city_boundary.geojson"
}

# Indicators used in the dashboard
indicator_cols = [
    "road_density_km_km2_norm",
    "health_facilities_per_100k_pop_norm",
    "intersection_density_norm",
    "hotels_per_100k_norm",
    "airport_distance_km_norm",
    "open_space_per_100k_pop_norm",
    "population_density_norm"
]

indicator_labels = {
    "road_density_km_km2_norm": "Road Density (km/km²)",
    "health_facilities_per_100k_pop_norm": "Health Facilities / 100k",
    "intersection_density_norm": "Intersection Density",
    "hotels_per_100k_norm": "Hotels / 100k Pop",
    "airport_distance_km_norm": "Airport Proximity Score",
    "open_space_per_100k_pop_norm": "Open Space / 100k Pop",
    "population_density_norm": "Population Density"
}

# ───────────────────────────────────────────────
# Sidebar controls
# ───────────────────────────────────────────────
st.sidebar.title("Dashboard Controls")

selected_cities = st.sidebar.multiselect(
    "Select Cities to Compare",
    cities,
    default=cities[:3]
)

st.sidebar.header("Custom Weights (Event Readiness Index)")

weights = {}
for col in indicator_cols:
    label = indicator_labels.get(col, col)
    weights[col] = st.sidebar.slider(label, 0.0, 1.0, 1.0 / len(indicator_cols), step=0.05)

# Normalize weights
total = sum(weights.values())
if total > 0:
    weights = {k: v / total for k, v in weights.items()}
else:
    weights = {k: 1.0 / len(indicator_cols) for k in indicator_cols}

# Compute custom ERI
if not df_metrics.empty:
    df_metrics["ERI"] = df_metrics[indicator_cols].mul(list(weights.values()), axis=1).sum(axis=1)

# ───────────────────────────────────────────────
# Main content
# ───────────────────────────────────────────────
st.title("🏟️ African Cities Event Readiness Dashboard")
st.markdown("""
    Compare infrastructure readiness for hosting mega events across five African cities.  
    Data sources: OpenStreetMap, WorldPop, GADM, World Bank indicators.  
    Customize weights in the sidebar to see how rankings change.
""")

# ───────────────────────────────────────────────
# Rankings table
# ───────────────────────────────────────────────
if not df_metrics.empty:
    st.header("City Rankings – Event Readiness Index")
    df_ranked = df_metrics[df_metrics["city"].isin(selected_cities)][["city", "ERI"]].copy()
    if not df_ranked.empty:
        df_ranked = df_ranked.sort_values("ERI", ascending=False).reset_index(drop=True)
        df_ranked["Rank"] = df_ranked.index + 1
        st.dataframe(
            df_ranked.style
               .background_gradient(subset=["ERI"], cmap="YlGn")
               .format({"ERI": "{:.3f}"})
        )
    else:
        st.info("No selected cities found in the metrics data.")
else:
    st.error("Could not load city_metrics.csv – check file path and contents.")

# ───────────────────────────────────────────────
# Bar chart comparison
# ───────────────────────────────────────────────
st.header("ERI Comparison")
if not df_metrics.empty and selected_cities:
    fig_bar = px.bar(
        df_metrics[df_metrics["city"].isin(selected_cities)],
        x="city",
        y="ERI",
        color="city",
        title="Event Readiness Index by City",
        labels={"ERI": "ERI Score", "city": "City"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_bar.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Select cities and ensure metrics data is loaded.")

# ───────────────────────────────────────────────
# Radar chart for indicator breakdown
# ───────────────────────────────────────────────
st.header("Indicator Breakdown (Radar)")
if selected_cities and not df_metrics.empty:
    fig_radar = go.Figure()
    for city in selected_cities:
        row = df_metrics[df_metrics["city"] == city]
        if not row.empty:
            values = row[indicator_cols].values.flatten().tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # close the polygon
                theta=list(indicator_labels.values()) + [list(indicator_labels.values())[0]],
                fill='toself',
                name=city
            ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Normalized Indicators – Multi-City Comparison"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Select cities with available metrics data to view radar chart.")

# ───────────────────────────────────────────────
# Interactive maps – pure json + folium (no geopandas)
# ───────────────────────────────────────────────
st.header("City Infrastructure Maps")

if selected_cities:
    map_cols = st.columns(min(3, len(selected_cities)))
    for idx, city in enumerate(selected_cities):
        with map_cols[idx % 3]:
            st.subheader(city)
            feat_path = geojson_files.get(city)
            bound_path = boundary_files.get(city)

            if feat_path and bound_path and os.path.exists(feat_path) and os.path.exists(bound_path):
                try:
                    # Load raw GeoJSON
                    with open(feat_path, 'r', encoding='utf-8') as f:
                        gj_features = json.load(f)
                    with open(bound_path, 'r', encoding='utf-8') as f:
                        gj_boundary = json.load(f)

                    # Crude centroid calculation
                    def extract_all_coords(geom):
                        coords = []
                        def recurse(obj):
                            if isinstance(obj, list) and len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
                                coords.append(obj)
                            elif isinstance(obj, list):
                                for item in obj:
                                    recurse(item)
                        recurse(geom.get('coordinates', []))
                        return coords

                    center = [0.0, 0.0]
                    if 'features' in gj_boundary and gj_boundary['features']:
                        first_geom = gj_boundary['features'][0]['geometry']
                        all_coords = extract_all_coords(first_geom)
                        if all_coords:
                            lons = [c[0] for c in all_coords]
                            lats = [c[1] for c in all_coords]
                            center = [mean(lats), mean(lons)]

                    # Create base map
                    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

                    # Boundary outline
                    folium.GeoJson(
                        gj_boundary,
                        style_function=lambda x: {'fillColor': 'none', 'color': '#1b5e20', 'weight': 3, 'opacity': 0.9}
                    ).add_to(m)

                    # Features with clustering
                    marker_cluster = MarkerCluster().add_to(m)
                    for feature in gj_features.get('features', []):
                        props = feature.get('properties', {})
                        popup_content = props.get('name') or props.get('feature_type', 'Feature') or 'Unnamed'
                        folium.GeoJson(
                            feature,
                            popup=popup_content,
                            style_function=lambda x: {'color': '#1976d2', 'weight': 2.5, 'opacity': 0.7}
                        ).add_to(marker_cluster)

                    # Render
                    st_folium(m, width=420, height=420, returned_objects=[])

                except Exception as e:
                    st.error(f"Could not render map for {city}\n{str(e)}")
            else:
                st.warning(f"Missing GeoJSON file(s) for {city}")
else:
    st.info("Select one or more cities to display their infrastructure maps.")

# ───────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: OpenStreetMap • WorldPop • GADM • World Bank WDI  |  "
    f"Dashboard updated: {st.session_state.get('last_update', 'January 19, 2026')}  |  "
    "Built for comparative urban event readiness analysis"
)
