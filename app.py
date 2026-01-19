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
# Page config & styling
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="African Cities Event Readiness Dashboard",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple custom styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 6px; }
    h1, h2, h3 { color: #1b5e20; }
    .sidebar .sidebar-content { background-color: #e8f5e9; padding: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Load metrics data
# ───────────────────────────────────────────────
@st.cache_data
def load_city_metrics():
    path = "city_metrics.csv"
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error reading city_metrics.csv: {str(e)}")
        return pd.DataFrame()

df_metrics = load_city_metrics()

# ───────────────────────────────────────────────
# City list and file mappings
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

# ───────────────────────────────────────────────
# Indicators & labels
# ───────────────────────────────────────────────
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
# Sidebar
# ───────────────────────────────────────────────
st.sidebar.title("Controls")

selected_cities = st.sidebar.multiselect(
    "Cities to compare",
    options=cities,
    default=cities[:3]
)

st.sidebar.header("ERI Weights")

weights = {}
for col in indicator_cols:
    label = indicator_labels.get(col, col)
    weights[col] = st.sidebar.slider(
        label,
        min_value=0.0,
        max_value=1.0,
        value=1.0 / len(indicator_cols),
        step=0.05,
        key=f"weight_{col}"
    )

# Normalize
total_weight = sum(weights.values())
normalized_weights = {k: (v / total_weight if total_weight > 0 else 1/len(indicator_cols)) for k, v in weights.items()}

# Compute ERI
if not df_metrics.empty and all(col in df_metrics.columns for col in indicator_cols):
    df_metrics["ERI"] = df_metrics[indicator_cols].mul(list(normalized_weights.values()), axis=1).sum(axis=1)
else:
    st.sidebar.warning("Some indicators missing from data → ERI may be incomplete")

# ───────────────────────────────────────────────
# Main dashboard
# ───────────────────────────────────────────────
st.title("🏟️ African Cities – Event Readiness Dashboard")
st.caption("Compare infrastructure readiness for hosting large-scale events | Data: OSM, WorldPop, GADM, World Bank")

# ───────────────────────────────────────────────
# Rankings
# ───────────────────────────────────────────────
st.header("City Rankings – Event Readiness Index")

if not df_metrics.empty:
    df_view = df_metrics[df_metrics["city"].isin(selected_cities)][["city", "ERI"]].copy()
    if not df_view.empty:
        df_view = df_view.sort_values("ERI", ascending=False).reset_index(drop=True)
        df_view["Rank"] = df_view.index + 1
        df_view["ERI"] = df_view["ERI"].round(3)

        st.dataframe(
            df_view.style
                .background_gradient(subset=["ERI"], cmap="YlGn")
                .format(precision=3),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("None of the selected cities are present in the metrics data.")
else:
    st.error("Could not load city_metrics.csv")

# ───────────────────────────────────────────────
# Bar chart
# ───────────────────────────────────────────────
st.header("ERI Comparison Bar Chart")

if not df_metrics.empty and selected_cities:
    fig_bar = px.bar(
        df_metrics[df_metrics["city"].isin(selected_cities)],
        x="city",
        y="ERI",
        color="city",
        text_auto='.3f',
        title="Event Readiness Index – Selected Cities",
        labels={"ERI": "ERI Score", "city": "City"},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(showlegend=False, xaxis_title=None, yaxis_title="ERI Score")
    st.plotly_chart(fig_bar, use_container_width=True)

# ───────────────────────────────────────────────
# Radar chart
# ───────────────────────────────────────────────
st.header("Indicator Radar Comparison")

if selected_cities and not df_metrics.empty:
    fig_radar = go.Figure()

    for city in selected_cities:
        row = df_metrics[df_metrics["city"] == city]
        if not row.empty:
            values = row[indicator_cols].values.flatten().tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=list(indicator_labels.values()) + [list(indicator_labels.values())[0]],
                fill='toself',
                name=city,
                line=dict(width=2)
            ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Normalized Indicators – Radar View",
        height=600
    )
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Select cities with available data to see radar comparison.")

# ───────────────────────────────────────────────
# Maps – pure JSON + folium (no geopandas)
# ───────────────────────────────────────────────
st.header("City Infrastructure Overview Maps")

if selected_cities:
    map_columns = st.columns(min(3, len(selected_cities)))

    for idx, city in enumerate(selected_cities):
        with map_columns[idx % len(map_columns)]:
            st.subheader(city)

            feat_file = geojson_files.get(city)
            bound_file = boundary_files.get(city)

            if feat_file and bound_file and os.path.exists(feat_file) and os.path.exists(bound_file):
                try:
                    # Load GeoJSON as dictionaries
                    with open(feat_file, 'r', encoding='utf-8') as f:
                        gj_features = json.load(f)
                    with open(bound_file, 'r', encoding='utf-8') as f:
                        gj_boundary = json.load(f)

                    # Calculate approximate centroid
                    def collect_coordinates(geometry):
                        coords = []
                        def recurse(item):
                            if isinstance(item, list) and len(item) == 2 and all(isinstance(x, (int, float)) for x in item):
                                coords.append(item)
                            elif isinstance(item, list):
                                for sub in item:
                                    recurse(sub)
                        recurse(geometry.get('coordinates', []))
                        return coords

                    center_lat, center_lon = 0.0, 0.0
                    if 'features' in gj_boundary and gj_boundary['features']:
                        geom = gj_boundary['features'][0]['geometry']
                        all_points = collect_coordinates(geom)
                        if all_points:
                            lons = [p[0] for p in all_points]
                            lats = [p[1] for p in all_points]
                            center_lat, center_lon = mean(lats), mean(lons)

                    # Create map
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=11,
                        tiles="CartoDB positron"
                    )

                    # Add boundary
                    folium.GeoJson(
                        gj_boundary,
                        style_function=lambda _: {
                            'fillColor': 'none',
                            'color': '#1b5e20',
                            'weight': 3.5,
                            'opacity': 0.9
                        }
                    ).add_to(m)

                    # Add features with clustering
                    marker_cluster = MarkerCluster().add_to(m)

                    for feature in gj_features.get('features', []):
                        props = feature.get('properties', {})
                        popup = (
                            props.get('name') or
                            props.get('feature_type', 'Feature') or
                            'Unnamed feature'
                        )
                        folium.GeoJson(
                            feature,
                            popup=popup,
                            style_function=lambda _: {'color': '#1976d2', 'weight': 2.2}
                        ).add_to(marker_cluster)

                    # Show map
                    st_folium(m, width=420, height=420, returned_objects=[])

                except Exception as e:
                    st.error(f"Map rendering failed for {city}\n{str(e)}")
            else:
                st.warning(f"GeoJSON file(s) missing for {city}")
else:
    st.info("Select cities above to display their maps.")

# ───────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data sources: OpenStreetMap • WorldPop • GADM • World Bank  |  "
    f"Last updated: January 19, 2026  |  "
    "Event Readiness Index Dashboard – Nairobi, KE"
)
