import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import os
from statistics import mean

# Set page config for wide layout and theme
st.set_page_config(
    page_title="African Cities Event Readiness Dashboard",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for visual appeal
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stSelectbox { background-color: white; }
    h1 { color: #2E8B57; }
    .sidebar .sidebar-content { background-color: #e6f2e6; }
    </style>
""", unsafe_allow_html=True)

# Load city metrics CSV
@st.cache_data
def load_metrics():
    return pd.read_csv("city_metrics.csv")  # Assume in root or adjust path

df_metrics = load_metrics()

# List of cities based on user's files
cities = ["Nairobi", "Kampala", "Dar es Salaam", "Kigali", "Casablanca"]

# Map city names to GeoJSON files (adjust paths if in 'data/' folder)
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

# Indicator columns from CSV
indicator_cols = [
    "road_density_km_km2_norm",
    "health_facilities_per_100k_pop_norm",
    "intersection_density_norm",
    "hotels_per_100k_norm",
    "airport_distance_km_norm",
    "open_space_per_100k_pop_norm",
    "population_density_norm"
]

# Human-readable labels for indicators
indicator_labels = {
    "road_density_km_km2_norm": "Road Density (km/km²)",
    "health_facilities_per_100k_pop_norm": "Health Facilities per 100k Pop",
    "intersection_density_norm": "Intersection Density",
    "hotels_per_100k_norm": "Hotels per 100k Pop",
    "airport_distance_km_norm": "Airport Distance (km)",
    "open_space_per_100k_pop_norm": "Open Space per 100k Pop",
    "population_density_norm": "Population Density"
}

# Sidebar for user inputs
st.sidebar.title("Dashboard Controls")
selected_cities = st.sidebar.multiselect("Select Cities to Compare", cities, default=cities[:3])

st.sidebar.header("Custom Weights for ERI")
weights = {}
for col in indicator_cols:
    label = indicator_labels.get(col, col)
    weights[col] = st.sidebar.slider(label, 0.0, 1.0, 1.0 / len(indicator_cols), step=0.05)

# Normalize weights
total_weight = sum(weights.values())
if total_weight > 0:
    weights = {k: v / total_weight for k, v in weights.items()}
else:
    weights = {k: 1.0 / len(indicator_cols) for k in indicator_cols}

# Compute custom ERI
df_metrics["ERI"] = df_metrics[indicator_cols].mul(list(weights.values()), axis=1).sum(axis=1)

# Filter for selected cities
df_selected = df_metrics[df_metrics["city"].isin(selected_cities)]

# Main content
st.title("🏟️ African Cities Event Readiness Dashboard")
st.markdown("""
    This interactive dashboard assesses the infrastructure readiness of selected African cities for hosting mega events. 
    Use the sidebar to customize weights and select cities. Data sourced from OSM, WorldPop, and World Bank.
""")

# Section 1: Overall Rankings
st.header("City Rankings by Event Readiness Index (ERI)")
df_ranked = df_selected[["city", "ERI"]].sort_values("ERI", ascending=False).reset_index(drop=True)
df_ranked["Rank"] = df_ranked.index + 1
st.dataframe(df_ranked.style.background_gradient(subset=["ERI"], cmap="YlGn"))

# Section 2: ERI Bar Chart
st.header("ERI Comparison")
fig_bar = px.bar(df_selected, x="city", y="ERI", color="city",
                 title="Event Readiness Index by City",
                 labels={"ERI": "ERI Score", "city": "City"},
                 color_discrete_sequence=px.colors.qualitative.Set3)
fig_bar.update_layout(showlegend=False)
st.plotly_chart(fig_bar, use_container_width=True)

# Section 3: Indicator Breakdown - Radar Chart
st.header("Indicator Breakdown")
if len(selected_cities) > 0:
    fig_radar = go.Figure()
    for city in selected_cities:
        city_data = df_selected[df_selected["city"] == city][indicator_cols].values.flatten().tolist()
        fig_radar.add_trace(go.Scatterpolar(
            r=city_data,
            theta=list(indicator_labels.values()),
            fill='toself',
            name=city
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Radar Chart of Normalized Indicators"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Select at least one city to view the radar chart.")

# Helper functions for GeoJSON parsing and centroids
def extract_all_coords(geom):
    """
    Recursively extract a flat list of [lon, lat] coordinate pairs from GeoJSON geometry.
    """
    coords = []

    def _rec(c):
        if isinstance(c[0], (float, int)):
            coords.append(c)
        else:
            for sub in c:
                _rec(sub)

    _rec(geom["coordinates"])
    return coords

def centroid_of_geom(geom):
    coords = extract_all_coords(geom)
    if not coords:
        return None, None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return mean(lats), mean(lons)

# Color mapping for feature types (keeps your visual scheme)
style_dict = {
    'roads': '#1f77b4',            # blue
    'health_emergency': '#d62728', # red
    'hotels_etc': '#2ca02c',       # green
    'stadiums': '#9467bd',         # purple
    'airports': '#ff7f0e',         # orange
    'public_spaces': '#98df8a',    # lightgreen
    'pedestrian': '#7f7f7f',       # gray
    'bus_stops': '#bcbd22',        # yellow/olive
    'intersections': '#000000',    # black
    'unknown': '#8c564b'           # brown fallback
}

# Section 4: Interactive Maps for Selected Cities (Plotly replacement for folium)
st.header("Interactive City Maps")
if selected_cities:
    cols = st.columns(min(len(selected_cities), 3))  # Up to 3 columns
    for i, city in enumerate(selected_cities):
        with cols[i % 3]:
            st.subheader(city)
            geo_path = geojson_files.get(city)
            boundary_path = boundary_files.get(city)
            if geo_path and boundary_path and os.path.exists(geo_path) and os.path.exists(boundary_path):
                try:
                    with open(geo_path, "r", encoding="utf-8") as f:
                        gj_features = json.load(f)
                    with open(boundary_path, "r", encoding="utf-8") as f:
                        gj_boundary = json.load(f)
                except Exception as e:
                    st.error(f"Failed to read GeoJSONs for {city}: {e}")
                    continue

                # Compute map center from boundary centroid (fallback to features centroid)
                center_lat, center_lon = None, None
                # Try boundary centroid first
                if gj_boundary.get("features"):
                    bcent = centroid_of_geom(gj_boundary["features"][0]["geometry"])
                    if bcent and all(bcent):
                        center_lat, center_lon = bcent
                # If not available, try features
                if (center_lat is None) and gj_features.get("features"):
                    fcent = centroid_of_geom(gj_features["features"][0]["geometry"])
                    if fcent and all(fcent):
                        center_lat, center_lon = fcent

                if center_lat is None:
                    st.warning(f"Could not compute map center for {city}. Skipping map.")
                    continue

                # Prepare scatter traces for feature points grouped by feature_type
                feature_type_groups = {}
                for feat in gj_features.get("features", []):
                    geom = feat.get("geometry")
                    props = feat.get("properties", {}) or {}
                    ftype = props.get("feature_type") or props.get("type") or "unknown"
                    name = props.get("name") or props.get("id") or ftype

                    if not geom:
                        continue

                    if geom.get("type") == "Point":
                        lon, lat = geom["coordinates"]
                    else:
                        lat, lon = centroid_of_geom(geom)  # note centroid returns (lat, lon)
                        if lat is None:
                            continue

                    group = feature_type_groups.setdefault(ftype, {"lons": [], "lats": [], "names": []})
                    group["lons"].append(lon)
                    group["lats"].append(lat)
                    group["names"].append(name)

                # Build figure
                fig = go.Figure()

                # Add boundary polygons / lines
                for feat in gj_boundary.get("features", []):
                    geom = feat.get("geometry")
                    if not geom:
                        continue
                    gtype = geom.get("type")
                    if gtype == "Polygon":
                        # outer ring
                        ring = geom["coordinates"][0]
                        lons = [c[0] for c in ring] + [ring[0][0]]
                        lats = [c[1] for c in ring] + [ring[0][1]]
                        fig.add_trace(go.Scattermapbox(
                            lon=lons, lat=lats,
                            mode="lines",
                            line=dict(color="black", width=2),
                            fill="none",
                            hoverinfo="skip",
                            showlegend=False
                        ))
                    elif gtype == "MultiPolygon":
                        for poly in geom["coordinates"]:
                            ring = poly[0]
                            lons = [c[0] for c in ring] + [ring[0][0]]
                            lats = [c[1] for c in ring] + [ring[0][1]]
                            fig.add_trace(go.Scattermapbox(
                                lon=lons, lat=lats,
                                mode="lines",
                                line=dict(color="black", width=2),
                                fill="none",
                                hoverinfo="skip",
                                showlegend=False
                            ))
                    else:
                        # for other geometry types attempt to plot centroids as small markers
                        lat, lon = centroid_of_geom(geom)
                        if lat is not None:
                            fig.add_trace(go.Scattermapbox(
                                lon=[lon], lat=[lat],
                                mode="markers",
                                marker=dict(size=6, color="black"),
                                hoverinfo="skip",
                                showlegend=False
                            ))

                # Add feature points grouped by type
                for ftype, grp in feature_type_groups.items():
                    color = style_dict.get(ftype, style_dict["unknown"])
                    fig.add_trace(go.Scattermapbox(
                        lon=grp["lons"],
                        lat=grp["lats"],
                        mode="markers",
                        marker=dict(size=8, color=color),
                        text=grp["names"],
                        hoverinfo="text",
                        name=ftype
                    ))

                # Layout: open-street-map style requires no token
                fig.update_layout(
                    mapbox=dict(
                        style="open-street-map",
                        center=dict(lat=center_lat, lon=center_lon),
                        zoom=12
                    ),
                    margin={"r":0,"t":30,"l":0,"b":0},
                    height=400,
                    legend=dict(title="Feature Types")
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"GeoJSON files for {city} not found.")
else:
    st.info("Select cities to view their interactive maps.")

# Section 5: Detailed Metrics Table
st.header("Detailed City Metrics")
st.dataframe(df_selected.style.background_gradient(cmap="YlGn"))

# Footer
st.markdown("---")
st.markdown("""
    **Data Sources**: OpenStreetMap (infrastructure), WorldPop (population), GADM (boundaries), World Bank (national indicators).  
    **Developed by**: AI Assistant | Current Date: January 19, 2026 | Location: Nairobi, KE  
    For more analysis, check the notebooks in the repository.
""")
st.markdown(f"Hello, {st.session_state.get('user_name', 'Leslie_Gedion')}! If you need custom exports or further insights, let me know.")
