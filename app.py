import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import os

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

# Section 4: Interactive Maps for Selected Cities
st.header("Interactive City Maps")
if selected_cities:
    cols = st.columns(min(len(selected_cities), 3))  # Up to 3 columns
    for i, city in enumerate(selected_cities):
        with cols[i % 3]:
            st.subheader(city)
            if os.path.exists(geojson_files[city]) and os.path.exists(boundary_files[city]):
                # Load GeoJSONs
                gdf_features = gpd.read_file(geojson_files[city])
                gdf_boundary = gpd.read_file(boundary_files[city])
                
                # Create Folium map
                m = folium.Map(location=[gdf_boundary.centroid.y.mean(), gdf_boundary.centroid.x.mean()], zoom_start=12, tiles="CartoDB positron")
                
                # Add boundary
                folium.GeoJson(gdf_boundary, style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 2}).add_to(m)
                
                # Add features with styles based on feature_type
                style_dict = {
                    'roads': {'color': 'blue', 'weight': 2},
                    'health_emergency': {'color': 'red', 'weight': 3},
                    'hotels_etc': {'color': 'green', 'weight': 3},
                    'stadiums': {'color': 'purple', 'weight': 4},
                    'airports': {'color': 'orange', 'weight': 4},
                    'public_spaces': {'color': 'lightgreen', 'weight': 2},
                    'pedestrian': {'color': 'gray', 'weight': 1},
                    'bus_stops': {'color': 'yellow', 'weight': 3},
                    'intersections': {'color': 'black', 'weight': 1}
                }
                
                marker_cluster = MarkerCluster().add_to(m)
                for idx, row in gdf_features.iterrows():
                    feature_type = row.get('feature_type', 'unknown')
                    style = style_dict.get(feature_type, {'color': 'gray', 'weight': 1})
                    popup = f"{feature_type}: {row.get('name', 'Unnamed')}"
                    folium.GeoJson(row.geometry, style_function=lambda x=style: style, popup=popup).add_to(marker_cluster)
                
                # Display map
                st_folium(m, width=400, height=400)
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
