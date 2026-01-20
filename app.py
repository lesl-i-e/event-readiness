import streamlit as st
import pandas as pd
import json
from statistics import mean
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import os

# ───────────────────────────────────────────────
# Page config & dark theme styling
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Event Readiness – African Cities",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
    <style>
    .main { background-color: #1e1e1e; color: #e0e0e0; }
    h1, h2, h3, h4 { color: #a8dadc; }
    .stButton>button { background-color: #3c6e71; color: #e0e0e0; border: none; border-radius: 6px; }
    .sidebar .sidebar-content { background-color: #2b2b2b; color: #e0e0e0; padding: 1rem; }
    .stAlert { background-color: #3a3a3a; color: #e0e0e0; }
    .stDataFrame { background-color: #2b2b2b; color: #e0e0e0; }
    .stMarkdown { color: #e0e0e0; }
    .block-container { padding: 1rem; background-color: #252525; border-radius: 8px; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Load metrics with debug
# ───────────────────────────────────────────────
@st.cache_data
def load_metrics():
    path = "city_metrics.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            st.sidebar.success(f"Loaded {len(df)} cities from CSV")
            return df
        except Exception as e:
            st.sidebar.error(f"CSV load error: {str(e)}")
            return pd.DataFrame()
    else:
        st.sidebar.error("city_metrics.csv not found")
        return pd.DataFrame()

df_metrics = load_metrics()

# ───────────────────────────────────────────────
# City & files
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

# Indicators
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
# Sidebar for global controls
# ───────────────────────────────────────────────
st.sidebar.title("Global Controls")

selected_cities = st.sidebar.multiselect(
    "Compare Cities",
    cities,
    default=cities[:2]
)

st.sidebar.header("ERI Weights")
weights = {}
for col in indicator_cols:
    label = indicator_labels.get(col, col)
    weights[col] = st.sidebar.slider(label, 0.0, 1.0, 1.0 / len(indicator_cols), step=0.05)

# Normalize weights
total = sum(weights.values())
weights = {k: v / total if total > 0 else 1.0 / len(indicator_cols) for k, v in weights.items()}

# Compute ERI
if not df_metrics.empty:
    df_metrics["ERI"] = df_metrics[indicator_cols].mul(list(weights.values()), axis=1).sum(axis=1)

# ───────────────────────────────────────────────
# Tabs structure
# ───────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview & Rankings", "Indicator Breakdown", "Interactive Maps", "Single City Focus"])

with tab1:
    st.header("Overview & Rankings")
    st.markdown("Global comparison of Event Readiness Index (ERI) across selected cities. ERI is a weighted average of normalized indicators.")

    if not df_metrics.empty and selected_cities:
        df_rank = df_metrics[df_metrics["city"].isin(selected_cities)][["city", "ERI"]].sort_values("ERI", ascending=False).reset_index(drop=True)
        df_rank["Rank"] = df_rank.index + 1
        df_rank["ERI"] = df_rank["ERI"].round(3)

        st.dataframe(
            df_rank,
            use_container_width=True,
            hide_index=True
        )

        st.caption("Comment: Higher ERI indicates better overall readiness. Adjust weights in sidebar to prioritize aspects like transport or health.")
    else:
        st.info("Select cities or check if CSV data is loaded.")

with tab2:
    st.header("Indicator Breakdown")
    st.markdown("Detailed view of individual indicators and radar comparison.")

    if not df_metrics.empty and selected_cities:
        # Radar chart
        fig_radar = go.Figure()
        for city in selected_cities:
            row = df_metrics[df_metrics["city"] == city]
            if not row.empty:
                values = row[indicator_cols].values.flatten().tolist()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=list(indicator_labels.values()) + [list(indicator_labels.values())[0]],
                    fill='toself',
                    name=city
                ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.caption("Comment: Radar shows strengths/weaknesses. E.g., high road density suggests good connectivity but may imply traffic issues.")
    else:
        st.info("No data for breakdown.")

with tab3:
    st.header("Interactive Maps")
    st.markdown("Explore infrastructure features on maps for selected cities.")

    if selected_cities:
        cols = st.columns(min(3, len(selected_cities)))
        for i, city in enumerate(selected_cities):
            with cols[i % 3]:
                st.subheader(city)
                feat_path = geojson_files.get(city)
                bound_path = boundary_files.get(city)
                if feat_path and bound_path and os.path.exists(feat_path) and os.path.exists(bound_path):
                    try:
                        with open(feat_path, 'r') as f:
                            gj_feat = json.load(f)
                        with open(bound_path, 'r') as f:
                            gj_bound = json.load(f)

                        # Centroid
                        def get_coords(g):
                            coords = []
                            def rec(c):
                                if isinstance(c, list) and len(c) == 2 and isinstance(c[0], (float, int)):
                                    coords.append(c)
                                elif isinstance(c, list):
                                    for x in c:
                                        rec(x)
                            rec(g.get('coordinates', []))
                            return coords

                        coords = get_coords(gj_bound['features'][0]['geometry']) if 'features' in gj_bound else []
                        center = [mean([c[1] for c in coords]), mean([c[0] for c in coords])] if coords else [0, 0]

                        m = folium.Map(location=center, zoom_start=11, tiles="CartoDB dark_matter")

                        folium.GeoJson(gj_bound, style_function=lambda _: {'color': 'lightblue', 'weight': 2}).add_to(m)

                        cluster = MarkerCluster().add_to(m)
                        for feat in gj_feat.get('features', []):
                            folium.GeoJson(feat, popup=feat.get('properties', {}).get('name', 'Feature')).add_to(cluster)

                        st_folium(m, width=400, height=400)
                    except Exception as e:
                        st.warning(f"Map error: {str(e)}")
                else:
                    st.warning("Files missing")
    else:
        st.info("Select cities")

# =============================================================================
# TAB 4 – Single City Deep Dive
# =============================================================================
with tab4:
    st.header("Single City Deep Dive")
    st.markdown(
        "Explore detailed normalized indicators and infrastructure map for **one city at a time**.\n\n"
        "Change the city using the dropdown below — the view updates instantly."
    )

    # Persistent city choice within this tab
    if "deep_dive_city" not in st.session_state:
        st.session_state.deep_dive_city = "Nairobi"  # default

    single_city = st.selectbox(
        "Select city to explore",
        options=cities,
        index=cities.index(st.session_state.deep_dive_city),
        key="deep_dive_city_tab_selector"
    )

    # Update session state so choice survives reruns / tab switches
    st.session_state.deep_dive_city = single_city

    st.subheader(f"Deep Dive: **{single_city}**")

    if not df_metrics.empty:
        row = df_metrics[df_metrics["city"] == single_city]

        if not row.empty:
            # ───────────────────────────────
            # ERI Metric
            # ───────────────────────────────
            eri_value = row["ERI"].iloc[0]
            st.metric(
                label="Event Readiness Index (ERI)",
                value=f"{eri_value:.3f}",
                help="Weighted average of normalized indicators below"
            )

            # ───────────────────────────────
            # Bar chart – already working well
            # ───────────────────────────────
            scores_dict = row[indicator_cols].iloc[0].to_dict()

            plot_df = pd.DataFrame({
                "Indicator": [indicator_labels.get(k, k.replace("_norm", "").replace("_", " ").title()) 
                              for k in scores_dict],
                "Score": list(scores_dict.values())
            })

            fig_single = px.bar(
                plot_df,
                x="Indicator",
                y="Score",
                title=f"Normalized Indicator Scores – {single_city}",
                labels={"Score": "Normalized Score (0–1)"},
                text_auto=".3f",
                height=480,
                color="Score",
                color_continuous_scale="viridis"
            )

            fig_single.update_layout(
                xaxis_tickangle=-45,
                xaxis_title=None,
                showlegend=False,
                margin=dict(l=20, r=20, t=60, b=120),
                font=dict(size=12)
            )

            st.plotly_chart(fig_single, use_container_width=True)

            st.caption(
                "**Quick interpretation**\n"
                "• Taller bars = stronger performance in that area\n"
                "• High road/intersection density → good accessibility\n"
                "• High health/hotels → better crowd & emergency handling\n"
                "• Population density gives context for crowd pressure"
            )

            # ───────────────────────────────
            # Infrastructure Map – with strong debugging
            # ───────────────────────────────
            st.subheader(f"Infrastructure Map – {single_city}")

            feat_path = geojson_files.get(single_city)
            bound_path = boundary_files.get(single_city)

            if not feat_path or not bound_path:
                st.error(f"Mapping paths not defined for {single_city}")
            elif not os.path.exists(feat_path):
                st.error(f"Features file missing: {feat_path}")
            elif not os.path.exists(bound_path):
                st.error(f"Boundary file missing: {bound_path}")
            else:
                try:
                    st.info("Loading map files...")

                    with open(feat_path, 'r', encoding='utf-8') as f:
                        gj_features = json.load(f)

                    with open(bound_path, 'r', encoding='utf-8') as f:
                        gj_boundary = json.load(f)

                    st.success("Files loaded successfully")

                    # Centroid calculation
                    def extract_coordinates(geom):
                        coords = []
                        def recurse(obj):
                            if isinstance(obj, list) and len(obj) == 2 and isinstance(obj[0], (int, float)):
                                coords.append(obj)
                            elif isinstance(obj, list):
                                for item in obj:
                                    recurse(item)
                        recurse(geom.get('coordinates', []))
                        return coords

                    all_points = []
                    if 'features' in gj_boundary and gj_boundary['features']:
                        all_points = extract_coordinates(gj_boundary['features'][0]['geometry'])

                    if all_points:
                        center_lat = mean(c[1] for c in all_points)
                        center_lon = mean(c[0] for c in all_points)
                    else:
                        center_lat, center_lon = 0.0, 0.0
                        st.warning("Could not calculate center – using fallback (0,0)")

                    # Create map
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=11,
                        tiles="CartoDB positron"
                    )

                    # Boundary
                    folium.GeoJson(
                        gj_boundary,
                        style_function=lambda x: {'color': '#1f77b4', 'weight': 3, 'fillOpacity': 0.1}
                    ).add_to(m)

                    # Features cluster
                    marker_cluster = MarkerCluster().add_to(m)
                    for feat in gj_features.get('features', []):
                        props = feat.get('properties', {})
                        popup = props.get('name', '') or props.get('feature_type', 'Feature') or 'Unnamed'
                        folium.GeoJson(
                            feat,
                            popup=popup
                        ).add_to(marker_cluster)

                    # Render
                    st_folium(m, width=None, height=600, returned_objects=[])

                except json.JSONDecodeError as je:
                    st.error(f"JSON parsing error: {str(je)}")
                except Exception as e:
                    st.error(f"Map rendering failed: {str(e)}")
                    st.caption("Common causes: invalid GeoJSON format, empty features, or coordinate issues.")
        else:
            st.warning(f"No data row found for **{single_city}** in city_metrics.csv")
    else:
        st.info("Metrics data not loaded – check if city_metrics.csv exists and has correct columns.")

# ───────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────
st.markdown("---")
st.caption("Data: OSM/WorldPop/GADM/World Bank | Jan 19, 2026 | Nairobi, KE")
