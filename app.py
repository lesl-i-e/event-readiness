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
tab1, tab2, tab3, = st.tabs(["Overview & Rankings", "Indicator Breakdown", "Single City Focus"])
# Global quick links / navigation hint
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <strong>Navigate the dashboard:</strong>  
        Overview → compare all cities • Breakdown → deep multi-city analysis •  
        Deep Dive → explore one city + map
    </div>
    """,
    unsafe_allow_html=True
)
# =============================================================================
# TAB 1 – Overview & Rankings (Landing Page)
# =============================================================================
with tab1:
    st.header("Welcome to Event Readiness Dashboard")
    st.markdown(
        """
        Compare how well five African cities are prepared to host major events  
        based on transport, health, accommodation, and crowd management indicators.
        """
    )

    # Quick stats cards – makes it feel like a dashboard landing page
    if not df_metrics.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cities Compared", len(df_metrics), help="Total cities in dataset")
        with col2:
            avg_eri = df_metrics["ERI"].mean()
            st.metric("Average ERI", f"{avg_eri:.3f}", help="Mean readiness across all cities")
        with col3:
            top_city = df_metrics.loc[df_metrics["ERI"].idxmax(), "city"]
            top_score = df_metrics["ERI"].max()
            st.metric("Top City", f"{top_city} ({top_score:.3f})", delta_color="normal")

    st.markdown("---")

    # Bar chart – placed first, prominent
    st.subheader("Event Readiness Index – All Cities")
    if not df_metrics.empty:
        # Sort for visual ranking
        df_bar = df_metrics.sort_values("ERI", ascending=False).copy()
        df_bar["ERI"] = df_bar["ERI"].round(3)

        fig_bar = px.bar(
            df_bar,
            x="city",
            y="ERI",
            color="city",
            text="ERI",                      # show value on bars
            title="Event Readiness Index Comparison",
            labels={"ERI": "ERI Score", "city": "City"},
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=450
        )

        fig_bar.update_traces(
            textposition="auto",
            textfont_size=14,
            marker_line_color="black",
            marker_line_width=1.2
        )

        fig_bar.update_layout(
            xaxis_title=None,
            yaxis_title="ERI Score (0–1)",
            showlegend=False,
            bargap=0.2,
            margin=dict(l=20, r=20, t=60, b=60),
            font=dict(size=13)
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        st.caption(
            "Higher bars = better overall readiness. "
            "Adjust weights in the sidebar to see how priorities change rankings."
        )
    else:
        st.info("No metrics data loaded – check city_metrics.csv")

    st.markdown("---")

    # Smaller, centered rankings table
    st.subheader("City Rankings")
    if not df_metrics.empty:
        df_rank = df_metrics[["city", "ERI"]].sort_values("ERI", ascending=False).reset_index(drop=True)
        df_rank["Rank"] = df_rank.index + 1
        df_rank["ERI"] = df_rank["ERI"].round(3)

        # Center the table visually (using columns)
        col_empty1, col_table, col_empty2 = st.columns([1, 2, 1])

        with col_table:
            st.dataframe(
                df_rank[["Rank", "city", "ERI"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "city": st.column_config.TextColumn("City", width="medium"),
                    "ERI": st.column_config.NumberColumn("ERI Score", format="%.3f", width="small")
                }
            )

        st.caption("Tip: Click column headers to sort. Explore individual cities in the 'Single City Deep Dive' tab.")
    else:
        st.info("Rankings will appear once data is loaded.")

    st.markdown("---")

    # Call-to-action / landing page footer feel
    st.info(
        "**Next steps**\n"
        "• Adjust weights in sidebar to customize rankings\n"
        "• Go to **Single City Deep Dive** tab to explore one city in detail\n"
        "• Use **Interactive Maps** tab to see spatial infrastructure layout"
    )

# =============================================================================
# TAB 2 – Indicator Breakdown (Comparison)
# =============================================================================
with tab2:
    st.header("Indicator Breakdown & Comparison")
    st.markdown(
        "See how selected cities perform across all readiness dimensions.\n"
        "The **radar chart** shows strengths and weaknesses visually, "
        "while the **parallel coordinates** plot helps spot trade-offs."
    )

    # ───────────────────────────────────────────────
    # City selection specific to this tab
    # ───────────────────────────────────────────────
    st.subheader("Select cities to compare")
    compare_cities = st.multiselect(
        "Choose 2–5 cities",
        options=cities,
        default=["Nairobi", "Kampala", "Kigali"],
        key="compare_cities_tab2"
    )

    if compare_cities:
        df_comp = df_metrics[df_metrics["city"].isin(compare_cities)].copy()
        
        if not df_comp.empty:
            # ───────────────────────────────
            # Radar Chart (main visual)
            # ───────────────────────────────
            st.subheader("Radar Chart – Multi-City Comparison")
            
            fig_radar = go.Figure()

            for city in compare_cities:
                row = df_comp[df_comp["city"] == city]
                if not row.empty:
                    values = row[indicator_cols].values.flatten().tolist()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]],  # close the loop
                        theta=list(indicator_labels.values()) + [list(indicator_labels.values())[0]],
                        fill='toself',
                        name=city,
                        line=dict(width=2),
                        opacity=0.75
                    ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=True),
                    angularaxis=dict(showticklabels=True, tickfont_size=11)
                ),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                height=600,
                title="Normalized Indicator Scores – Radar View",
                margin=dict(l=40, r=40, t=80, b=120),
                font=dict(size=12)
            )

            st.plotly_chart(fig_radar, use_container_width=True)

            st.caption(
                "**How to read the radar chart**\n"
                "• Each axis = one indicator (higher = better)\n"
                "• Larger area = more balanced/strong overall readiness\n"
                "• Spikes outward = strengths; inward dips = weaknesses"
            )

            # ───────────────────────────────
            # Parallel Coordinates (alternative view)
            # ───────────────────────────────
            st.subheader("Parallel Coordinates – Trade-offs View")
            fig_pc = px.parallel_coordinates(
                df_comp,
                color="ERI",
                labels={col: indicator_labels.get(col, col) for col in indicator_cols},
                color_continuous_scale=px.colors.sequential.Viridis,
                height=500
            )
            fig_pc.update_traces(line=dict(width=3))
            fig_pc.update_layout(
                title="Parallel Coordinates Plot – Indicator Trade-offs",
                margin=dict(l=80, r=80, t=80, b=80)
            )
            st.plotly_chart(fig_pc, use_container_width=True)

            st.caption(
                "**Parallel coordinates explained**\n"
                "• Each vertical line = one indicator\n"
                "• Each colored line = one city\n"
                "• Higher ERI cities tend to have lines higher up on most axes"
            )

            # ───────────────────────────────
            # Comparison Table
            # ───────────────────────────────
            st.subheader("Detailed Scores Table")
            st.dataframe(
                df_comp[["city"] + indicator_cols + ["ERI"]].round(3),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "city": st.column_config.TextColumn("City", width="medium"),
                    "ERI": st.column_config.NumberColumn("ERI", format="%.3f", width="small")
                }
            )

            st.caption("Tip: Sort columns by clicking headers. Higher values = better performance.")
        else:
            st.info("No data found for selected cities – check city_metrics.csv")
    else:
        st.info("Select at least two cities above to compare.")


# =============================================================================
# TAB 3 – Single City Deep Dive
# =============================================================================
with tab3:
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
