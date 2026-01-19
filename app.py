import json
from statistics import mean

# ... later, in your map loop for each city:
if os.path.exists(geojson_files[city]) and os.path.exists(boundary_files[city]):
    with open(geojson_files[city], 'r', encoding='utf-8') as f:
        gj_features = json.load(f)
    with open(boundary_files[city], 'r', encoding='utf-8') as f:
        gj_boundary = json.load(f)

    # Compute a simple centroid from the boundary bbox (works for Polygons/MultiPolygons)
    def geom_coords(g):
        # recursively extract all lon/lat coordinate pairs
        out = []
        def rec(coords):
            if isinstance(coords[0], (float, int)):
                out.append(coords)
            else:
                for c in coords:
                    rec(c)
        rec(g['coordinates'])
        return out

    # get first feature geometry for centroid fallback
    bgeom = gj_boundary['features'][0]['geometry']
    coords = geom_coords(bgeom)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    center = [mean(lats), mean(lons)]

    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    folium.GeoJson(gj_boundary, style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 2}).add_to(m)

    # add clustered features, use feature properties for popup
    marker_cluster = MarkerCluster().add_to(m)
    for feat in gj_features.get('features', []):
        folium.GeoJson(feat, popup=str(feat.get('properties', {}).get('name', ''))).add_to(marker_cluster)

    st_folium(m, width=400, height=400)
else:
    st.warning(f"GeoJSON files for {city} not found.")
    st.warning(f"GeoJSON files for {city} not found.")
