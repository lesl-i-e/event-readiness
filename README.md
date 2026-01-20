# Urban Event Readiness Index (UERI)

**A transparent, data-driven composite index to assess urban infrastructure readiness for hosting large-scale international events in African cities**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-January%20202026-success)](#)

## Project Overview

Large-scale international events (sports tournaments, summits, exhibitions) place significant pressure on host cities in terms of infrastructure, mobility, public services, and safety. Host city selection is often influenced by political or qualitative considerations, while quantitative, transparent readiness assessments remain limited—particularly in the East African context.

This project develops an **Urban Event Readiness Index (UERI)** to objectively assess and compare the readiness of selected cities to host major international events using **open, reproducible data sources**.

The index is **not event-specific**. Instead, it evaluates baseline urban capacity that can later be interpreted in the context of events such as AFCON, CHAN, UN meetings, or other large-scale gatherings.

### Objectives

1. Construct a composite readiness index based on infrastructure, health, mobility, accommodation, and urban capacity indicators.
2. Compare selected East African cities against each other and against an external benchmark city (Casablanca).
3. Identify strengths, weaknesses, and bottlenecks in urban readiness across cities.
4. Ensure transparency and interpretability through clear indicator definitions, normalization, and weighting.
5. Assess the robustness of rankings through sensitivity analysis.

### Study Area

The analysis focuses on the following cities:

- **Nairobi** (Kenya)
- **Kampala** (Uganda)
- **Dar es Salaam** (Tanzania)
- **Kigali** (Rwanda)
- **Casablanca** (Morocco – external benchmark)

(Note: Rabat was considered but not included in final analysis; Mombasa is a potential future addition.)

## Data Sources

All data used in this project are publicly available and reproducible.

| Category                  | Source                              | Link / Access Point                                                                 | Format(s)          | Notes                                                                 |
|---------------------------|-------------------------------------|-------------------------------------------------------------------------------------|--------------------|-----------------------------------------------------------------------|
| Administrative Boundaries | GADM v4.1                           | [https://gadm.org/download_country.html](https://gadm.org/download_country.html)   | Shapefile / GeoJSON | Used to define consistent city extents and calculate city-level areas |
| Administrative Boundaries | Humanitarian Data Exchange (HDX)    | [https://data.humdata.org/](https://data.humdata.org/)                              | Shapefile / GeoJSON | Alternative source for more detailed or updated boundaries            |
| OpenStreetMap Features    | Geofabrik country extracts          | [https://download.geofabrik.de/](https://download.geofabrik.de/)                    | .osm.pbf           | Extracted roads, health, hotels, stadiums, public spaces, etc.        |
| Population Density        | WorldPop (2020 UN-adjusted)         | [https://hub.worldpop.org/](https://hub.worldpop.org/)                              | GeoTIFF (100m/1km) | Used for density and per-capita normalization                         |
| Nighttime Lights (proxy)  | NASA VIIRS / EOG Black Marble       | [https://eogdata.mines.edu/products/vnl/](https://eogdata.mines.edu/products/vnl/) | GeoTIFF            | Contextual urban activity & service proxy                             |
| National Indicators       | World Bank World Development Indicators | [https://databank.worldbank.org/source/world-development-indicators](https://databank.worldbank.org/source/world-development-indicators) | CSV                | Health beds, electricity access, etc. (national-level context)        |

## Indicators Used

The final index uses a restricted, non-redundant set of indicators:

| Dimension          | Indicator                                      | Computation Method                          | Directionality |
|--------------------|------------------------------------------------|---------------------------------------------|----------------|
| Mobility           | Road density (km/km²)                          | Total road length / city area               | Higher = better |
| Mobility           | Intersection density                           | Number of intersections / km²               | Higher = better |
| Health & Safety    | Health facilities per 100,000 population       | Count of hospitals/clinics / population     | Higher = better |
| Urban Capacity     | Population density                             | WorldPop raster sum / city area             | Contextual     |
| Urban Capacity     | Open space per 100,000 population              | Area of public spaces / population          | Higher = better |
| Accommodation      | Hotels per 100,000 population                  | Count of hotels/guest houses / population   | Higher = better |
| Connectivity       | Distance to nearest airport (km)               | Euclidean distance from city centroid       | Lower = better  |

All indicators are **min-max normalized** (0–1 scale), with directionality handled (e.g., lower airport distance = higher score).

## Methodology Summary

1. **Feature Extraction**  
   OSM features extracted per city using consistent tag filters via `osmnx` and `geopandas`.

2. **Indicator Computation**  
   Raw spatial features converted into density or per-capita indicators.

3. **Normalization**  
   Min–max scaling applied; directionality flipped where necessary.

4. **Index Construction**  
   Composite ERI = weighted average of normalized indicators (default: equal weights).

5. **Sensitivity Analysis**  
   Alternative weighting schemes tested; rank stability assessed.

## Visualization & Application

The project includes:

- Comparative bar charts of city readiness scores
- Multi-city radar plots for indicator profiles
- Parallel coordinates plots for trade-offs
- Single-city detailed bar charts + infrastructure maps
- Interactive Streamlit application with:
  - Custom weighting scenarios
  - City rankings
  - Indicator breakdowns
  - Single-city deep dive (with map)

Live demo (if deployed): [Add Streamlit Cloud / Render link here when available]

## Technologies Used

- Python 3.10+
- GeoPandas, OSMnx
- Pandas / NumPy
- Plotly (interactive charts)
- Folium + streamlit-folium (maps)
- Streamlit (web app)

## Installation & Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/lesl-i-e-/event-readiness.git
   cd event-readiness
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

Data sources retain their original licenses (mostly open / CC-BY or equivalent).

## Author

**Gedion Leslie Kweya**  
Data Science Student  
Nairobi, Kenya  
<a href="https://x.com/lesl_i_e_">
  <img src="https://img.shields.io/badge/X-%40lesl_i_e_-1DA1F2?style=for-the-badge&logo=x&logoColor=white" alt="X Profile">
</a>

## Acknowledgments

Special thanks to the following for making this project possible:

- **OpenStreetMap contributors** — for the foundational geospatial data
- **WorldPop team** — for high-resolution population datasets
- **GADM & HDX data providers** — for reliable administrative boundaries
- **NASA Earth Observation Group (VIIRS)** — for nighttime lights and urban proxy data
- **World Bank Open Data team** — for national development indicators

Built with ❤️ in Nairobi, January 2026
