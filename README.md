# GeoAgent: Agentic Earth Observation & Climate Intelligence Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-geoagent.rafiqflow.com-success.svg)](https://geoagent.rafiqflow.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph%20StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![GEE](https://img.shields.io/badge/Engine-Google%20Earth%20Engine-green.svg)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GeoAgent is an **agentic GIS platform** connecting human natural-language intentions with planetary Earth Observation data processing. Built on a compiled **LangGraph `StateGraph`** agent, GeoAgent orchestrates Google Earth Engine (GEE), OpenStreetMap Nominatim, and USGS Landsat 8/9 thermal radiative physics while mitigating spatial hallucinations through deterministic boundary grounding.

> 🌐 **Live Demo:** Try GeoAgent live at [https://geoagent.rafiqflow.com](https://geoagent.rafiqflow.com)

---

## 🌟 Key Capabilities

1. **Stateful LangGraph Agent Orchestration:**
   - Compiled `StateGraph` with deterministic conditional branching across intent reasoning, spatial boundary resolution, parameter clarification, multi-candidate disambiguation, and satellite pipeline dispatch.
2. **Dual-Engine LLM Spatial Reasoning:**
   - Primary: OpenAI-compatible **9Router Gateway** (`ds/deepseek-chat`).
   - Fallback: **Google Gemini API** (`gemini-2.5-flash` / `gemini-1.5-flash`) with bounded latency timeouts and robust regex tokenization safeguards.
3. **Hierarchy-Aware Spatial Disambiguation:**
   - Intercepts ambiguous geographic queries (e.g., *"analyze temperature of Tanjungsari"* or *"Bandung"*).
   - Queries OpenStreetMap Nominatim with administrative regency-level deduplication (`addressdetails=1`, `limit: 25`).
   - Renders interactive clickable choice chips in the UI for instant selection (e.g., Tanjungsari Sumedang vs. Bogor vs. Gunungkidul).
4. **Smart Climate-Adaptive Temporal Intelligence:**
   - Automatically determines optimal cloud-free seasons based on latitude:
     - **Tropical Zone ($|\text{Lat}| < 23.5^\circ$):** Dry Season (June–September) for minimal cloud cover.
     - **Northern Temperate ($\text{Lat} \ge 23.5^\circ$):** Summer (June–August) for peak thermal dynamics.
     - **Southern Temperate ($\text{Lat} \le -23.5^\circ$):** Summer (December–February).
   - Merges USGS Landsat 8 and Landsat 9 Tier 1 collections to double observation frequency and reduce cloud dropouts.
5. **Tri-Tool Scientific Routing:**
   - The LangGraph agent reasons over user spatial intent and dispatches verified scientific tools:
     - `tool_lst_pipeline`: Computes single-channel absolute Land Surface Temperature (in °C) via TIRS Band 10 radiative transfer, NDVI vegetation fraction, and Sobrino emissivity formulations ($\varepsilon = 0.004 \cdot P_v + 0.986$).
     - `tool_suhi_utfvi_pipeline`: Derives the **Urban Thermal Field Variance Index (UTFVI)** ($\text{UTFVI} = \frac{T_s - T_{\text{mean}}}{T_s}$) across 6 ecological comfort zones (None, Weak, Moderate, Strong, Stronger, Strongest), computing the exact percentage of urban heat island thermal stress areas ($\text{UTFVI} \ge 0.010$).
     - `tool_timeseries_pipeline`: Extracts scene-by-scene satellite overpass observations (`YYYY-MM-DD`), masking water bodies (MNDWI/NDVI) and computing fractional-time Ordinary Least Squares (OLS) warming rates (°C/year) transmitted to an interactive trend panel.
6. **Transparent Execution Audit Trace:**
   - Generates collapsible execution audit cards detailing tool dispatch, satellite scene counts, cloud thresholds, mathematical formulas, and processing latency.
7. **Dual-Panel Interactive UI with Slide Analytics Panel:**
   - Modern glassmorphism chat panel paired with a full-height Leaflet dark-canvas map, dynamic dual-palette tile rendering (LST heat gradient vs. 6-class UTFVI comfort ramp), floating statistical HUD, and an animated **Right Analytics Slide Panel** powered by **Chart.js**.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- Google Earth Engine authenticated account (`earthengine authenticate`)

### 2. Installation
```bash
git clone https://github.com/rafiqns/geoagent.git
cd geoagent
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
# Google Earth Engine
EE_PROJECT_ID=ee-your-project-id

# 9Router LLM Gateway (Primary)
NINEROUTER_API_KEY=your_9router_key
NINEROUTER_BASE_URL=https://your-9router-url.com/v1
NINEROUTER_MODEL=ds/deepseek-chat

# Google AI / Gemini API (Fallback)
GOOGLE_API_KEY=your_gemini_api_key

# Server
PORT=8088
HOST=127.0.0.1
```

### 4. Run Locally
```bash
python main.py
```
Or directly with Uvicorn:
```bash
uvicorn main:app --host 127.0.0.1 --port 8088 --reload
```

### 5. Run with Docker (Recommended for Containerized Environments)
```bash
docker compose up --build
```
Open your browser at: **`http://127.0.0.1:8088`**

### 6. Run Unit Test Suite
```bash
python test_geoagent.py
```
*(12 automated tests verifying LangGraph reasoning, GEE LST thermal physics, SUHI UTFVI tool dispatch, multi-temporal overpass timeseries regression, spatial disambiguation, and FastAPI endpoints).*

---

## 📁 Repository Structure

```
geoagent/
├── Dockerfile             # Multi-stage production container configuration
├── docker-compose.yml     # Container orchestration specification
├── gee_engine.py          # GEE Landsat thermal processing & OSM boundary resolver
├── agent.py               # LangGraph StateGraph agent & spatial reasoning pipeline
├── main.py                # FastAPI application & REST endpoint router
├── test_geoagent.py       # Automated unit test suite (12 test cases)
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment template
├── .gitignore             # Secrets & temp files filter
├── LICENSE                # MIT License
└── static/
    ├── index.html         # Dual-panel conversational GIS interface
    ├── style.css          # Modern dark glassmorphism stylesheet & disambiguation chips
    └── app.js             # Leaflet layer controller & Chart.js renderer
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
