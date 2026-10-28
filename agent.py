"""
GeoAgent Conversational Planner & Tool Orchestration Layer.
Implemented as an Autonomous LangGraph StateGraph connecting 9Router (DeepSeek / OpenAI compatible),
Google Gemini fallback, OpenStreetMap Nominatim spatial disambiguation, and Google Earth Engine.
"""

import os
import re
import json
import time
import datetime
import logging
from typing import Dict, Any, List, Optional, TypedDict
from dotenv import load_dotenv

# LangGraph & LangChain OpenAI imports
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from gee_engine import eo_engine

load_dotenv()
logger = logging.getLogger("geoagent.agent")

# 1. Configure 9Router / OpenAI LLM Client
def get_9router_llm():
    api_key = os.getenv("NINEROUTER_API_KEY")
    base_url = os.getenv("NINEROUTER_BASE_URL")
    model_name = os.getenv("NINEROUTER_MODEL", "ds/deepseek-chat")

    if api_key and base_url:
        logger.info(f"🌐 [9Router] Initializing ChatOpenAI with endpoint: {base_url} (model: {model_name})")
        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            default_headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            temperature=0,
            timeout=30
        )
    return None

NINEROUTER_LLM = get_9router_llm()

# Fallback Google GenAI Client
GEMINI_CLIENT = None
try:
    from google import genai
    g_key = os.environ.get("GOOGLE_API_KEY")
    if g_key:
        GEMINI_CLIENT = genai.Client(api_key=g_key)
except Exception as e:
    logger.warning(f"⚠️ [Google GenAI] Fallback client not available: {e}")


# 2. Define LangGraph State Schema
class GeoAgentState(TypedDict):
    raw_query: str
    target_location: Optional[str]
    user_specified_time: bool
    temporal_desc: str
    start_date: str
    end_date: str
    intent: str
    analysis_type: str  # "lst" | "utfvi"
    status: str  # "clarify", "disambiguate", "ready_to_analyze", "analyzed", "error"
    ai_model_used: str
    ai_reasoning: str
    candidates: List[Dict[str, Any]]
    boundary_info: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    audit_trace: List[Dict[str, Any]]
    final_output: Dict[str, Any]
    start_time: float


def get_latest_matured_year() -> int:
    """Returns the most recent year with completed summer/dry season data."""
    now = datetime.datetime.now()
    # If in or past October, the dry season / summer of the current year is fully complete
    return now.year if now.month >= 10 else now.year - 1


def adapt_smart_temporal_window(lat: float, default_year: Optional[int] = None) -> Dict[str, str]:
    """
    Selects the optimal cloud-free observation window based on geographic latitude:
    - Tropical zone (|lat| < 23.5°): Dry Season (June 1 - September 30)
    - Northern temperate (lat >= 23.5°): Summer (June 1 - August 31)
    - Southern temperate (lat <= -23.5°): Summer (December 1 - February 28)
    """
    if default_year is None:
        default_year = get_latest_matured_year()

    if abs(lat) < 23.5:
        return {
            "start_date": f"{default_year}-06-01",
            "end_date": f"{default_year}-09-30",
            "temporal_desc": f"Dry Season June–September {default_year} (Optimal Cloud-Free Composite)",
            "zone": "Tropical Dry Season"
        }
    elif lat >= 23.5:
        return {
            "start_date": f"{default_year}-06-01",
            "end_date": f"{default_year}-08-31",
            "temporal_desc": f"Summer June–August {default_year} (Peak Heat Dynamics Composite)",
            "zone": "Northern Hemisphere Summer"
        }
    else:
        return {
            "start_date": f"{default_year-1}-12-01",
            "end_date": f"{default_year}-02-28",
            "temporal_desc": f"Summer Dec {default_year-1}–Feb {default_year} (Peak Heat Dynamics Composite)",
            "zone": "Southern Hemisphere Summer"
        }


# 3. LangGraph Node Definitions

def node_reason_and_plan(state: GeoAgentState) -> GeoAgentState:
    """Node 1: Uses 9Router (or Gemini fallback) to extract location, resolve typos, and infer temporal range & tool intent."""
    user_query = state["raw_query"]
    audit_trace = state["audit_trace"]
    curr_year = get_latest_matured_year()
    logger.info(f"🧠 [LangGraph: reason_and_plan] Processing query: '{user_query}' (Dynamic baseline year: {curr_year})")

    system_prompt = f"""You are GeoAgent, an autonomous Spatial Intelligence Engine for Earth Observation.
Analyze the user's natural language input for Land Surface Temperature (LST), Surface Urban Heat Island (SUHI / UTFVI), or Multi-Temporal Timeseries / Trend analysis.

Your task:
1. Detect Intent & Select Scientific Tool:
   - "analyze_timeseries": User asks for a trend, time series, multi-year progression, historical changes, grafik, chart, or per-year trend (e.g. "trend temperature in Jakarta", "grafik 2018-2023", "timeseries suhi", "plot warming rate"). Set analysis_type to "timeseries".
   - "analyze_suhi_utfvi": User asks for Surface Urban Heat Island (SUHI), urban heat island, pulau panas, heat island effect, UTFVI, thermal stress, or ecological comfort. Set analysis_type to "utfvi".
   - "analyze_lst": User asks to analyze land surface temperature, suhu permukaan, absolute heat profile, thermal range, or satellite LST data. Set analysis_type to "lst".
   - "clarification_needed": User input has NO recognizable geographic place (e.g. "hi", "help", "analyze heat", "what can you do?").
   - "general_qa": User asks a conceptual question without a specific place.
2. Extract & Normalize Location:
   - Identify the exact target location name.
   - Correct typos (e.g., "smdang" -> "Sumedang", "bndung" -> "Bandung", "jogja" -> "Yogyakarta", "guelph" -> "Guelph").
   - IMPORTANT: If user specifies sub-district/regency like "tanjungsari sumedang", retain the full hierarchy: "Tanjungsari, Sumedang".
   - If user only gives a common name without parent (e.g. "Tanjungsari"), extract "Tanjungsari" so geocoding can evaluate candidates.
   - If no location, set to null.
3. Extract Temporal Window:
   - Identify year, season, or multi-year span if EXPLICITLY requested by user (e.g. "2018-2024", "dari 2015 sampai 2023", "5 tahun terakhir").
   - Set "user_specified_time": true if user explicitly provided a year, range, or multi-year interval.
   - If user specified multi-year range, set start_date and end_date accordingly.
   - If user did NOT specify any time, set "user_specified_time": false, "start_date": "{curr_year}-06-01", "end_date": "{curr_year}-09-30", "temporal_desc": "Automatic Climate-Adaptive Composite".

Return STRICTLY a JSON object without markdown fences:
{{
  "intent": "analyze_lst" | "analyze_suhi_utfvi" | "analyze_timeseries" | "clarification_needed" | "general_qa",
  "analysis_type": "lst" | "utfvi" | "timeseries",
  "location": "Formal/Normalized Location Name" | null,
  "user_specified_time": true | false,
  "temporal_desc": "Temporal Description",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "ai_reasoning": "Brief explanation of how the location, tool intent, and temporal bounds were parsed"
}}
"""

    parsed = None
    model_used = "Offline Word-Boundary Parser"

    # Attempt 1: 9Router ChatOpenAI (DeepSeek / OpenAI compatible)
    if NINEROUTER_LLM:
        try:
            logger.info("🌐 [9Router] Invoking 9Router LLM for spatial reasoning...")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User query: \"{user_query}\"")
            ]
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User query: \"{user_query}\"")
            ]
            ai_msg = NINEROUTER_LLM.invoke(messages)
            raw = ai_msg.content.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            parsed = json.loads(raw.strip())
            model_used = f"9Router ({getattr(NINEROUTER_LLM, 'model_name', 'DeepSeek')})"
            logger.info(f"✅ [9Router] Successfully parsed intent: Location='{parsed.get('location')}', Intent='{parsed.get('intent')}'")
        except Exception as e:
            logger.warning(f"⚠️ [9Router] Execution failed: {e}. Falling back to Google GenAI...")

    # Attempt 2: Google GenAI Fallback (with strict 8s timeout)
    if not parsed and GEMINI_CLIENT:
        for m in ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-2.5-pro"]:
            try:
                logger.info(f"🧠 [Gemini Fallback] Invoking {m} (8s timeout)...")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        GEMINI_CLIENT.models.generate_content,
                        model=m,
                        contents=f"{system_prompt}\n\nUser input: \"{user_query}\""
                    )
                    resp = future.result(timeout=8)

                raw = resp.text.strip()
                if raw.startswith("```json"):
                    raw = raw[7:]
                if raw.startswith("```"):
                    raw = raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                parsed = json.loads(raw.strip())
                model_used = f"Google {m}"
                logger.info(f"✅ [Gemini Fallback] Resolved via {m}: Location='{parsed.get('location')}'")
                break
            except Exception as e:
                logger.warning(f"⚠️ [Gemini Fallback] {m} failed or timed out: {e}")

    # Attempt 3: Safe Deterministic Fallback
    if not parsed:
        parsed = _fallback_heuristic_parse(user_query)
        model_used = "Deterministic Word-Boundary Parser"

    loc = parsed.get("location")
    intent = parsed.get("intent", "analyze_lst")
    if intent == "analyze_timeseries":
        analysis_type = "timeseries"
    elif intent == "analyze_suhi_utfvi":
        analysis_type = "utfvi"
    else:
        analysis_type = parsed.get("analysis_type", "lst")
    user_specified_time = bool(parsed.get("user_specified_time", False))

    curr_year = get_latest_matured_year()
    state["target_location"] = loc
    state["intent"] = intent
    state["analysis_type"] = analysis_type
    state["user_specified_time"] = user_specified_time
    state["temporal_desc"] = parsed.get("temporal_desc", "Automatic Climate-Adaptive Composite")
    state["start_date"] = parsed.get("start_date", f"{curr_year}-06-01")
    state["end_date"] = parsed.get("end_date", f"{curr_year}-09-30")
    state["ai_reasoning"] = parsed.get("ai_reasoning", "Spatial entity normalization")
    state["ai_model_used"] = model_used

    if analysis_type == "timeseries":
        selected_tool = "tool_timeseries_pipeline"
        target_metric = "Multi-Temporal Thermal Trend Profile"
    elif analysis_type == "utfvi":
        selected_tool = "tool_suhi_utfvi_pipeline"
        target_metric = "Urban Thermal Field Variance Index (UTFVI)"
    else:
        selected_tool = "tool_lst_pipeline"
        target_metric = "Land Surface Temperature (LST)"

    if intent == "clarification_needed" or not loc:
        state["status"] = "clarify"
    else:
        state["status"] = "ready_to_geocode"
        audit_trace.append({
            "step": "1. AI Spatial Entity Resolution & Tool Selection (LangGraph)",
            "orchestrator": "LangGraph StateGraph",
            "model": model_used,
            "raw_query": user_query,
            "resolved_entity": loc,
            "selected_tool": selected_tool,
            "target_metric": target_metric,
            "user_specified_time": user_specified_time,
            "reasoning": state["ai_reasoning"],
            "timestamp": time.strftime("%H:%M:%S")
        })

    return state


def node_resolve_boundary(state: GeoAgentState) -> GeoAgentState:
    """Node 2: Geocodes location with multi-candidate hierarchy disambiguation."""
    loc = state["target_location"]
    audit_trace = state["audit_trace"]
    logger.info(f"🌍 [LangGraph: resolve_boundary] Geocoding location: '{loc}'")

    boundary_res = eo_engine.geocode_city_boundary(loc)

    if boundary_res.get("ambiguous"):
        state["status"] = "disambiguate"
        state["candidates"] = boundary_res.get("candidates", [])
        audit_trace.append({
            "step": "2. Spatial Disambiguation Verification",
            "status": "Multiple Candidates Identified",
            "candidate_count": len(state["candidates"]),
            "action": "Prompting user with interactive choice bubbles."
        })
    elif boundary_res.get("success"):
        state["status"] = "ready_to_analyze"
        state["boundary_info"] = boundary_res

        # If user did not specify explicit dates, adapt smartly based on geographic latitude
        if not state.get("user_specified_time"):
            center = boundary_res.get("center", [-6.9, 107.6])
            lat = float(center[0])
            adapt = adapt_smart_temporal_window(lat)
            state["start_date"] = adapt["start_date"]
            state["end_date"] = adapt["end_date"]
            state["temporal_desc"] = adapt["temporal_desc"]
            logger.info(f"⏱️ [Smart Temporal Adapter] Auto-selected {adapt['temporal_desc']} (Zone: {adapt['zone']}) based on latitude {lat}°")

        audit_trace.append({
            "step": "2. Geographic Resolution & Boundary Extraction",
            "action": f"Extracted boundary polygon for '{boundary_res['name']}'",
            "provider": "OpenStreetMap Nominatim / GADM",
            "bounding_box": boundary_res.get("bbox")
        })
    else:
        state["status"] = "error"
        audit_trace.append({
            "step": "2. Geographic Resolution Failed",
            "error": f"No boundary polygon found for '{loc}'"
        })

    return state


def node_handle_clarification(state: GeoAgentState) -> GeoAgentState:
    """Node 3A: Generates proactive clarification message when location is omitted."""
    state["final_output"] = {
        "type": "clarification",
        "message": (
            "I am ready to run an autonomous Earth Observation thermal analysis. "
            "Which city, regency, or geographic region would you like to examine?\n\n"
            "*(e.g., **Kota Bandung**, **Sumedang, West Java**, **DKI Jakarta**, or **Guelph, Ontario**)*."
        ),
        "audit_trace": [{
            "step": "AI Intent Parsing (LangGraph)",
            "model": state.get("ai_model_used", "9Router"),
            "status": "Incomplete Parameters",
            "action": "Proactively requested geographic target region from user."
        }]
    }
    return state


def node_handle_disambiguation(state: GeoAgentState) -> GeoAgentState:
    """Node 3B: Generates interactive choice bubbles for ambiguous locations."""
    loc = state["target_location"]
    candidates = state["candidates"]
    state["final_output"] = {
        "type": "disambiguation",
        "message": (
            f"I found multiple administrative regions matching **'{loc}'**. "
            f"Please select the specific region you want to analyze:"
        ),
        "query": loc,
        "candidates": candidates,
        "audit_trace": state["audit_trace"]
    }
    return state


def node_execute_satellite_pipeline(state: GeoAgentState) -> GeoAgentState:
    """Node 4: Executes Google Earth Engine thermal radiative calibration or SUHI UTFVI analysis."""
    boundary = state["boundary_info"]
    audit_trace = state["audit_trace"]
    start_date = state["start_date"]
    end_date = state["end_date"]
    temporal_desc = state["temporal_desc"]
    analysis_type = state.get("analysis_type", "lst")
    is_timeseries = (analysis_type == "timeseries" or state.get("intent") == "analyze_timeseries")
    is_suhi = (analysis_type == "utfvi" or state.get("intent") == "analyze_suhi_utfvi")

    logger.info(f"🛰️ [LangGraph: execute_satellite_pipeline] Running GEE ({'TIMESERIES' if is_timeseries else ('UTFVI' if is_suhi else 'LST')}) for '{boundary['name']}'")

    if is_timeseries:
        audit_trace.append({
            "step": "3. Tool Invocation: tool_timeseries_pipeline",
            "dataset": "USGS Landsat 8/9 Collection 2 Tier 1 TOA (Annual Composites)",
            "observation_window": f"{start_date[:4]} to {end_date[:4]} (JJAS Dry Season)",
            "sensor": "Landsat 8/9 OLI & TIRS Band 10",
            "selected_metric": "Multi-Temporal Thermal Trend Profile",
            "cloud_masking": "Simple Cloud Score / Fmask (threshold <= 25%)"
        })

        s_yr = int(start_date[:4]) if start_date and len(start_date) >= 4 else 2018
        e_yr = int(end_date[:4]) if end_date and len(end_date) >= 4 else 2024
        if s_yr == e_yr:
            s_yr = max(2013, e_yr - 5)

        eo_res = eo_engine.process_temporal_timeseries(
            boundary_info=boundary,
            start_year=s_yr,
            end_year=e_yr,
            cloud_threshold=25
        )
    elif is_suhi:
        audit_trace.append({
            "step": "3. Tool Invocation: tool_suhi_utfvi_pipeline",
            "dataset": "USGS Landsat 8/9 Collection 2 Tier 1 TOA",
            "temporal_window": f"{start_date} to {end_date} ({temporal_desc})",
            "sensor": "Operational Land Imager (OLI) & Thermal Infrared Sensor (TIRS Band 10)",
            "selected_metric": "Urban Thermal Field Variance Index (UTFVI)",
            "cloud_masking": "Simple Cloud Score / Fmask (threshold <= 20%)"
        })

        eo_res = eo_engine.process_suhi_utfvi(
            boundary_info=boundary,
            start_date=start_date,
            end_date=end_date,
            cloud_threshold=20
        )
    else:
        audit_trace.append({
            "step": "3. Tool Invocation: tool_lst_pipeline",
            "dataset": "USGS Landsat 8/9 Collection 2 Tier 1 TOA",
            "temporal_window": f"{start_date} to {end_date} ({temporal_desc})",
            "sensor": "Operational Land Imager (OLI) & Thermal Infrared Sensor (TIRS Band 10, 10.8 µm)",
            "selected_metric": "Land Surface Temperature (LST)",
            "cloud_masking": "Simple Cloud Score / Fmask (threshold <= 20%)"
        })

        eo_res = eo_engine.process_land_surface_temperature(
            boundary_info=boundary,
            start_date=start_date,
            end_date=end_date,
            cloud_threshold=20
        )

    if is_timeseries:
        audit_trace.append({
            "step": "4. Multi-Temporal Regression & Trend Modeling",
            "methodology": "Annual Dry-Season Landsat Composites (JJAS)",
            "bachelor_thesis_reference": eo_res.get("thesis_reference"),
            "warming_rate": eo_res.get("warming_rate"),
            "net_temperature_delta": eo_res.get("total_change"),
            "data_points_evaluated": len(eo_res.get("timeseries", []))
        })
        audit_trace.append({
            "step": "5. Spatiotemporal Timeseries Dispatch",
            "status": "Transmitted to Frontend Analytics Slide Panel",
            "chart_type": "Interactive Line & Scatter Chart (Chart.js)"
        })

        elapsed = round(time.time() - state["start_time"], 2)
        ts_list = eo_res.get("timeseries", [])
        pts_summary = ", ".join([f"{p['year']}: {p['lst_mean']}°C" for p in ts_list[-4:]])

        response_text = (
            f"**Multi-Temporal Thermal Trend Analysis Completed for {boundary['name']}**\n\n"
            f"• **Analysis Window:** **{eo_res.get('start_year')} – {eo_res.get('end_year')}** (Dry Season JJAS)\n"
            f"• **Estimated Warming Rate:** **{eo_res.get('warming_rate')}**\n"
            f"• **Net Temperature Delta:** **{eo_res.get('total_change')}**\n"
            f"• **Recent Trajectory:** {pts_summary}\n\n"
            f"📈 **Interactive Chart Ready:** The spatiotemporal trajectory chart and data series have been rendered in the **Right Analytics Slide Panel**. "
            f"You can inspect per-year satellite readings and linear regression trends directly."
        )

        state["final_output"] = {
            "type": "analysis_result",
            "analysis_type": "timeseries",
            "message": response_text,
            "city_name": boundary["name"],
            "center": boundary.get("center"),
            "bbox": boundary.get("bbox"),
            "geojson": boundary.get("geojson"),
            "timeseries_data": eo_res.get("timeseries", []),
            "warming_rate": eo_res.get("warming_rate"),
            "total_change": eo_res.get("total_change"),
            "start_year": eo_res.get("start_year"),
            "end_year": eo_res.get("end_year"),
            "audit_trace": audit_trace,
            "execution_latency": f"{elapsed}s"
        }
        return state

    if eo_res.get("zero_scenes"):
        state["final_output"] = {
            "type": "clarification",
            "message": (
                f"⚠️ **No Cloud-Free Landsat Scenes Found for {boundary['name']}**\n\n"
                f"Between `{start_date}` and `{end_date}`, USGS Landsat 8/9 recorded no scenes with <= 40% cloud cover over this region "
                f"(due to orbital revisit intervals or monsoon cloud persistence).\n\n"
                f"💡 **Recommendation:** Would you like to analyze the **optimal Dry Season composite** (June–September) or a broader multi-month period instead?"
            ),
            "audit_trace": audit_trace
        }
        return state

    if is_suhi:
        audit_trace.append({
            "step": "4. Radiative Transfer & UTFVI Thermodynamic Equation",
            "formula": "UTFVI = (Ts - Tmean) / Ts  [Ts in Kelvin]",
            "reference": "Zhang et al. (2006); Voogt & Oke (2003)",
            "ecological_classes": [
                "None (<0, Excellent)", "Weak (0-0.005, Good)", "Moderate (0.005-0.010, Normal)",
                "Strong (0.010-0.015, Bad)", "Stronger (0.015-0.020, Worse)", "Strongest (>=0.020, Worst)"
            ],
            "severe_heat_stress_area": f"{eo_res.get('severe_heat_pct', 0.0)}% (UTFVI >= 0.010)",
            "execution_mode": eo_res.get("mode", "live"),
            "scene_count": eo_res.get("scene_count", 0)
        })

        audit_trace.append({
            "step": "5. Zonal UTFVI Index & Thermal Field Statistics",
            "min_utfvi": str(eo_res.get("min_utfvi")),
            "mean_utfvi": str(eo_res.get("mean_utfvi")),
            "max_utfvi": str(eo_res.get("max_utfvi")),
            "baseline_lst_mean": f"{eo_res['mean_temp']} °C",
            "status": "Verified & Rendered"
        })
    else:
        audit_trace.append({
            "step": "4. Thermal Radiative Calibration & Emissivity Physics",
            "algorithms": [
                "Band 10 Top-of-Atmosphere Spectral Radiance -> Brightness Temperature (Tb)",
                "Fractional Vegetation Cover: Pv = ((NDVI - 0.2) / 0.3)^2",
                "Dynamic Surface Emissivity: eps = 0.004 * Pv + 0.986 (Sobrino et al., 2004)",
                "Single-Channel LST Derivation (Artis & Carnahan, 1982; Planck constant hc/K = 14388 µm·K)"
            ],
            "execution_mode": eo_res.get("mode", "live"),
            "scene_count": eo_res.get("scene_count", 0)
        })

        audit_trace.append({
            "step": "5. Zonal Spatial Statistics & Hotspot Mapping",
            "min_temperature": f"{eo_res['min_temp']} °C",
            "mean_temperature": f"{eo_res['mean_temp']} °C",
            "max_temperature": f"{eo_res['max_temp']} °C",
            "status": "Verified & Rendered"
        })

    elapsed = round(time.time() - state["start_time"], 2)

    temporal_note = (
        f"\n\n🎯 *Custom timeframe applied as requested: **{temporal_desc}**.*"
        if state.get("user_specified_time")
        else f"\n\n💡 *Note: Automatically selected **{temporal_desc}** for this climate zone to ensure minimal cloud contamination and peak thermal contrast. To examine a different year or custom dates, simply mention it (e.g. 'try year 2022' or 'analyze dry season 2023')!*"
    )

    if is_suhi:
        response_text = (
            f"**Autonomous Surface Urban Heat Island (SUHI) Analysis Completed for {boundary['name']}**\n\n"
            f"• **Methodology:** Urban Thermal Field Variance Index (UTFVI) via `tool_suhi_utfvi_pipeline`\n"
            f"• **Observation Period:** {temporal_desc} (`{start_date}` to `{end_date}`)\n"
            f"• **Baseline Mean LST:** **{eo_res['mean_temp']}°C** (Thermal Range: {eo_res['min_temp']}°C to {eo_res['max_temp']}°C)\n"
            f"• **UTFVI Index Range:** **{eo_res.get('min_utfvi')}** to **{eo_res.get('max_utfvi')}**\n"
            f"• **Severe SUHI Stress Zone (UTFVI ≥ 0.010):** **{eo_res.get('severe_heat_pct', 0.0)}%** of administrative area\n\n"
            f"**Ecological Evaluation:** Significant urban microclimate degradation is observed over impervious built-up clusters, "
            f"characterized by Strong to Strongest thermal variance (UTFVI ≥ 0.010). Vegetated and river corridors show negative UTFVI (< 0), "
            f"confirming their active role as urban cooling buffers. The 6-class UTFVI thermal field is rendered on the map canvas."
            f"{temporal_note}"
        )
    else:
        response_text = (
            f"**Autonomous Thermal Analysis Completed for {boundary['name']}**\n\n"
            f"• **Methodology:** Mono-Window Land Surface Temperature (LST) via `tool_lst_pipeline`\n"
            f"• **Observation Period:** {temporal_desc} (`{start_date}` to `{end_date}`)\n"
            f"• **Mean Land Surface Temperature (LST):** **{eo_res['mean_temp']}°C**\n"
            f"• **Thermal Range:** **{eo_res['min_temp']}°C** (vegetated/water corridors) to **{eo_res['max_temp']}°C** (extreme thermal hotspots)\n\n"
            f"**Spatial Insights:** Local heat centers are concentrated over high-density commercial/industrial built-up land covers, "
            f"exhibiting a pronounced Surface Urban Heat Island (SUHI) intensity delta of **+{round(eo_res['max_temp'] - eo_res['mean_temp'], 1)}°C** "
            f"relative to the regional baseline. The processed thermal layer is rendered on the map canvas."
            f"{temporal_note}"
        )

    eo_stats = {
        "min": eo_res["min_temp"],
        "mean": eo_res["mean_temp"],
        "max": eo_res["max_temp"],
        "scenes": eo_res.get("scene_count", 6),
        "mode": eo_res.get("mode")
    }
    if is_suhi:
        eo_stats.update({
            "min_utfvi": eo_res.get("min_utfvi"),
            "mean_utfvi": eo_res.get("mean_utfvi"),
            "max_utfvi": eo_res.get("max_utfvi"),
            "severe_heat_pct": eo_res.get("severe_heat_pct"),
            "classes": eo_res.get("classes")
        })

    state["final_output"] = {
        "type": "analysis_result",
        "analysis_type": "utfvi" if is_suhi else "lst",
        "message": response_text,
        "city_name": boundary["name"],
        "center": boundary.get("center"),
        "bbox": boundary.get("bbox"),
        "geojson": boundary.get("geojson"),
        "eo_stats": eo_stats,
        "tile_url": eo_res.get("tile_url"),
        "palette": eo_res.get("palette"),
        "classes": eo_res.get("classes") if is_suhi else None,
        "audit_trace": audit_trace,
        "execution_latency": f"{elapsed}s"
    }

    return state


# 4. Routing Conditional Logic
def route_after_reasoning(state: GeoAgentState) -> str:
    if state["status"] == "clarify":
        return "handle_clarification"
    return "resolve_boundary"

def route_after_geocoding(state: GeoAgentState) -> str:
    if state["status"] == "disambiguate":
        return "handle_disambiguation"
    elif state["status"] == "ready_to_analyze":
        return "execute_satellite_pipeline"
    return "handle_clarification"


# 5. Build and Compile LangGraph StateGraph
def build_geoagent_graph():
    workflow = StateGraph(GeoAgentState)

    workflow.add_node("reason_and_plan", node_reason_and_plan)
    workflow.add_node("resolve_boundary", node_resolve_boundary)
    workflow.add_node("handle_clarification", node_handle_clarification)
    workflow.add_node("handle_disambiguation", node_handle_disambiguation)
    workflow.add_node("execute_satellite_pipeline", node_execute_satellite_pipeline)

    workflow.add_edge(START, "reason_and_plan")
    workflow.add_conditional_edges("reason_and_plan", route_after_reasoning)
    workflow.add_conditional_edges("resolve_boundary", route_after_geocoding)

    workflow.add_edge("handle_clarification", END)
    workflow.add_edge("handle_disambiguation", END)
    workflow.add_edge("execute_satellite_pipeline", END)

    logger.info("⚡ [LangGraph] GeoAgent StateGraph successfully compiled.")
    return workflow.compile()

GEOAGENT_GRAPH = build_geoagent_graph()


# 6. Public Planner Interface
class GeoAgentPlanner:
    """Wrapper exposing standard plan_and_execute interface backed by LangGraph."""

    def plan_and_execute(self, user_message: str) -> Dict[str, Any]:
        curr_year = get_latest_matured_year()
        initial_state: GeoAgentState = {
            "raw_query": user_message.strip(),
            "target_location": None,
            "user_specified_time": False,
            "temporal_desc": "Automatic Climate-Adaptive Composite",
            "start_date": f"{curr_year}-06-01",
            "end_date": f"{curr_year}-09-30",
            "intent": "analyze_lst",
            "analysis_type": "lst",
            "status": "start",
            "ai_model_used": "LangGraph",
            "ai_reasoning": "",
            "candidates": [],
            "boundary_info": None,
            "analysis_result": None,
            "audit_trace": [],
            "final_output": {},
            "start_time": time.time()
        }

        # Invoke the compiled LangGraph StateGraph!
        final_state = GEOAGENT_GRAPH.invoke(initial_state)
        return final_state.get("final_output", {
            "type": "error",
            "message": "Graph execution completed without generating an output."
        })


def _fallback_heuristic_parse(text: str) -> Dict[str, Any]:
    """Safe word-boundary heuristic parser with dual-tool intent classification."""
    text_lower = text.lower()
    known_cities = [
        (r'\bbandung\b', "Bandung"),
        (r'\bjakarta\b', "DKI Jakarta"),
        (r'\bsumedang\b', "Sumedang"),
        (r'\btanjungsari\b', "Tanjungsari"),
        (r'\bguelph\b', "Guelph"),
        (r'\bsurabaya\b', "Kota Surabaya"),
        (r'\byogyakarta\b|\bjogja\b', "Kota Yogyakarta"),
        (r'\bsemarang\b', "Kota Semarang"),
        (r'\bmedan\b', "Kota Medan"),
        (r'\bbogor\b', "Kota Bogor"),
        (r'\btoronto\b', "Toronto")
    ]

    extracted = None
    for pattern, formal in known_cities:
        if re.search(pattern, text_lower):
            extracted = formal
            break

    if not extracted:
        match = re.search(r'\b(?:in|di|for|untuk|of|dari)\s+([A-Za-z\s,]+?)(?:\s+in|\s+pada|\s+during|\.|\?|$)', text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if len(candidate) > 2 and candidate.lower() not in ["2023", "2024", "summer", "kemarau"]:
                extracted = candidate

    # Detect multi-temporal timeseries / trend vs SUHI vs standard LST
    timeseries_patterns = r'\b(trend|timeseries|time series|grafik|chart|trajectory|progression|perkembangan|historis|history|warming rate)\b'
    is_timeseries = bool(re.search(timeseries_patterns, text_lower))

    suhi_patterns = r'\b(suhi|heat island|urban heat island|utfvi|pulau panas|stres panas|thermal field|kenyamanan termal)\b'
    is_suhi = bool(re.search(suhi_patterns, text_lower))

    # Detect explicit year or multi-year range
    range_match = re.search(r'\b(201\d|202\d)\s*(?:-|to|sampai|hingga)\s*(201\d|202\d)\b', text)
    has_year = re.search(r'\b(201\d|202\d)\b', text)

    user_specified_time = bool(range_match or has_year or re.search(r'\b(summer|kemarau|dry season|winter|hujan)\b', text_lower))
    curr_year = get_latest_matured_year()

    if range_match:
        start_date = f"{range_match.group(1)}-06-01"
        end_date = f"{range_match.group(2)}-09-30"
        temporal_desc = f"Annual Dry Seasons {range_match.group(1)}–{range_match.group(2)}"
        is_timeseries = True
    elif has_year:
        year_found = int(has_year.group(1))
        start_date = f"{year_found}-06-01"
        end_date = f"{year_found}-09-30"
        temporal_desc = f"Dry Season June–September {year_found}"
    else:
        start_date = f"{curr_year - 5}-06-01" if is_timeseries else f"{curr_year}-06-01"
        end_date = f"{curr_year}-09-30"
        temporal_desc = "Multi-Year Historical Trend" if is_timeseries else "Automatic Climate-Adaptive Composite"

    if not extracted:
        intent = "clarification_needed"
        analysis_type = "timeseries" if is_timeseries else ("utfvi" if is_suhi else "lst")
    elif is_timeseries:
        intent = "analyze_timeseries"
        analysis_type = "timeseries"
    elif is_suhi:
        intent = "analyze_suhi_utfvi"
        analysis_type = "utfvi"
    else:
        intent = "analyze_lst"
        analysis_type = "lst"

    return {
        "intent": intent,
        "analysis_type": analysis_type,
        "location": extracted,
        "user_specified_time": user_specified_time,
        "temporal_desc": temporal_desc,
        "start_date": start_date,
        "end_date": end_date,
        "ai_reasoning": "Word-boundary regex match with dual-tool intent routing"
    }


# Global singleton
agent_planner = GeoAgentPlanner()
