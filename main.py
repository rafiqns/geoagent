"""
GeoAgent FastAPI Application.
Hosts conversational spatial endpoints, preset demonstrations,
and serves the dual-panel interactive Earth observation canvas.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure unified application logger format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("geoagent.main")

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from agent import agent_planner

app = FastAPI(
    title="GeoAgent: Autonomous Earth Observation & Spatial Intelligence",
    description="Conversational agent orchestrating planetary Earth Engine processing, thermal physics, and spatial analytics.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class PresetScenario(BaseModel):
    id: str
    title: str
    description: str
    prompt: str
    city: str


PRESET_SCENARIOS = [
    {
        "id": "bandung_heat",
        "title": "🌡️ Bandung LST",
        "description": "Analyze surface temperature & hotspots via tool_lst_pipeline.",
        "prompt": "Analyze land surface temperature in Kota Bandung during the dry season.",
        "city": "Kota Bandung"
    },
    {
        "id": "jakarta_suhi",
        "title": "🏙️ Jakarta SUHI (UTFVI)",
        "description": "Urban Heat Island & thermal variance via tool_suhi_utfvi_pipeline.",
        "prompt": "Analyze Surface Urban Heat Island (SUHI) and UTFVI thermal stress in DKI Jakarta.",
        "city": "DKI Jakarta"
    },
    {
        "id": "jakarta_trend",
        "title": "📈 Jakarta 2018–2024 Trend",
        "description": "Multi-temporal warming rate & chart via tool_timeseries_pipeline.",
        "prompt": "Show multi-temporal temperature trend in Jakarta from 2018 to 2024.",
        "city": "DKI Jakarta"
    },
    {
        "id": "sumedang_suhi",
        "title": "🌿 Sumedang SUHI",
        "description": "Ecological thermal field variance index in Sumedang.",
        "prompt": "Evaluate urban heat island and UTFVI index in Sumedang.",
        "city": "Sumedang"
    },
    {
        "id": "guelph_summer",
        "title": "🍁 Guelph Ontario",
        "description": "Vegetation cooling & temperature distribution in Guelph.",
        "prompt": "Evaluate land surface temperature profile in Guelph, Ontario.",
        "city": "Guelph"
    }
]


@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "GeoAgent", "version": "1.0.0"}


@app.get("/api/presets")
def get_presets() -> List[Dict[str, Any]]:
    return PRESET_SCENARIOS


@app.post("/api/chat")
def handle_chat(req: ChatRequest) -> Dict[str, Any]:
    if not req.message.strip():
        logger.warning("Empty message rejected by /api/chat endpoint.")
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    logger.info(f"📨 [API Chat] Received request: '{req.message}'")
    result = agent_planner.plan_and_execute(req.message)
    logger.info(f"📤 [API Chat] Responding with type='{result.get('type')}', city='{result.get('city_name', 'N/A')}'")
    return result


# Mount Static Files (Frontend UI)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8088, reload=True)
