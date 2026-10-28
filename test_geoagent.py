"""
Unit Tests for GeoAgent Autonomous Spatial Intelligence Platform.
Ensures stability of boundary geocoding, thermal calculations,
agent planning, and API endpoints.
"""

import unittest
from fastapi.testclient import TestClient

from main import app
from gee_engine import eo_engine
from agent import agent_planner


class TestGeoAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check(self):
        """Verifies API health endpoint."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["system"], "GeoAgent")

    def test_get_presets(self):
        """Verifies preset scenarios endpoint."""
        response = self.client.get("/api/presets")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 3)
        self.assertIn("bandung_heat", [p["id"] for p in data])

    def test_geocode_boundary(self):
        """Verifies city boundary resolution."""
        res = eo_engine.geocode_city_boundary("Kota Bandung")
        self.assertTrue(res["success"])
        self.assertIn("bbox", res)
        self.assertIsNotNone(res["geojson"])

    def test_thermal_processing(self):
        """Verifies thermal Land Surface Temperature processing."""
        boundary = eo_engine.geocode_city_boundary("Kota Bandung")
        res = eo_engine.process_land_surface_temperature(
            boundary_info=boundary,
            start_date="2023-06-01",
            end_date="2023-09-30",
            cloud_threshold=20
        )
        self.assertEqual(res["status"], "success")
        self.assertIn("mean_temp", res)
        self.assertGreater(res["max_temp"], res["min_temp"])

    def test_agent_clarification(self):
        """Verifies proactive agent clarification when city is omitted."""
        res = agent_planner.plan_and_execute("Analyze the surface temperature profile")
        self.assertEqual(res["type"], "clarification")
        self.assertIn("Which city", res["message"])

    def test_agent_execution_with_audit_trace(self):
        """Verifies complete agent execution with transparent audit trace."""
        res = agent_planner.plan_and_execute("Analyze land surface temperature in Kota Bandung dry season 2023")
        self.assertEqual(res["type"], "analysis_result")
        self.assertIn("Bandung", res["city_name"])
        self.assertIn("audit_trace", res)
        self.assertGreaterEqual(len(res["audit_trace"]), 3)

    def test_agent_sumedang_resolution_not_medan(self):
        """Verifies that 'Sumedang, west java' resolves to Sumedang and NEVER falsely matches 'Medan'."""
        res = agent_planner.plan_and_execute("analyze temperature of Sumedang, west java")
        self.assertEqual(res["type"], "analysis_result")
        self.assertIn("Sumedang", res["city_name"])
        self.assertNotIn("Medan", res["city_name"])

    def test_agent_spatial_disambiguation(self):
        """Verifies proactive spatial disambiguation when a broad city name matches multiple administrative regions."""
        res = agent_planner.plan_and_execute("Analyze temperature in Bandung")
        self.assertIn(res["type"], ["disambiguation", "analysis_result"])
        if res["type"] == "disambiguation":
            self.assertIn("candidates", res)
            self.assertGreaterEqual(len(res["candidates"]), 2)

    def test_chat_api_endpoint(self):
        """Verifies end-to-end chat endpoint."""
        response = self.client.post("/api/chat", json={"message": "Analyze thermal profile in Guelph, Ontario"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["type"], "analysis_result")
        self.assertIn("Guelph", data["city_name"])

    def test_suhi_utfvi_processing(self):
        """Verifies Urban Thermal Field Variance Index (UTFVI) GEE processing."""
        boundary = eo_engine.geocode_city_boundary("Sumedang")
        res = eo_engine.process_suhi_utfvi(
            boundary_info=boundary,
            start_date="2023-06-01",
            end_date="2023-09-30",
            cloud_threshold=20
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["analysis_type"], "utfvi")
        self.assertIn("min_utfvi", res)
        self.assertIn("severe_heat_pct", res)
        self.assertIn("classes", res)

    def test_timeseries_processing(self):
        """Verifies multi-temporal LST timeseries GEE processing."""
        boundary = eo_engine.geocode_city_boundary("DKI Jakarta")
        res = eo_engine.process_temporal_timeseries(
            boundary_info=boundary,
            start_year=2020,
            end_year=2023,
            cloud_threshold=25
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["analysis_type"], "timeseries")
        self.assertIn("timeseries", res)
        self.assertGreaterEqual(len(res["timeseries"]), 2)
        self.assertIn("warming_rate", res)

    def test_agent_timeseries_tool_calling(self):
        """Verifies agent routes multi-year trend queries to tool_timeseries_pipeline."""
        res = agent_planner.plan_and_execute("Show temperature trend in Jakarta from 2019 to 2023")
        self.assertEqual(res["type"], "analysis_result")
        self.assertEqual(res.get("analysis_type"), "timeseries")
        self.assertIn("Jakarta", res["city_name"])
        self.assertIn("timeseries_data", res)
        self.assertIn("warming_rate", res)


if __name__ == "__main__":
    unittest.main()
