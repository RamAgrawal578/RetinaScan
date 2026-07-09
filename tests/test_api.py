"""
Flask route and REST API tests. Run against the app factory with an
isolated, temp-directory configuration (see conftest.py) — no real
dataset or trained model is required.
"""
from __future__ import annotations


class TestPageRoutes:
    def test_index_page_loads(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"RetinaAI" in response.data or b"Retina" in response.data

    def test_predict_form_loads(self, client):
        response = client.get("/predict")
        assert response.status_code == 200

    def test_about_page_loads(self, client):
        response = client.get("/about")
        assert response.status_code == 200

    def test_contact_page_loads(self, client):
        response = client.get("/contact")
        assert response.status_code == 200

    def test_model_info_page_loads(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_unknown_route_returns_404(self, client):
        response = client.get("/this-route-does-not-exist")
        assert response.status_code == 404


class TestApiRoutes:
    def test_health_endpoint_returns_json(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["status"] == "ok"
        assert "model_loaded" in payload

    def test_model_info_endpoint_returns_json(self, client):
        response = client.get("/api/model-info")
        assert response.status_code == 200
        payload = response.get_json()
        assert "architecture" in payload
        assert "class_names" in payload

    def test_version_endpoint_returns_json(self, client):
        response = client.get("/api/version")
        assert response.status_code == 200
        payload = response.get_json()
        assert "app_version" in payload

    def test_metrics_endpoint_returns_404_when_no_report(self, client):
        response = client.get("/api/metrics")
        assert response.status_code == 404
        payload = response.get_json()
        assert payload["available"] is False

    def test_predict_endpoint_without_file_returns_400(self, client):
        response = client.post("/api/predict", data={})
        assert response.status_code == 400
        payload = response.get_json()
        assert payload["success"] is False

    def test_unknown_api_route_returns_json_404(self, client):
        response = client.get("/api/this-does-not-exist")
        assert response.status_code == 404
        payload = response.get_json()
        assert payload["success"] is False
